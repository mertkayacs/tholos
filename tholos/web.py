import asyncio
import hashlib
import hmac
import html
import inspect
import ipaddress
import json
import logging
import os
import re
import secrets
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, urlsplit

import httpx
from jinja2 import pass_context
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from tholos import db as db_module
from tholos import prompt, rules, runner, tools, worker
from tholos import workspace as w

log = logging.getLogger("tholos.web")
STATIC = Path(__file__).parent / "static"
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

COOKIE = "tholos_session"
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'self'; "
    "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; "
    "form-action 'self'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
}
TELEGRAM_TOKEN = "telegram.bot_token"
TELEGRAM_CHAT = "telegram.owner_chat_id"
TEAMS = [("research-desk", "Research desk"), ("price-watch", "Price watch")]


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def markdown(text: str) -> str:
    escaped = html.escape(text)
    out: list[str] = []
    in_code = False
    in_list = False
    for line in escaped.splitlines():
        if line.strip().startswith("```"):
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append("</code></pre>" if in_code else "<pre><code>")
            in_code = not in_code
            continue
        if in_code:
            out.append(line)
            continue
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline(line[2:])}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        heading = re.match(r"^(#{1,3}) (.*)$", line)
        if heading:
            out.append(f"<h{len(heading[1])}>{_inline(heading[2])}</h{len(heading[1])}>")
        elif line.strip():
            out.append(f"<p>{_inline(line)}</p>")
    if in_list:
        out.append("</ul>")
    if in_code:
        out.append("</code></pre>")
    return "\n".join(out)


def _inline(text: str) -> str:
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*\n]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*\n]+)\*", r"<em>\1</em>", text)
    return re.sub(
        r"\[([^]\n]+)\]\((https?://[^\s)]+)\)",
        r'<a href="\2" rel="noopener noreferrer">\1</a>',
        text,
    )


def _sign(secret: str, payload: str) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def _session_value(secret: str, csrf: str, authed: bool) -> str:
    payload = f"{csrf}.{'1' if authed else '0'}"
    return f"{payload}.{_sign(secret, payload)}"


def _read_session(secret: str, value: str) -> dict | None:
    try:
        payload, sig = value.rsplit(".", 1)
        csrf, authed = payload.rsplit(".", 1)
    except ValueError:
        return None
    if not hmac.compare_digest(sig, _sign(secret, payload)):
        return None
    return {"csrf": csrf, "authed": authed == "1"}


def _set_session_cookie(request: Request, response: Response, session: dict, authed: bool) -> None:
    response.set_cookie(
        COOKIE,
        _session_value(request.app.state.secret, session["csrf"], authed),
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
    )


async def _csrf_error(request: Request, session: dict) -> str | None:
    host = request.headers.get("host", "")
    source = request.headers.get("origin") or request.headers.get("referer") or ""
    origin = urlsplit(source).netloc if source else ""
    if not origin or not host or origin.lower() != host.lower():
        return "The request origin does not match this server. Reload the page and try again."
    if request.url.path == "/login":
        return None
    token = request.headers.get("x-csrf-token")
    content_type = request.headers.get("content-type", "")
    if token is None and content_type.startswith(
        ("application/x-www-form-urlencoded", "multipart/form-data")
    ):
        form = await request.form()
        token = form.get("csrf_token")
    if not token or not hmac.compare_digest(str(token), session["csrf"]):
        return "The CSRF token is missing or wrong. Reload the page and try again."
    return None


class GuardMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/healthz" and request.method in {"GET", "HEAD"}:
            return await call_next(request)
        state = request.app.state
        if state.loopback:
            hosts = request.headers.getlist("host")
            if len(hosts) != 1 or not re.fullmatch(
                r"(?:127\.0\.0\.1|localhost|\[::1\])(?::[0-9]+)?|::1", hosts[0], re.IGNORECASE
            ):
                port = (request.scope.get("server") or ("127.0.0.1", 7070))[1]
                return PlainTextResponse(
                    f"This address is not allowed. Open http://127.0.0.1:{port} instead.",
                    403,
                    headers=SECURITY_HEADERS,
                )
        session = _read_session(state.secret, request.cookies.get(COOKIE, ""))
        path = request.url.path
        authed = state.loopback or bool(session and session["authed"])
        if not authed and path != "/login" and not path.startswith("/static/"):
            if request.method == "GET":
                return RedirectResponse("/login", 303)
            return PlainTextResponse("Log in first.", 403)
        if session is None:
            session = {"csrf": secrets.token_urlsafe(24), "authed": authed}
            request.state.new_session = True
        request.state.session = session
        if request.method in {"POST", "PUT", "DELETE"}:
            error = await _csrf_error(request, session)
            if error:
                return PlainTextResponse(error, 403)
        response = await call_next(request)
        for key, value in SECURITY_HEADERS.items():
            response.headers.setdefault(key, value)
        if getattr(request.state, "new_session", False):
            _set_session_cookie(request, response, session, session["authed"])
        return response


async def _maybe(value) -> None:
    if inspect.isawaitable(value):
        await value


async def _restart_telegram(app: Starlette) -> None:
    poller = getattr(app.state, "telegram", None)
    if poller:
        await _maybe(poller.stop())
    app.state.telegram = None
    db = app.state.db
    token, chat_id = w.get_setting(db, TELEGRAM_TOKEN), w.get_setting(db, TELEGRAM_CHAT)
    if token and chat_id:
        from tholos import telegram

        app.state.telegram = telegram.Poller(db, token, chat_id, wake=app.state.worker.wake)
        await app.state.telegram.start()


@asynccontextmanager
async def lifespan(app: Starlette):
    db = db_module.connect()
    db_module.init(db)
    app.state.db = db
    app.state.sse_active = 0
    app.state.host = os.environ.get("THOLOS_HOST", "127.0.0.1")
    app.state.loopback = is_loopback(app.state.host)
    secret = w.get_setting(db, "secret_key")
    if not secret:
        secret = secrets.token_hex(32)
        w.set_setting(db, "secret_key", secret)
    app.state.secret = secret
    if not app.state.loopback and not w.get_setting(db, "access_token"):
        w.set_setting(db, "access_token", secrets.token_urlsafe(24))
    app.state.worker = worker.Worker(db)
    await _maybe(app.state.worker.start())
    await _restart_telegram(app)
    try:
        yield
    finally:
        if app.state.telegram:
            await _maybe(app.state.telegram.stop())
        await _maybe(app.state.worker.stop())
        db.close()


@pass_context
def _localtime(ctx, value: str | None) -> str:
    if not value:
        return ""
    try:
        dt = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        return value
    return dt.astimezone(ctx["tz"]).strftime("%d %b %H:%M")


templates.env.filters["localtime"] = _localtime
templates.env.filters["urlquote"] = lambda value: quote(str(value), safe="")
templates.env.filters["md"] = markdown
templates.env.globals["tool_names"] = sorted(tools.SPECS)


def db_of(request: Request):
    return request.app.state.db


def _agent_or_404(request: Request) -> dict:
    agent = w.get_agent(db_of(request), request.path_params["name"])
    if agent is None:
        raise HTTPException(404)
    return agent


def _run_or_404(request: Request) -> dict:
    run = w.get_run(db_of(request), int(request.path_params["id"]))
    if run is None:
        raise HTTPException(404)
    return run


def render(request: Request, template: str, status: int = 200, **ctx) -> Response:
    db = db_of(request)
    base = {
        "request": request,
        "csrf": request.state.session["csrf"],
        "page": ctx.pop("page", ""),
        "waiting_count": len(w.list_waiting(db)),
        "tz": w.timezone(db),
        "loopback": request.app.state.loopback,
    }
    return templates.TemplateResponse(request, template, base | ctx, status_code=status)


def toast(response: Response, message: str) -> Response:
    response.headers["HX-Trigger"] = json.dumps({"toast": message})
    return response


def hx_redirect(location: str) -> Response:
    return Response(status_code=200, headers={"HX-Redirect": location})


def wake(request: Request) -> None:
    request.app.state.worker.wake()


# Board


def board_ctx(db) -> dict:
    agents = w.list_agents(db)
    names = {agent["id"]: agent["name"] for agent in agents}
    schedules = [s for s in w.list_schedules(db) if s["enabled"]]
    schedules.sort(key=lambda s: s["next_at"])
    for schedule in schedules:
        schedule["agent"] = names.get(schedule["agent_id"], "?")
    follow_ups = [
        r for r in w.list_runs(db, status="queued", limit=50) if r["trigger_kind"] == "follow_up"
    ]
    running = w.list_runs(db, status="running", limit=50)
    for run in running:
        step = db.execute(
            "SELECT thought, tool FROM steps WHERE run_id=? ORDER BY n DESC LIMIT 1", (run["id"],)
        ).fetchone()
        run["step"] = dict(step) if step else None
    done = w.list_runs(db, status="done", limit=10) + [
        r for r in w.list_runs(db, status="failed", limit=10)
    ]
    done.sort(key=lambda r: r["id"], reverse=True)
    waiting = w.list_waiting(db)
    for card in waiting:
        try:
            card["target"] = rules.target(card["tool"], card["args"])
        except KeyError:
            card["target"] = "*"
        card["args_json"] = json.dumps(card["args"], indent=2)
    return {
        "agents": agents,
        "schedules": schedules,
        "follow_ups": follow_ups,
        "running": running,
        "waiting": waiting,
        "done_runs": done[:12],
        "done_tasks": w.list_tasks(db, status="done", limit=10),
        "has_model": bool(w.list_models(db)),
        "has_agents": bool(agents),
    }


async def board(request: Request) -> Response:
    return render(request, "board.html", page="board", **board_ctx(db_of(request)))


async def board_lane(request: Request) -> Response:
    lane = request.path_params["lane"]
    if lane not in {"scheduled", "working", "waiting", "done"}:
        raise HTTPException(404)
    return render(request, f"partials/{lane}.html", **board_ctx(db_of(request)))


def _board_partial(request: Request, partial: str, message: str) -> Response:
    return toast(render(request, partial, **board_ctx(db_of(request))), message)


async def task_new(request: Request) -> Response:
    db = db_of(request)
    form = await request.form()
    title = str(form.get("title", "")).strip()
    to = str(form.get("to", "")).strip()
    details = str(form.get("details", "")).strip()
    if not title:
        return _board_partial(request, "partials/working.html", "Give the task a title.")
    w.add_task(db, title, details, to=to if to and to != "you" else None)
    wake(request)
    return _board_partial(request, "partials/working.html", "Task added.")


async def team_load(request: Request) -> Response:
    return await _load_team(request, "partials/working.html")


# Approvals


def _pending(request: Request, approval_id: int) -> dict | None:
    row = (
        db_of(request)
        .execute("SELECT * FROM approvals WHERE id=? AND status='pending'", (approval_id,))
        .fetchone()
    )
    return dict(row) if row else None


def _waiting_page(request: Request, message: str) -> Response:
    return _board_partial(request, "partials/waiting.html", message)


async def approval_decide(request: Request) -> Response:
    db = db_of(request)
    approval_id = int(request.path_params["id"])
    action = str((await request.form()).get("action", ""))
    if action not in {"approve", "deny", "always"}:
        raise HTTPException(404)
    if _pending(request, approval_id):
        runner.decide(db, approval_id, approve=action != "deny", always=action == "always")
        wake(request)
        message = {"approve": "Approved.", "deny": "Denied.", "always": "Always allowed."}[action]
    else:
        message = "That approval was already decided."
    return _waiting_page(request, message)


async def approval_answer(request: Request) -> Response:
    db = db_of(request)
    approval_id = int(request.path_params["id"])
    text = str((await request.form()).get("text", "")).strip()
    if not text:
        message = "Add an answer first."
    elif _pending(request, approval_id):
        runner.answer(db, approval_id, text)
        wake(request)
        message = "Answer sent."
    else:
        message = "That question was already answered."
    return _waiting_page(request, message)


# Agents


async def agents(request: Request) -> Response:
    return render(request, "agents.html", page="agents", agents=w.list_agents(db_of(request)))


def _agent_form_ctx(db, agent: dict | None, error: str = "") -> dict:
    return {
        "agent": agent,
        "models": w.list_models(db),
        "error": error,
    }


def _agent_form_values(form) -> dict:
    max_steps_raw = str(form.get("max_steps", "12")).strip()
    return {
        "name": str(form.get("name", "")).strip(),
        "role": str(form.get("role", "")).strip(),
        "model_id": int(form["model_id"]) if form.get("model_id") else None,
        "tools": [str(t) for t in form.getlist("tools")],
        "max_steps": int(max_steps_raw) if max_steps_raw.isdigit() else 0,
        "paused": form.get("paused") == "1",
    }


async def agent_new_page(request: Request) -> Response:
    return render(request, "agent_new.html", page="agents", **_agent_form_ctx(db_of(request), None))


async def agent_create(request: Request) -> Response:
    db = db_of(request)
    form = await request.form()
    values = _agent_form_values(form)
    try:
        agent_id = w.save_agent(db, None, **values)
    except ValueError as exc:
        return render(
            request, "partials/agent_form.html", **_agent_form_ctx(db, None, str(exc)) | values
        )
    return hx_redirect(f"/agents/{quote(w.get_agent(db, agent_id)['name'])}")


async def agent_page(request: Request) -> Response:
    db = db_of(request)
    agent = _agent_or_404(request)
    return render(
        request,
        "agent.html",
        page="agents",
        memories=w.list_memories(db, agent["id"]),
        runs=w.list_runs(db, agent_id=agent["id"], limit=10),
        schedules=w.list_schedules(db, agent["id"]),
        error_id="",
        values={},
        **_agent_form_ctx(db, agent),
    )


async def agent_update(request: Request) -> Response:
    db = db_of(request)
    agent = _agent_or_404(request)
    values = _agent_form_values(await request.form()) | {"name": agent["name"]}
    try:
        w.save_agent(db, agent["id"], **values)
    except ValueError as exc:
        ctx = _agent_form_ctx(db, agent | values, str(exc))
        return render(request, "partials/agent_form.html", **ctx)
    return toast(
        render(
            request,
            "partials/agent_form.html",
            **_agent_form_ctx(db, w.get_agent(db, agent["id"])),
        ),
        "Agent saved.",
    )


def _schedules_page(
    request: Request,
    agent: dict,
    message: str = "",
    error: str = "",
    error_id: str = "",
    values: dict | None = None,
) -> Response:
    db = db_of(request)
    response = render(
        request,
        "partials/schedules.html",
        agent=agent,
        schedules=w.list_schedules(db, agent["id"]),
        error=error,
        error_id=str(error_id),
        values=values or {},
    )
    return toast(response, message) if message else response


def _schedule_form(form) -> tuple[str, str, bool]:
    return (
        str(form.get("every", "")).strip(),
        str(form.get("prompt", "")).strip(),
        form.get("enabled") == "1",
    )


async def schedule_add(request: Request) -> Response:
    db = db_of(request)
    agent = _agent_or_404(request)
    every, prompt_text, enabled = _schedule_form(await request.form())
    try:
        w.save_schedule(db, None, agent["id"], every, prompt_text, enabled)
    except ValueError as exc:
        return _schedules_page(
            request, agent, error=str(exc), error_id="add",
            values={"every": every, "prompt": prompt_text},
        )
    return _schedules_page(request, agent, "Schedule saved.")


def _schedule_or_404(request: Request) -> dict:
    db = db_of(request)
    schedule_id = int(request.path_params["id"])
    schedule = next((s for s in w.list_schedules(db) if s["id"] == schedule_id), None)
    if schedule is None:
        raise HTTPException(404)
    return schedule


async def schedule_edit(request: Request) -> Response:
    db = db_of(request)
    schedule = _schedule_or_404(request)
    agent = w.get_agent(db, schedule["agent_id"])
    every, prompt_text, enabled = _schedule_form(await request.form())
    try:
        w.save_schedule(db, schedule["id"], agent["id"], every, prompt_text, enabled)
    except ValueError as exc:
        return _schedules_page(request, agent, error=str(exc), error_id=str(schedule["id"]))
    return _schedules_page(request, agent, "Schedule saved.")


async def schedule_delete(request: Request) -> Response:
    db = db_of(request)
    schedule = _schedule_or_404(request)
    w.delete_schedule(db, schedule["id"])
    return _schedules_page(request, w.get_agent(db, schedule["agent_id"]), "Schedule deleted.")


async def schedule_run(request: Request) -> Response:
    db = db_of(request)
    schedule = _schedule_or_404(request)
    w.queue_run(db, schedule["agent_id"], prompt.schedule_trigger(schedule["prompt"]), "schedule")
    wake(request)
    return _schedules_page(request, w.get_agent(db, schedule["agent_id"]), "Run queued.")


def _memories_page(request: Request, agent: dict) -> Response:
    db = db_of(request)
    return render(
        request,
        "partials/memories.html",
        agent=agent,
        memories=w.list_memories(db, agent["id"]),
    )


async def memory_add(request: Request) -> Response:
    db = db_of(request)
    agent = _agent_or_404(request)
    text = str((await request.form()).get("text", "")).strip()
    if text:
        w.add_memory(db, agent["id"], text[:200], "you")
    return _memories_page(request, agent)


def _memory_agent(request: Request) -> tuple[dict | None, dict | None]:
    db = db_of(request)
    memory_id = int(request.path_params["id"])
    for agent in w.list_agents(db):
        memory = next((m for m in w.list_memories(db, agent["id"]) if m["id"] == memory_id), None)
        if memory:
            return agent, memory
    return None, None


async def memory_edit(request: Request) -> Response:
    agent, memory = _memory_agent(request)
    if memory is None:
        raise HTTPException(404)
    text = str((await request.form()).get("text", "")).strip()
    if text:
        w.update_memory(db_of(request), memory["id"], text[:200])
    return _memories_page(request, agent)


async def memory_delete(request: Request) -> Response:
    agent, memory = _memory_agent(request)
    if memory is None:
        raise HTTPException(404)
    w.delete_memory(db_of(request), memory["id"])
    return _memories_page(request, agent)


# Runs


def _run_ctx(db, run: dict) -> dict:
    agent = w.get_agent(db, run["agent_id"])
    run = run | {"agent": agent["name"] if agent else "?"}
    started = run.get("started_at")
    ended = run.get("ended_at")
    duration = ""
    if started:
        begin = datetime.strptime(started, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        end = (
            datetime.strptime(ended, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
            if ended
            else datetime.now(UTC)
        )
        seconds = max(0, int((end - begin).total_seconds()))
        duration = f"{seconds // 60}m {seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"
    return {"run": run, "duration": duration}


async def run_page(request: Request) -> Response:
    run = _run_or_404(request)
    return render(request, "run.html", page="board", **_run_ctx(db_of(request), run))


async def run_stop(request: Request) -> Response:
    db = db_of(request)
    run = _run_or_404(request)
    w.stop_run(db, run["id"])
    wake(request)
    return toast(
        render(request, "partials/run_header.html", **_run_ctx(db, w.get_run(db, run["id"]))),
        "Run stopped.",
    )


# Tables


async def tables(request: Request) -> Response:
    return render(
        request, "tables.html", page="tables", tables=w.list_tables(db_of(request)), error=""
    )


async def table_create(request: Request) -> Response:
    db = db_of(request)
    form = await request.form()
    name = str(form.get("name", "")).strip()
    columns = [c.strip() for c in str(form.get("columns", "")).split(",") if c.strip()]
    try:
        w.create_table(db, name, columns, "you")
    except ValueError as exc:
        return render(
            request,
            "partials/table_new.html",
            error=str(exc),
            name=name,
            columns=str(form.get("columns", "")),
        )
    return hx_redirect(f"/tables/{quote(name)}")


def _table_ctx(db, name: str) -> dict:
    sheet = w.get_table(db, name)
    if sheet is None:
        raise HTTPException(404)
    changes = [
        dict(row)
        for row in db.execute(
            "SELECT * FROM history WHERE kind='row' AND (json_extract(\"before\",'$.sheet_id')=? "
            "OR json_extract(\"after\",'$.sheet_id')=?) ORDER BY id DESC LIMIT 10",
            (sheet["id"], sheet["id"]),
        )
    ]
    return {"sheet": sheet, "totals": _totals(sheet), "changes": changes}


def _totals(sheet: dict) -> dict:
    totals = {}
    for column in sheet["columns"]:
        numbers = []
        for row in sheet["rows"]:
            value = row["data"].get(column, "")
            if value in ("", None):
                continue
            try:
                numbers.append(float(value))
            except (TypeError, ValueError):
                numbers = []
                break
        if numbers:
            totals[column] = sum(numbers)
    return totals


async def table_page(request: Request) -> Response:
    return render(
        request,
        "table.html",
        page="tables",
        **_table_ctx(db_of(request), request.path_params["name"]),
    )


def _grid(request: Request, name: str, **extra) -> Response:
    return render(
        request, "partials/table_grid.html", **_table_ctx(db_of(request), name) | extra
    )


async def table_grid(request: Request) -> Response:
    return _grid(request, request.path_params["name"])


def _int_or(form, key: str, default: int = -1) -> int:
    try:
        return int(str(form.get(key, "")))
    except ValueError:
        return default


async def table_row_add(request: Request) -> Response:
    db = db_of(request)
    name = request.path_params["name"]
    try:
        w.add_rows(db, name, [{}], "you")
    except ValueError as exc:
        return _grid(request, name, error=str(exc))
    return _grid(request, name)


async def table_cell_edit(request: Request) -> Response:
    db = db_of(request)
    name, row_id = request.path_params["name"], int(request.path_params["row_id"])
    form = await request.form()
    column, value = str(form.get("column", "")), str(form.get("value", ""))
    try:
        w.update_row(
            db, name, row_id, {column: value}, "you",
            expected_version=_int_or(form, "expected_version"),
        )
    except w.Conflict as exc:
        return _grid(request, name, conflict={"value": exc.current["data"].get(column, "")})
    except ValueError as exc:
        return _grid(request, name, error=str(exc))
    return _grid(request, name)


async def table_row_delete(request: Request) -> Response:
    db = db_of(request)
    name = request.path_params["name"]
    try:
        w.delete_row(db, name, int(request.path_params["row_id"]), "you")
    except ValueError as exc:
        return _grid(request, name, error=str(exc))
    return _grid(request, name)


async def table_export(request: Request) -> Response:
    name, fmt = request.path_params["name"], request.path_params["fmt"]
    try:
        data = w.export_table(db_of(request), name, fmt)
    except ValueError:
        raise HTTPException(404) from None
    media = (
        "text/csv; charset=utf-8"
        if fmt == "csv"
        else ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    )
    return Response(
        data,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{name}.{fmt}"'},
    )


# Notes


async def notes(request: Request) -> Response:
    return render(request, "notes.html", page="notes", notes=w.list_notes(db_of(request)), error="")


async def note_create(request: Request) -> Response:
    db = db_of(request)
    title = str((await request.form()).get("title", "")).strip()
    if not title:
        return render(request, "partials/note_new.html", error="Give the note a title.")
    if w.get_note(db, title) is None:
        w.write_note(db, title, "", "you")
    return hx_redirect(f"/notes/{quote(title)}")


def _note_ctx(db, title: str) -> dict:
    note = w.get_note(db, title)
    if note is None:
        raise HTTPException(404)
    return {
        "note": note,
        "versions": w.recent_changes(db, "note", note["id"], 10),
    }


def _note_render(request: Request, template: str, **extra) -> Response:
    return render(
        request, template, **_note_ctx(db_of(request), request.path_params["title"]) | extra
    )


async def note_page(request: Request) -> Response:
    return _note_render(request, "note.html", page="notes")


async def note_editor(request: Request) -> Response:
    return _note_render(request, "partials/note_editor.html")


async def note_save(request: Request) -> Response:
    db = db_of(request)
    title = request.path_params["title"]
    form = await request.form()
    text = str(form.get("text", ""))
    try:
        w.write_note(db, title, text, "you", "replace", _int_or(form, "expected_version"))
    except w.Conflict:
        ctx = _note_ctx(db, title)
        ctx["note"] = ctx["note"] | {"body": text}
        return render(request, "partials/note_editor.html", conflict=True, **ctx)
    return toast(_note_render(request, "partials/note_editor.html"), "Note saved.")


# Activity


def _events(db) -> list[dict]:
    return list(reversed(w.events_since(db, 0, 200)))


async def activity(request: Request) -> Response:
    return render(request, "activity.html", page="activity", events=_events(db_of(request)))


async def activity_rows(request: Request) -> Response:
    return render(request, "partials/activity_rows.html", events=_events(db_of(request)))


# Settings


def _settings_ctx(db, request: Request, **extra) -> dict:
    token = w.get_setting(db, TELEGRAM_TOKEN)
    ctx = {
        "models": [
            m | {"api_key_masked": ("····" + m["api_key"][-4:]) if m["api_key"] else ""}
            for m in w.list_models(db)
        ],
        "rules": w.list_rules(db),
        "teams": TEAMS,
        "telegram": {
            "configured": bool(token),
            "token_masked": ("····" + str(token)[-4:]) if token else "",
            "owner_chat_id": w.get_setting(db, TELEGRAM_CHAT) or "",
        },
        "access_token": None if request.app.state.loopback else w.get_setting(db, "access_token"),
        "model_error": "",
        "rule_error": "",
    }
    return ctx | extra


def _settings_page(request: Request, partial: str, message: str = "", **extra) -> Response:
    response = render(
        request, f"partials/{partial}.html", **_settings_ctx(db_of(request), request, **extra)
    )
    return toast(response, message) if message else response


async def settings(request: Request) -> Response:
    return render(
        request, "settings.html", page="settings", **_settings_ctx(db_of(request), request)
    )


def _model_values(form) -> dict:
    return {
        "name": str(form.get("name", "")).strip(),
        "base_url": str(form.get("base_url", "")).strip(),
        "model": str(form.get("model", "")).strip(),
        "json_mode": str(form.get("json_mode", "schema")),
        "temperature": float(form.get("temperature") or 0.2),
        "max_tokens": int(str(form.get("max_tokens") or 512)),
    }


async def model_save(request: Request) -> Response:
    db = db_of(request)
    form = await request.form()
    model_id = int(form["id"]) if form.get("id") else None
    try:
        values = _model_values(form)
    except ValueError:
        return _settings_page(
            request,
            "settings_models",
            model_error="Temperature and max tokens must be numbers.",
        )
    api_key = str(form.get("api_key", "")) or (
        w.get_model(db, model_id)["api_key"] if model_id and w.get_model(db, model_id) else None
    )
    try:
        w.save_model(db, model_id, **values | {"api_key": api_key})
    except ValueError as exc:
        return _settings_page(request, "settings_models", model_error=str(exc))
    return _settings_page(request, "settings_models", "Model saved.")


async def model_delete(request: Request) -> Response:
    w.delete_model(db_of(request), int(request.path_params["id"]))
    return _settings_page(request, "settings_models", "Model deleted.")


async def model_test(request: Request) -> Response:
    db = db_of(request)
    model = w.get_model(db, int(request.path_params["id"]))
    if model is None:
        raise HTTPException(404)
    payload = {
        "model": model["model"],
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {model['api_key']}"} if model.get("api_key") else {}
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
            response = await client.post(
                model["base_url"].rstrip("/") + "/chat/completions",
                json=payload,
                headers=headers,
            )
        if response.status_code >= 400:
            ok, message = False, f"HTTP {response.status_code}: {response.text[:200]}"
        else:
            ok, message = True, f"The model replied in {time.monotonic() - start:.1f}s."
    except httpx.HTTPError as exc:
        ok, message = False, str(exc)
    result = {"id": model["id"], "ok": ok, "message": message}
    return _settings_page(request, "settings_models", test_result=result)


async def model_detect(request: Request) -> Response:
    found = []
    for base in ("http://127.0.0.1:11434", "http://127.0.0.1:8080"):
        try:
            async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
                response = await client.get(base + "/v1/models")
            if response.status_code == 200:
                for item in response.json().get("data", []):
                    if item.get("id"):
                        found.append({"base_url": base + "/v1", "model": item["id"]})
        except (httpx.HTTPError, ValueError):
            continue
    return render(request, "partials/detect_results.html", found=found)


async def rule_add(request: Request) -> Response:
    db = db_of(request)
    form = await request.form()
    try:
        w.add_rule(
            db,
            str(form.get("tool", "")),
            str(form.get("decision", "")),
            str(form.get("agent", "")).strip() or "*",
            str(form.get("match", "")).strip() or "*",
        )
    except ValueError as exc:
        return _settings_page(request, "settings_rules", rule_error=str(exc))
    return _settings_page(request, "settings_rules", "Rule saved.")


async def rule_delete(request: Request) -> Response:
    w.delete_rule(db_of(request), int(request.path_params["id"]))
    return _settings_page(request, "settings_rules", "Rule deleted.")


async def _load_team(request: Request, partial: str) -> Response:
    db = db_of(request)
    name = str((await request.form()).get("name", ""))
    try:
        team = w.load_team(db, name)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        message = f"Could not load the team: {exc}"
    else:
        wake(request)
        message = f"Loaded the {team.get('name', name)} team."
    if partial == "settings_teams":
        return _settings_page(request, partial, message)
    return _board_partial(request, partial, message)


async def team_load_settings(request: Request) -> Response:
    return await _load_team(request, "settings_teams")


async def telegram_save(request: Request) -> Response:
    db = db_of(request)
    form = await request.form()
    token = str(form.get("bot_token", "")).strip()
    chat_id = str(form.get("owner_chat_id", "")).strip()
    if token:
        w.set_setting(db, TELEGRAM_TOKEN, token)
    if chat_id:
        w.set_setting(db, TELEGRAM_CHAT, chat_id)
    await _restart_telegram(request.app)
    return _settings_page(request, "settings_telegram", "Telegram settings saved.")


async def telegram_test(request: Request) -> Response:
    db = db_of(request)
    token, chat_id = w.get_setting(db, TELEGRAM_TOKEN), w.get_setting(db, TELEGRAM_CHAT)
    if not token or not chat_id:
        result = {"ok": False, "message": "Save the bot token and chat id first."}
        return _settings_page(request, "settings_telegram", test_result=result)
    try:
        async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
            response = await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": "Tholos test message."},
            )
        data = response.json()
        if data.get("ok"):
            result = {"ok": True, "message": "Test message sent."}
        else:
            result = {
                "ok": False,
                "message": data.get("description", f"HTTP {response.status_code}"),
            }
    except (httpx.HTTPError, ValueError) as exc:
        result = {"ok": False, "message": str(exc)}
    return _settings_page(request, "settings_telegram", test_result=result)


# Auth and events


async def login_page(request: Request) -> Response:
    if request.app.state.loopback:
        return RedirectResponse("/", 303)
    return render(request, "login.html", error="")


async def login(request: Request) -> Response:
    expected = w.get_setting(db_of(request), "access_token") or ""
    form = await request.form()
    token = str(form.get("token", ""))
    if not expected or not hmac.compare_digest(token, expected):
        return render(
            request,
            "login.html",
            status=403,
            error="Wrong token. Check the terminal where Tholos started.",
        )
    session = request.state.session
    response = RedirectResponse("/", 303)
    _set_session_cookie(request, response, session, True)
    return response


async def events(request: Request) -> Response:
    db = db_of(request)
    try:
        since = int(request.headers.get("last-event-id") or request.query_params.get("since") or 0)
    except ValueError:
        since = 0

    async def stream():
        # The generator is cancelled by StreamingResponse when the client goes away.
        request.app.state.sse_active += 1
        last = since
        ping_at = time.monotonic() + 15
        try:
            while True:
                for row in w.events_since(db, last, 200):
                    last = row["seq"]
                    yield f"id: {row['seq']}\nevent: change\ndata: {w.dumps(row)}\n\n"
                if time.monotonic() >= ping_at:
                    yield ": ping\n\n"
                    ping_at = time.monotonic() + 15
                await asyncio.sleep(1)
        finally:
            request.app.state.sse_active -= 1

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def favicon(request: Request) -> Response:
    return RedirectResponse("/static/favicon.svg")


async def healthz(request: Request) -> Response:
    return PlainTextResponse("ok")


routes = [
    Route("/healthz", healthz),
    Route("/", board),
    Route("/login", login_page),
    Route("/login", login, methods=["POST"]),
    Route("/events", events),
    Route("/favicon.ico", favicon),
    Route("/partials/board/{lane}", board_lane),
    Route("/partials/activity", activity_rows),
    Route("/partials/tables/{name}/grid", table_grid),
    Route("/partials/notes/{title:path}/editor", note_editor),
    Route("/tasks", task_new, methods=["POST"]),
    Route("/teams/load", team_load, methods=["POST"]),
    Route("/approvals/{id}/decide", approval_decide, methods=["POST"]),
    Route("/approvals/{id}/answer", approval_answer, methods=["POST"]),
    Route("/agents", agents),
    Route("/agents/new", agent_new_page),
    Route("/agents", agent_create, methods=["POST"]),
    Route("/agents/{name}", agent_page),
    Route("/agents/{name}", agent_update, methods=["POST"]),
    Route("/agents/{name}/schedules", schedule_add, methods=["POST"]),
    Route("/agents/{name}/memories", memory_add, methods=["POST"]),
    Route("/schedules/{id}/edit", schedule_edit, methods=["POST"]),
    Route("/schedules/{id}/delete", schedule_delete, methods=["POST"]),
    Route("/schedules/{id}/run", schedule_run, methods=["POST"]),
    Route("/memories/{id}/edit", memory_edit, methods=["POST"]),
    Route("/memories/{id}/delete", memory_delete, methods=["POST"]),
    Route("/runs/{id}", run_page),
    Route("/runs/{id}/stop", run_stop, methods=["POST"]),
    Route("/tables", tables),
    Route("/tables", table_create, methods=["POST"]),
    Route("/tables/{name}.{fmt}", table_export),
    Route("/tables/{name}", table_page),
    Route("/tables/{name}/rows", table_row_add, methods=["POST"]),
    Route("/tables/{name}/rows/{row_id}", table_cell_edit, methods=["PUT"]),
    Route("/tables/{name}/rows/{row_id}/delete", table_row_delete, methods=["POST"]),
    Route("/notes", notes),
    Route("/notes", note_create, methods=["POST"]),
    Route("/notes/{title:path}", note_page),
    Route("/notes/{title:path}", note_save, methods=["POST"]),
    Route("/activity", activity),
    Route("/settings", settings),
    Route("/settings/models", model_save, methods=["POST"]),
    Route("/settings/models/detect", model_detect, methods=["POST"]),
    Route("/settings/models/{id}/delete", model_delete, methods=["POST"]),
    Route("/settings/models/{id}/test", model_test, methods=["POST"]),
    Route("/settings/rules", rule_add, methods=["POST"]),
    Route("/settings/rules/{id}/delete", rule_delete, methods=["POST"]),
    Route("/settings/teams", team_load_settings, methods=["POST"]),
    Route("/settings/telegram", telegram_save, methods=["POST"]),
    Route("/settings/telegram/test", telegram_test, methods=["POST"]),
    Mount("/static", StaticFiles(directory=STATIC), name="static"),
]

app = Starlette(
    routes=routes,
    lifespan=lifespan,
    middleware=[Middleware(GuardMiddleware)],
)
