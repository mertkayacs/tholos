import asyncio
import hashlib
import hmac
import logging
from contextlib import suppress

import httpx

from tholos import runner
from tholos import workspace as w

log = logging.getLogger("tholos.telegram")
API = "https://api.telegram.org"
POLL_TIMEOUT = 25


class Poller:
    def __init__(
        self,
        db,
        token: str,
        owner_chat_id: str,
        *,
        transport: httpx.BaseTransport | None = None,
        wake=None,
    ):
        self.db = db
        self.owner = str(owner_chat_id)
        self.client = httpx.AsyncClient(
            transport=transport,
            base_url=f"{API}/bot{token}",
            timeout=POLL_TIMEOUT + 15,
            trust_env=False,
        )
        secret = w.get_setting(db, "secret_key") or "tholos"
        self.secret = str(secret).encode()
        self.wake = wake
        self.offset = 0
        self.last_seq = 0
        self.notified_approvals: set[int] = set()
        self.question_messages: dict[int, int] = {}
        self.seen_runs: set[int] = set()
        self.seen_tasks: set[int] = set()
        self.ignored_chats: set[str] = set()
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()

    def sign(self, action: str, approval_id: int) -> str:
        msg = f"tg:{action}:{approval_id}".encode()
        return hmac.new(self.secret, msg, hashlib.sha256).hexdigest()[:12]

    def callback(self, action: str, approval_id: int) -> str:
        return f"{action}:{approval_id}:{self.sign(action, approval_id)}"

    def check_callback(self, data: str) -> tuple[str, int] | None:
        try:
            action, raw_id, sig = data.split(":", 2)
            approval_id = int(raw_id)
        except ValueError:
            return None
        if action not in {"a", "d", "w"}:
            return None
        if not hmac.compare_digest(sig, self.sign(action, approval_id)):
            return None
        return action, approval_id

    async def api(self, method: str, **data) -> dict:
        response = await self.client.post(f"/{method}", json=data)
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            raise ValueError(f"Telegram {method} failed: {payload.get('description', '?')}")
        return payload["result"]

    async def start(self) -> None:
        row = self.db.execute("SELECT coalesce(max(seq), 0) FROM events").fetchone()
        self.last_seq = row[0]
        self.seen_runs = {r["id"] for r in w.list_runs(self.db, status="failed", limit=50)}
        self.seen_tasks = {
            t["id"] for t in w.list_tasks(self.db, status="todo") if t["agent"] is None
        }
        self._stopping.clear()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stopping.set()
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        await self.client.aclose()

    async def _loop(self) -> None:
        while not self._stopping.is_set():
            try:
                updates = await self.api(
                    "getUpdates", timeout=POLL_TIMEOUT, offset=self.offset
                )
                for update in updates:
                    self.offset = max(self.offset, update["update_id"] + 1)
                    await self.handle_update(update)
                await self.notify()
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                log.warning("telegram poll failed: %s", exc)
                await asyncio.sleep(5)

    def _allowed(self, chat_id) -> bool:
        chat_id = str(chat_id)
        if chat_id == self.owner:
            return True
        if chat_id not in self.ignored_chats:
            self.ignored_chats.add(chat_id)
            log.info("ignoring messages from chat %s", chat_id)
        return False

    async def handle_update(self, update: dict) -> None:
        callback = update.get("callback_query")
        if callback:
            await self._handle_callback(callback)
            return
        message = update.get("message")
        if message and message.get("text"):
            await self._handle_message(message)

    async def _handle_callback(self, callback: dict) -> None:
        chat_id = (callback.get("message") or {}).get("chat", {}).get("id")
        if not self._allowed(chat_id):
            return
        parsed = self.check_callback(callback.get("data", ""))
        if parsed is None:
            await self.api(
                "answerCallbackQuery",
                callback_query_id=callback["id"],
                text="That button is not valid anymore.",
            )
            return
        action, approval_id = parsed
        row = self.db.execute(
            "SELECT status FROM approvals WHERE id=?", (approval_id,)
        ).fetchone()
        if not row or row[0] != "pending":
            await self.api(
                "answerCallbackQuery",
                callback_query_id=callback["id"],
                text="Already decided.",
            )
            return
        runner.decide(
            self.db, approval_id, approve=action != "d", always=action == "w"
        )
        if self.wake:
            self.wake()
        label = {"a": "Approved.", "d": "Denied.", "w": "Always allowed."}[action]
        await self.api("answerCallbackQuery", callback_query_id=callback["id"], text=label)

    async def _handle_message(self, message: dict) -> None:
        chat_id = message.get("chat", {}).get("id")
        if not self._allowed(chat_id):
            return
        text = message["text"].strip()
        reply_to = (message.get("reply_to_message") or {}).get("message_id")
        if reply_to is not None and reply_to in self.question_messages:
            runner.answer(self.db, self.question_messages[reply_to], text)
            if self.wake:
                self.wake()
            await self.api("sendMessage", chat_id=chat_id, text="Answer sent.")
            return
        agent_name, title = None, text
        if ":" in text:
            head, rest = text.split(":", 1)
            if rest.strip():
                candidate = w.get_agent(self.db, head.strip())
                if candidate:
                    agent_name, title = candidate["name"], rest.strip()
        if agent_name is None:
            agents = w.list_agents(self.db)
            if not agents:
                await self.api(
                    "sendMessage",
                    chat_id=chat_id,
                    text="There are no agents yet. Add one in the web app first.",
                )
                return
            agent_name = agents[0]["name"]
        w.add_task(self.db, title, to=agent_name, created_by="telegram")
        if self.wake:
            self.wake()
        await self.api("sendMessage", chat_id=chat_id, text=f"Task added for {agent_name}.")

    async def notify(self) -> None:
        events = w.events_since(self.db, self.last_seq, 200)
        for event in events:
            self.last_seq = max(self.last_seq, event["seq"])
        for card in w.list_waiting(self.db):
            if card["id"] in self.notified_approvals:
                continue
            self.notified_approvals.add(card["id"])
            if card["kind"] == "question":
                result = await self.api(
                    "sendMessage",
                    chat_id=self.owner,
                    text=f"{card['agent']} asks: {card['preview']}\n\n"
                    "Reply to this message with your answer.",
                )
                self.question_messages[result["message_id"]] = card["id"]
            else:
                keyboard = [
                    [
                        {"text": "Approve", "callback_data": self.callback("a", card["id"])},
                        {"text": "Deny", "callback_data": self.callback("d", card["id"])},
                        {
                            "text": "Always allow",
                            "callback_data": self.callback("w", card["id"]),
                        },
                    ]
                ]
                await self.api(
                    "sendMessage",
                    chat_id=self.owner,
                    text=f"{card['agent']} wants to run {card['tool']}: {card['preview']}",
                    reply_markup={"inline_keyboard": keyboard},
                )
        for run in w.list_runs(self.db, status="failed", limit=10):
            if run["id"] not in self.seen_runs:
                self.seen_runs.add(run["id"])
                await self.api(
                    "sendMessage",
                    chat_id=self.owner,
                    text=f"Run #{run['id']} for {run['agent']} failed: "
                    f"{run['error'] or 'unknown'}.",
                )
        for task in w.list_tasks(self.db, status="todo", limit=50):
            if task["agent"] is None and task["id"] not in self.seen_tasks:
                self.seen_tasks.add(task["id"])
                await self.api(
                    "sendMessage",
                    chat_id=self.owner,
                    text=f"A task was handed to you: {task['title']}",
                )
