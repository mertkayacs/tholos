import argparse
import importlib
import json
import os
import secrets
from pathlib import Path

from tholos import workspace as w
from tholos.db import connect, init, tx


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tholos")
    parser.add_argument("--host", default=os.environ.get("THOLOS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=7070)
    parser.add_argument("--home")
    commands = parser.add_subparsers(dest="command")
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default=argparse.SUPPRESS)
    serve.add_argument("--port", type=int, default=argparse.SUPPRESS)
    serve.add_argument("--home", default=argparse.SUPPRESS)
    bench = commands.add_parser("bench")
    bench.add_argument("--base-url", required=True)
    bench.add_argument("--model", required=True)
    bench.add_argument("--api-key-env")
    bench.add_argument("--json-mode", choices=["schema", "object", "none"], default="schema")
    bench.add_argument("--scenarios")
    bench.add_argument("--only")
    bench.add_argument("--limit", type=int)
    bench.add_argument("--out")
    export = commands.add_parser("export")
    export.add_argument("--run", type=int, required=True)
    export.add_argument("--home", default=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.home:
        os.environ["THOLOS_HOME"] = str(Path(args.home).expanduser())
    if args.command == "bench":
        from tholos.bench.runner import bench

        api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
        if args.api_key_env and not api_key:
            parser.error("The API key environment variable is empty or unset")
        try:
            bench(
                args.base_url,
                args.model,
                api_key,
                args.json_mode,
                args.scenarios,
                args.only,
                args.limit,
                args.out,
            )
        except ValueError as exc:
            parser.error(str(exc))
        return
    if args.command == "export":
        db = connect()
        try:
            init(db)
            run = w.get_run(db, args.run)
            if run is None:
                parser.error("Run was not found")
            print(json.dumps(run, ensure_ascii=True, indent=2))
        finally:
            db.close()
        return
    os.environ["THOLOS_HOST"] = args.host
    web = importlib.import_module("tholos.web")
    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(args.host, args.host)
    host = f"[{host}]" if ":" in host else host
    lines = [f"Tholos: http://{host}:{args.port}"]
    db = connect()
    try:
        init(db)
        if not web.is_loopback(args.host):
            with tx(db):
                token = w.get_setting(db, "access_token")
                if not token:
                    token = secrets.token_urlsafe(24)
                    w.set_setting(db, "access_token", token)
            lines.append(f"Access token: {token}")
        if not w.list_models(db):
            lines.append(
                "No model yet. Run: ollama pull hf.co/mertkayacs/Tholos-2B-GGUF:Q4_K_M, "
                "then open Settings > Detect."
            )
    finally:
        db.close()
    print("\n".join(lines), flush=True)
    import uvicorn

    uvicorn.run(web.app, host=args.host, port=args.port)
