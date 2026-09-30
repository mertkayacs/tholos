import argparse
import importlib
import json
import os
from pathlib import Path

from tholos import workspace as w
from tholos.db import connect, init


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tholos")
    parser.add_argument("--host", default="127.0.0.1")
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
    try:
        app = importlib.import_module("tholos.web").app
    except ModuleNotFoundError as exc:
        if exc.name != "tholos.web":
            raise
        print("The web interface is not installed yet. The core CLI supports bench and export.")
        return
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
