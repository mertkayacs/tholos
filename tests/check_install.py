import argparse
import os
import socket
import subprocess
import time
import venv
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import URLError
from urllib.request import urlopen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    wheel = parser.parse_args().wheel.resolve(strict=True)
    with TemporaryDirectory(prefix="tholos-install-") as directory:
        root = Path(directory)
        environment = root / "venv"
        venv.EnvBuilder().create(environment)
        python = str(environment / "bin" / "python")
        subprocess.run(
            ["uv", "pip", "install", "--python", python, str(wheel)],
            cwd=root,
            check=True,
            timeout=120,
        )
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env.update(THOLOS_HOME=str(root / "data"), THOLOS_HOST="0.0.0.0", PYTHONUNBUFFERED="1")
        subprocess.run(
            [
                python,
                "-c",
                "import sys, tholos; from pathlib import Path; "
                "assert Path(tholos.__file__).is_relative_to(Path(sys.prefix))",
            ],
            cwd=root,
            env=env,
            check=True,
            timeout=10,
        )
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        process = subprocess.Popen(
            [str(environment / "bin" / "tholos"), "--port", str(port)],
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 15
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Installed app exited during startup")
                try:
                    with urlopen(url + "/healthz", timeout=1) as response:
                        assert response.status == 200 and response.read() == b"ok"
                    break
                except URLError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Installed app did not become healthy") from None
                    time.sleep(0.05)
            with urlopen(url + "/", timeout=3) as response:
                assert response.geturl() == url + "/login"
                assert b"Access token" in response.read()
            with urlopen(url + "/static/app.css", timeout=3) as response:
                assert response.status == 200 and response.read()
        finally:
            process.terminate()
            try:
                output, _ = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=5)
                raise
        assert f"Tholos: {url}" in output and "Access token:" in output
        assert "No model yet. Run: ollama pull" in output
    print(
        "Clean wheel install passed: imports, CLI, health, auth, templates, and static files."
    )


if __name__ == "__main__":
    main()
