from __future__ import annotations

import sys

from miniclaw.cli import main as run_cli


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "web":
        from miniclaw.web_server import run as run_web

        run_web()
        return

    run_cli()


if __name__ == "__main__":
    main()
