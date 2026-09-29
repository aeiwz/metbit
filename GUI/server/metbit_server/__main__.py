"""Entry point: METBIT_TOKEN=<token> python -m metbit_server --port <port>"""
import argparse
import os
import sys

import uvicorn

from .app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="metbit_server")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    token = os.environ.get("METBIT_TOKEN")
    if not token:
        sys.exit("METBIT_TOKEN is not set")
    uvicorn.run(create_app(token), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
