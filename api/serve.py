#!/usr/bin/env python3

import argparse
import os
import pathlib
import subprocess


def _argparse() -> dict:
    parser = argparse.ArgumentParser(description="Start lettucedetect Web API.")
    parser.add_argument(
        "--model",
        help='Path or huggingface URL to the model. The default value is "KRLabsOrg/lettucedect-base-modernbert-en-v1".',
        default="KRLabsOrg/lettucedect-base-modernbert-en-v1",
    )
    parser.add_argument(
        "--method",
        help='Hallucination detection method. The default value is "transformer".',
        choices=["transformer"],
        default="transformer",
    )
    return parser.parse_args()


def _run_fastapi(args: dict) -> None:
    api_folder = pathlib.Path(__file__).parent.resolve()
    env = os.environ.copy()
    env["LETTUCEDETECT_MODEL"] = args.model
    env["LETTUCEDETECT_METHOD"] = args.method
    try:
        # Ignore S603: Validate input to run method. False positive because
        # api_folder is not input by the user.
        # Ignore S607: Use relative path for fastapi. Unavoidable here because
        # fastapi executable location is not known.
        subprocess.run(["fastapi", "dev", api_folder / "main.py"], env=env)  # noqa: S603, S607
    except KeyboardInterrupt:
        pass


def main() -> None:
    """Entry point for script."""
    args = _argparse()
    _run_fastapi(args)


if __name__ == "__main__":
    main()
