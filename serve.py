"""Entrypoint for serving the PhysAE model via FastAPI."""
from __future__ import annotations

import argparse

import uvicorn

from physae.config import load_config
from physae.inference.service import InferenceService, create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve PhysAE FastAPI app")
    parser.add_argument("--config", default="configs/deploy.yaml")
    parser.add_argument("--checkpoint", default="outputs/latest.ckpt")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    service = InferenceService.from_checkpoint(args.checkpoint, config)
    app = create_app(service)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
