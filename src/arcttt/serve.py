"""Minimal adaptation endpoint: POST a task, get ranked predictions.

The product seed as a service (stdlib only — no framework dependency):

    python -m arcttt.serve --model <hf-dir> --port 8332 [--raw-format --dfs]

    POST /adapt   body: ARC task JSON ({"train": [...], "test": [...]})
                  -> {"task_id", "seconds", "predictions": [...]}
    GET  /health  -> {"status": "ok", "model": "<dir>"}

One request at a time by design: per-task adaptation owns the GPU; a
queueing layer belongs in front of this, not inside it.
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from arcttt.augment import DIHEDRAL_SWEEP, expanded_sweep
from arcttt.model import CausalLMPredictor, TTTConfig
from arcttt.solve import SolveConfig, solve_task
from arcttt.tasks import Pair, Task, TaskFormatError, grid_to_lists, to_grid


def task_from_payload(payload: Any, task_id: str = "request") -> Task:
    if not isinstance(payload, dict) or "train" not in payload or "test" not in payload:
        raise TaskFormatError("body must be a task object with train and test")

    def pairs(items: Any, split: str) -> tuple[Pair, ...]:
        if not isinstance(items, list) or not items:
            raise TaskFormatError(f"{split} must be a non-empty list")
        built = []
        for item in items:
            if not isinstance(item, dict) or "input" not in item:
                raise TaskFormatError(f"{split} pairs need an input grid")
            output = item.get("output")
            built.append(
                Pair(
                    input=to_grid(item["input"]),
                    output=to_grid(output) if output is not None else None,
                )
            )
        return tuple(built)

    task = Task(task_id=task_id, train=pairs(payload["train"], "train"),
                test=pairs(payload["test"], "test"))
    task.validate()
    return task


class AdaptService:
    """Owns the loaded model and runs one adaptation per request."""

    def __init__(self, model: Any, tokenizer: Any, config: TTTConfig,
                 solve: SolveConfig, device: Any, model_name: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.solve = solve
        self.device = device
        self.model_name = model_name
        self.requests_served = 0

    def adapt(self, payload: Any) -> dict[str, Any]:
        task = task_from_payload(payload, task_id=f"request-{self.requests_served}")
        started = time.monotonic()
        predictor = CausalLMPredictor(self.model, self.tokenizer, self.config, self.device)
        ranked = solve_task(task, predictor, self.solve)
        self.requests_served += 1
        predictions: list[dict[str, Any]] = []
        for attempts in ranked:
            predictions.append(
                {"attempts": [grid_to_lists(grid) for grid in attempts[:2]]}
            )
        return {
            "task_id": task.task_id,
            "seconds": round(time.monotonic() - started, 1),
            "predictions": predictions,
        }


def make_handler(service: AdaptService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            if self.path == "/health":
                self._send(200, {"status": "ok", "model": service.model_name,
                                 "requests_served": service.requests_served})
            else:
                self._send(404, {"error": "unknown path"})

        def do_POST(self) -> None:  # noqa: N802 (http.server API)
            if self.path != "/adapt":
                self._send(404, {"error": "unknown path"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                self._send(200, service.adapt(payload))
            except TaskFormatError as error:
                self._send(400, {"error": str(error)})
            except Exception as error:  # noqa: BLE001 — surface, don't die
                self._send(500, {"error": type(error).__name__})

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[serve] {fmt % args}", flush=True)

    return Handler


def build_service(args: argparse.Namespace) -> AdaptService:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)  # type: ignore[no-untyped-call]
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(
        device  # type: ignore[arg-type]
    )
    config = TTTConfig(
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        epochs=args.epochs,
        raw_qwen_format=args.raw_format,
        gradient_checkpointing=device.type == "cuda",
        use_dfs=args.dfs,
        dfs_probability_cutoff=args.cutoff,
        shuffle_examples=True,
    )
    solve = SolveConfig(
        augmentations=DIHEDRAL_SWEEP,
        samples_per_augmentation=1,
        rescore_augmentations=DIHEDRAL_SWEEP,
        ttt_augmentations=(
            expanded_sweep(seed=0, palettes_per_element=args.palettes)
            if args.palettes
            else None
        ),
    )
    return AdaptService(model, tokenizer, config, solve, device, args.model)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8332)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--palettes", type=int, default=0)
    parser.add_argument("--raw-format", action="store_true")
    parser.add_argument("--dfs", action="store_true")
    parser.add_argument("--cutoff", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    service = build_service(args)
    server = HTTPServer(("127.0.0.1", args.port), make_handler(service))
    print(f"[serve] listening on 127.0.0.1:{args.port} | model {args.model}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
