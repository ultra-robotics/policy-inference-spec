from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from policy_inference_spec.client import (
    DEFAULT_PREDICT_URL,
    RemotePolicyClient,
    _random_warmup_wire_frame,
)
from policy_inference_spec.hardware_model import HardwareModel


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Warmup + predict smoke test against a policy inference WebSocket server.")
    p.add_argument(
        "--url",
        default=os.environ.get("POLICY_SERVER_URL") or DEFAULT_PREDICT_URL,
        help=f"WebSocket URL (default: $POLICY_SERVER_URL or {DEFAULT_PREDICT_URL!r})",
    )
    p.add_argument("--hardware-model", choices=("gen1", "gen2"), default="gen2")
    p.add_argument(
        "--predicts",
        type=int,
        default=2,
        metavar="N",
        help="Number of predict requests after warmup (default: 2)",
    )
    p.add_argument("--api-key", default=os.environ.get("POLICY_AUTH_TOKEN"), help="Optional x-api-key header")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


async def _run() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    if args.predicts < 1:
        print("--predicts must be >= 1", file=sys.stderr)
        return 2

    hm = HardwareModel(args.hardware_model)
    headers = {"x-api-key": args.api_key} if args.api_key else None
    client = RemotePolicyClient(args.url, policy_auth_headers=headers)

    print(f"URL {args.url!r} hardware_model={hm.value}", flush=True)
    if not client.warmup(hardware_model=hm):
        print("Warmup failed.", file=sys.stderr)
        return 1

    async with client:
        for i in range(args.predicts):
            frame = _random_warmup_wire_frame(hm)
            pred = await client.predict(frame)
            print(
                f"predict {i + 1}/{args.predicts}: "
                f"total_latency_ms={pred.total_latency_ms:.1f} "
                f"actions_shape={tuple(pred.actions_d.shape)} "
                f"policy_id={pred.policy_id!r}",
            )
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
