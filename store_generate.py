#!/usr/bin/env python3
"""
Generate *more store-like* user journeys (e-commerce clickstream) as flattened transitions.

Design goals vs the simple generator:
- Use meaningful event states (HOME/PLP/PDP/CART/CHECKOUT/etc.) instead of STATE_1..N
- Add user heterogeneity via simple user segments (new/returning/high_intent/window_shopper)
- Use constrained transitions (a lightweight state machine) so paths look like real funnels
- Use heavy-tailed, state-dependent dwell times (clamped to a user-provided range)

Output format (flattened array; one object per transition):
  {
    "user_id": 12,
    "session_id": 8123,
    "segment": "returning",
    "source": "PDP",
    "dist": "ADD_TO_CART",
    "source_timestamp": 0,
    "dist_timestamp": 18
  }

Notes:
- Journeys always start at START and end with NULL (exit) or CONVERT (purchase).
- We interpret "journey length range" as number of INTERMEDIATE states between START and terminal,
  to stay compatible with generate.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TOTAL_JOURNEYS_DEFAULT = 100_000


@dataclass(frozen=True)
class Params:
    total_journeys: int
    journeys_per_user_range: Tuple[int, int]
    journey_length_range: Tuple[int, int]  # intermediate states between START and terminal
    session_length_seconds_range: Tuple[int, int]  # clamp for per-hop dwell seconds
    conversion_rate: float
    output: Path
    seed: Optional[int]


SEGMENTS = ["new", "returning", "high_intent", "window_shopper"]
SEGMENT_PRIORS = {
    "new": 0.45,
    "returning": 0.35,
    "high_intent": 0.10,
    "window_shopper": 0.10,
}


# Non-terminal store events (i.e., candidates for intermediate states)
STORE_EVENTS: List[str] = [
    "HOME",
    "SEARCH",
    "SEARCH_RESULTS",
    "PLP",
    "FILTER",
    "SORT",
    "PDP",
    "VIEW_REVIEWS",
    "ADD_TO_CART",
    "VIEW_CART",
    "APPLY_COUPON",
    "BEGIN_CHECKOUT",
    "SHIPPING_INFO",
    "PAYMENT",
    "PAYMENT_ERROR",
    "ACCOUNT_LOGIN",
]


# State-dependent dwell time parameters (lognormal in seconds).
# Typical e-comm: PDP longer than PLP, checkout steps moderate, etc.
# delta_seconds = clamp(lognormal(mu, sigma), min,max)
DWELL_LOGNORMAL: Dict[str, Tuple[float, float]] = {
    "START": (0.0, 0.20),
    "HOME": (1.8, 0.55),
    "SEARCH": (1.6, 0.50),
    "SEARCH_RESULTS": (2.0, 0.55),
    "PLP": (1.9, 0.55),
    "FILTER": (1.4, 0.45),
    "SORT": (1.2, 0.35),
    "PDP": (2.4, 0.65),
    "VIEW_REVIEWS": (2.0, 0.60),
    "ADD_TO_CART": (1.0, 0.30),
    "VIEW_CART": (1.7, 0.55),
    "APPLY_COUPON": (1.4, 0.55),
    "BEGIN_CHECKOUT": (1.6, 0.55),
    "SHIPPING_INFO": (1.8, 0.60),
    "PAYMENT": (1.7, 0.65),
    "PAYMENT_ERROR": (1.2, 0.55),
    "ACCOUNT_LOGIN": (1.6, 0.60),
    "NULL": (0.0, 0.20),
    "CONVERT": (0.0, 0.20),
}


def _parse_int_range(text: str, *, name: str, minimum: int = 0) -> Tuple[int, int]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"{name} must be in the form 'min,max' (got: {text!r})")
    try:
        a = int(parts[0])
        b = int(parts[1])
    except ValueError as e:
        raise ValueError(f"{name} values must be integers (got: {text!r})") from e
    if a < minimum or b < minimum:
        raise ValueError(f"{name} must be >= {minimum} (got: {a},{b})")
    return (a, b) if a <= b else (b, a)


def _parse_conversion_rate(text: str) -> float:
    s = text.strip()
    is_percent = s.endswith("%")
    if is_percent:
        s = s[:-1].strip()
    try:
        v = float(s)
    except ValueError as e:
        raise ValueError(f"Conversion rate must be a number like 0.12 or 12% (got: {text!r})") from e
    if is_percent or v > 1.0:
        v = v / 100.0
    if not (0.0 <= v <= 1.0):
        raise ValueError(f"Conversion rate must be between 0 and 1 (got: {v})")
    return v


def _prompt(text: str, *, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{text}{suffix}: ").strip()
        if raw:
            return raw
        if default is not None:
            return default


def _weighted_choice(rng: random.Random, items: List[Tuple[str, float]]) -> str:
    total = sum(w for _, w in items)
    if total <= 0:
        # Fallback: choose uniformly if weights are degenerate
        return rng.choice([k for k, _ in items])
    x = rng.random() * total
    acc = 0.0
    for k, w in items:
        acc += w
        if x <= acc:
            return k
    return items[-1][0]


def _pick_segment(rng: random.Random) -> str:
    items = [(k, SEGMENT_PRIORS.get(k, 0.0)) for k in SEGMENTS]
    return _weighted_choice(rng, items)


def _segment_bias(segment: str, event: str) -> float:
    """
    Multiplicative bias applied when picking next events (simple + explainable).
    """
    if segment == "high_intent":
        if event in {"ADD_TO_CART", "VIEW_CART", "BEGIN_CHECKOUT", "SHIPPING_INFO", "PAYMENT"}:
            return 1.8
        if event in {"FILTER", "SORT", "VIEW_REVIEWS"}:
            return 0.8
    if segment == "window_shopper":
        if event in {"FILTER", "SORT", "VIEW_REVIEWS", "PLP", "PDP"}:
            return 1.4
        if event in {"ADD_TO_CART", "BEGIN_CHECKOUT", "PAYMENT"}:
            return 0.6
    if segment == "new":
        if event in {"ACCOUNT_LOGIN"}:
            return 1.3
    if segment == "returning":
        if event in {"PDP", "ADD_TO_CART", "VIEW_CART"}:
            return 1.2
    return 1.0


def _store_next_candidates(current: str) -> List[Tuple[str, float]]:
    """
    Base constrained transitions (no terminals here; we append terminal at end).
    We allow some loops to mimic browsing.
    """
    if current == "START":
        return [("HOME", 0.45), ("PLP", 0.20), ("SEARCH_RESULTS", 0.20), ("PDP", 0.15)]
    if current == "HOME":
        return [("PLP", 0.35), ("SEARCH", 0.15), ("SEARCH_RESULTS", 0.20), ("PDP", 0.20), ("ACCOUNT_LOGIN", 0.10)]
    if current == "SEARCH":
        return [("SEARCH_RESULTS", 0.85), ("HOME", 0.15)]
    if current == "SEARCH_RESULTS":
        return [("FILTER", 0.20), ("SORT", 0.15), ("PDP", 0.40), ("PLP", 0.20), ("SEARCH", 0.05)]
    if current == "PLP":
        return [("FILTER", 0.20), ("SORT", 0.15), ("PDP", 0.45), ("SEARCH", 0.10), ("HOME", 0.10)]
    if current == "FILTER":
        return [("PLP", 0.50), ("SEARCH_RESULTS", 0.30), ("PDP", 0.15), ("FILTER", 0.05)]
    if current == "SORT":
        return [("PLP", 0.55), ("SEARCH_RESULTS", 0.25), ("PDP", 0.15), ("SORT", 0.05)]
    if current == "PDP":
        return [("VIEW_REVIEWS", 0.20), ("ADD_TO_CART", 0.20), ("PLP", 0.30), ("SEARCH_RESULTS", 0.20), ("PDP", 0.10)]
    if current == "VIEW_REVIEWS":
        return [("PDP", 0.60), ("PLP", 0.25), ("ADD_TO_CART", 0.15)]
    if current == "ADD_TO_CART":
        return [("VIEW_CART", 0.60), ("PDP", 0.25), ("PLP", 0.15)]
    if current == "VIEW_CART":
        return [("APPLY_COUPON", 0.15), ("BEGIN_CHECKOUT", 0.50), ("PDP", 0.20), ("PLP", 0.15)]
    if current == "APPLY_COUPON":
        return [("VIEW_CART", 0.50), ("BEGIN_CHECKOUT", 0.35), ("APPLY_COUPON", 0.15)]
    if current == "BEGIN_CHECKOUT":
        return [("ACCOUNT_LOGIN", 0.20), ("SHIPPING_INFO", 0.70), ("VIEW_CART", 0.10)]
    if current == "ACCOUNT_LOGIN":
        return [("BEGIN_CHECKOUT", 0.35), ("SHIPPING_INFO", 0.35), ("HOME", 0.10), ("PDP", 0.20)]
    if current == "SHIPPING_INFO":
        return [("PAYMENT", 0.80), ("VIEW_CART", 0.10), ("SHIPPING_INFO", 0.10)]
    if current == "PAYMENT":
        return [("PAYMENT_ERROR", 0.15), ("PAYMENT", 0.10), ("BEGIN_CHECKOUT", 0.10), ("SHIPPING_INFO", 0.10), ("VIEW_CART", 0.55)]
    if current == "PAYMENT_ERROR":
        return [("PAYMENT", 0.65), ("VIEW_CART", 0.20), ("PAYMENT_ERROR", 0.15)]
    # fallback for unknown
    return [("HOME", 1.0)]


def _sample_dwell_seconds(
    *,
    rng: random.Random,
    source_state: str,
    clamp_range: Tuple[int, int],
) -> int:
    lo, hi = clamp_range
    mu, sigma = DWELL_LOGNORMAL.get(source_state, (1.7, 0.6))
    # Python's random.lognormvariate takes mu/sigma in log space.
    x = rng.lognormvariate(mu, sigma)
    # Avoid huge values; clamp as requested
    if x < lo:
        return lo
    if x > hi:
        return hi
    return int(round(x))


def generate_store_journey(
    *,
    rng: random.Random,
    segment: str,
    intermediate_len_range: Tuple[int, int],
    conversion_rate: float,
) -> List[str]:
    """
    Generate a store-like sequence of states:
      START, <intermediates...>, (CONVERT|NULL)
    """
    k = rng.randint(intermediate_len_range[0], intermediate_len_range[1])
    states: List[str] = ["START"]
    current = "START"

    for _ in range(k):
        base = _store_next_candidates(current)
        biased: List[Tuple[str, float]] = []
        for nxt, w in base:
            w2 = w * _segment_bias(segment, nxt)
            biased.append((nxt, w2))
        nxt = _weighted_choice(rng, biased)
        states.append(nxt)
        current = nxt

    terminal = "CONVERT" if rng.random() < conversion_rate else "NULL"
    states.append(terminal)
    return states


def _collect_params(args: argparse.Namespace) -> Params:
    total_journeys = args.total_journeys
    output = Path(args.output).expanduser().resolve()
    seed = args.seed

    jpur_s = args.journeys_per_user_range or _prompt("Journeys per user range (min,max)", default="1,5")
    journeys_per_user_range = _parse_int_range(jpur_s, name="Journeys per user range", minimum=1)

    jl_s = args.journey_length_range or _prompt(
        "Journey length range of INTERMEDIATE events between START and terminal (min,max)",
        default="3,15",
    )
    journey_length_range = _parse_int_range(jl_s, name="Journey length range", minimum=0)

    sl_s = args.session_length_seconds_range or _prompt("Session length seconds range per hop/state (min,max)", default="2,180")
    session_length_seconds_range = _parse_int_range(sl_s, name="Session length seconds range", minimum=0)

    cr_s = args.conversion_rate or _prompt("Conversion rate (0..1, or percent like 12%)", default="3%")
    conversion_rate = _parse_conversion_rate(cr_s)

    return Params(
        total_journeys=total_journeys,
        journeys_per_user_range=journeys_per_user_range,
        journey_length_range=journey_length_range,
        session_length_seconds_range=session_length_seconds_range,
        conversion_rate=conversion_rate,
        output=output,
        seed=seed,
    )


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="Generate store-like journeys as flattened transitions JSON.")
    p.add_argument("--total-journeys", type=int, default=TOTAL_JOURNEYS_DEFAULT)
    p.add_argument("--journeys-per-user-range", type=str, default=None)
    p.add_argument("--journey-length-range", type=str, default=None)
    p.add_argument("--session-length-seconds-range", type=str, default=None)
    p.add_argument("--conversion-rate", type=str, default=None)
    p.add_argument("--output", type=str, default="transitions_store.json")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)

    try:
        params = _collect_params(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    if params.total_journeys < 1:
        print("Error: total journeys must be >= 1", file=sys.stderr)
        return 2

    rng = random.Random(params.seed)
    params.output.parent.mkdir(parents=True, exist_ok=True)

    journeys_written = 0
    transitions_written = 0
    user_id = 1
    session_id = 1

    with params.output.open("w", encoding="utf-8") as f:
        f.write("[\n")
        first_item = True

        while journeys_written < params.total_journeys:
            segment = _pick_segment(rng)
            per_user = rng.randint(params.journeys_per_user_range[0], params.journeys_per_user_range[1])

            for _ in range(per_user):
                if journeys_written >= params.total_journeys:
                    break

                journey_states = generate_store_journey(
                    rng=rng,
                    segment=segment,
                    intermediate_len_range=params.journey_length_range,
                    conversion_rate=params.conversion_rate,
                )

                timestamps: List[int] = [0]
                for s in journey_states[:-1]:
                    delta = _sample_dwell_seconds(
                        rng=rng,
                        source_state=s,
                        clamp_range=params.session_length_seconds_range,
                    )
                    timestamps.append(timestamps[-1] + delta)

                for i, (src, dst) in enumerate(zip(journey_states, journey_states[1:])):
                    obj = {
                        "user_id": user_id,
                        "session_id": session_id,
                        "segment": segment,
                        "source": src,
                        "dist": dst,
                        "source_timestamp": timestamps[i],
                        "dist_timestamp": timestamps[i + 1],
                    }
                    if first_item:
                        first_item = False
                    else:
                        f.write(",\n")
                    f.write(json.dumps(obj, ensure_ascii=False))
                    transitions_written += 1

                journeys_written += 1
                session_id += 1

                if journeys_written in (1, 10, 100, 1000) or journeys_written % 10_000 == 0:
                    print(
                        f"Wrote {journeys_written:,}/{params.total_journeys:,} journeys "
                        f"({transitions_written:,} transitions)...",
                        file=sys.stderr,
                    )

            user_id += 1

        f.write("\n]\n")

    print(
        f"Done. Wrote {journeys_written:,} journeys and {transitions_written:,} transitions to {params.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


