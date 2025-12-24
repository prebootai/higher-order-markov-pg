#!/usr/bin/env python3
"""
Build an N-order Markov model (e.g., 6th-order) from a transitions JSON file.

Input format: flattened JSON array of per-transition objects, one per line (as produced by
`generate.py` / `store_generate.py`), e.g.:
[
  {"user_id": 1, "session_id": 1, "source": "START", "dist": "HOME", "source_timestamp": 0, "dist_timestamp": 5},
  ...
]

Journey reconstruction:
- If `session_id` exists, we group transitions by (user_id, session_id).
- Otherwise we detect journey boundaries via `source == "START"` (and/or timestamp reset),
  and also realign if the chain breaks (when next transition's source doesn't match our last state).

Output:
- A compact JSON file containing counts for orders 0..N to enable backoff prediction:
  {
    "order": 6,
    "delimiter": "|",
    "root_key": "__ROOT__",
    "states": [...],
    "meta": {"total_sequences": ..., "total_transitions": ...},
    "counts": {
      "0": {"__ROOT__": {"HOME": 123, ...}},
      "1": {"START": {"HOME": 100, ...}, ...},
      "2": {"START|HOME": {"PLP": 50, ...}, ...},
      ...
      "6": {...}
    }
  }
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, Iterator, List, Optional, Tuple


ROOT_KEY = "__ROOT__"
DELIM = "|"
TERMINALS = {"NULL", "CONVERT"}


def iter_transition_lines(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s == "[" or s == "]":
                continue
            if s.endswith(","):
                s = s[:-1].rstrip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON object line: {s[:200]!r}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object per line, got: {type(obj).__name__}")
            yield obj


def _context_key(states: List[str], start: int, end_inclusive: int) -> str:
    # states[start:end_inclusive+1]
    return DELIM.join(states[start : end_inclusive + 1])


def _update_counts_for_sequence(
    seq: List[str],
    order: int,
    counts_by_order: List[DefaultDict[str, Counter[str]]],
) -> int:
    """
    Updates counts for orders 0..order inclusive.
    Returns number of transitions processed.
    """
    if len(seq) < 2:
        return 0

    transitions = 0
    for i in range(len(seq) - 1):
        nxt = seq[i + 1]
        # order 0 (unconditional)
        counts_by_order[0][ROOT_KEY][nxt] += 1

        max_o = min(order, i + 1)  # history available ending at seq[i]
        for o in range(1, max_o + 1):
            key = _context_key(seq, i - o + 1, i)
            counts_by_order[o][key][nxt] += 1

        transitions += 1

    return transitions


def _flush_sequence(
    seq: List[str],
    *,
    order: int,
    counts_by_order: List[DefaultDict[str, Counter[str]]],
    states_set: set[str],
) -> int:
    for s in seq:
        states_set.add(s)
    return _update_counts_for_sequence(seq, order, counts_by_order)


def build_k_order_model(
    transitions: Iterable[dict],
    *,
    order: int,
) -> Tuple[Dict[str, Dict[str, Dict[str, int]]], Dict[str, int], List[str]]:
    """
    Returns (counts_json, meta, states_sorted).
    counts_json is a JSON-serializable dict: order -> context_key -> next -> count
    """
    if order < 0:
        raise ValueError("order must be >= 0")

    counts_by_order: List[DefaultDict[str, Counter[str]]] = [
        defaultdict(Counter) for _ in range(order + 1)
    ]
    states_set: set[str] = set()

    total_sequences = 0
    total_transitions = 0

    current_seq: List[str] = []
    current_key: Optional[Tuple[Optional[int], Optional[int]]] = None  # (user_id, session_id)
    seen_session_id = False

    def start_new_sequence(start_state: str) -> None:
        nonlocal current_seq
        current_seq = [start_state]

    def maybe_flush() -> None:
        nonlocal total_sequences, total_transitions, current_seq
        if len(current_seq) >= 2:
            total_transitions += _flush_sequence(
                current_seq,
                order=order,
                counts_by_order=counts_by_order,
                states_set=states_set,
            )
            total_sequences += 1
        current_seq = []

    for t in transitions:
        src = t.get("source")
        dst = t.get("dist", t.get("dest"))
        if not isinstance(src, str) or not isinstance(dst, str):
            continue

        uid = t.get("user_id")
        sid = t.get("session_id")
        key: Optional[Tuple[Optional[int], Optional[int]]] = None
        if sid is not None:
            seen_session_id = True
            key = (uid if isinstance(uid, int) else None, sid if isinstance(sid, int) else None)

        # If session_id is present, treat key changes as hard boundaries
        if seen_session_id:
            if current_key is None:
                current_key = key
            elif key != current_key:
                maybe_flush()
                current_key = key

        # If we don't have an active sequence, start it.
        if not current_seq:
            start_new_sequence(src)

        # If chain breaks (our last state doesn't match this transition's source), realign.
        if current_seq and current_seq[-1] != src:
            # Try to salvage: if this looks like a new journey (START), flush and restart.
            if src == "START":
                maybe_flush()
                start_new_sequence("START")
            else:
                # Otherwise, treat as boundary and restart from src.
                maybe_flush()
                start_new_sequence(src)

        # Append dst
        current_seq.append(dst)

        if dst in TERMINALS:
            maybe_flush()
            current_key = None if not seen_session_id else current_key

    # flush tail
    maybe_flush()

    # Convert to JSON-serializable
    counts_json: Dict[str, Dict[str, Dict[str, int]]] = {}
    for o in range(order + 1):
        # stable sort contexts
        ctx_sorted: Dict[str, Dict[str, int]] = {}
        for ctx in sorted(counts_by_order[o].keys()):
            nxt_ctr = counts_by_order[o][ctx]
            ctx_sorted[ctx] = {k: int(nxt_ctr[k]) for k in sorted(nxt_ctr.keys())}
        counts_json[str(o)] = ctx_sorted

    meta = {"total_sequences": total_sequences, "total_transitions": total_transitions}
    states_sorted = sorted(states_set)
    return counts_json, meta, states_sorted


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="Build an N-order Markov model (with backoff counts) from transitions JSON.")
    p.add_argument("--input", required=True, help="Transitions JSON file (flattened array, 1 object per line).")
    p.add_argument("--order", type=int, default=6, help="Markov order (e.g. 6).")
    p.add_argument("--output", default="markov_chain.json", help="Output model JSON.")
    args = p.parse_args(argv)

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()

    if not in_path.exists():
        print(f"Error: input not found: {in_path}", file=sys.stderr)
        return 2

    try:
        counts, meta, states = build_k_order_model(iter_transition_lines(in_path), order=args.order)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    payload = {
        "order": args.order,
        "delimiter": DELIM,
        "root_key": ROOT_KEY,
        "states": states,
        "meta": meta,
        "counts": counts,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"Done. Built order-{args.order} model from {meta['total_sequences']:,} sequences "
        f"and {meta['total_transitions']:,} transitions -> {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))



