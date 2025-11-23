#!/usr/bin/env python3
"""Interactive and CLI wrapper to generate jokes with configurable parameters."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict

from markov_jokes import (
    build_chain,
    build_sequences,
    collect_ngrams,
    generate_joke,
    load_jokes,
    uniqueness_score,
)


COMMON_DEFAULTS: Dict[str, Any] = {
    "dataset": Path("full_jokes.txt"),
    "encoding": "utf-8",
    "fallback_encoding": "cp1251",
    "errors": "ignore",
    "order": 3,
    "min_tokens": 5,
    "split_sentences": False,
    "count": 5,
    "max_length": 60,
    "sentences_per_joke": 2,
    "carry_context": True,
    "seed": None,
    "lowercase": False,
    "preserve_linebreaks": False,
}


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, default=COMMON_DEFAULTS["dataset"], help="Path to the source jokes file.")
    parser.add_argument("--encoding", default=COMMON_DEFAULTS["encoding"], help="Primary encoding used to read the dataset.")
    parser.add_argument("--fallback-encoding", dest="fallback_encoding", default=COMMON_DEFAULTS["fallback_encoding"], help="Fallback encoding.")
    parser.add_argument("--errors", default=COMMON_DEFAULTS["errors"], help="Encoding error handling strategy.")
    parser.add_argument("--order", type=int, default=COMMON_DEFAULTS["order"], help="Order of the Markov chain (n-gram size).")
    parser.add_argument("--min-tokens", type=int, default=COMMON_DEFAULTS["min_tokens"], help="Skip sequences with fewer tokens.")
    parser.add_argument("--split-sentences", action="store_true", default=COMMON_DEFAULTS["split_sentences"], help="Train on individual sentences.")
    parser.add_argument("--count", type=int, default=COMMON_DEFAULTS["count"], help="Number of jokes to generate.")
    parser.add_argument("--max-length", type=int, default=COMMON_DEFAULTS["max_length"], help="Maximum tokens per generated joke.")
    parser.add_argument("--sentences-per-joke", type=int, default=COMMON_DEFAULTS["sentences_per_joke"], help="Target number of sentences per generated joke.")
    parser.add_argument(
        "--no-context-carry",
        dest="carry_context",
        action="store_false",
        default=COMMON_DEFAULTS["carry_context"],
        help="Reset state between sentences instead of reusing the last context.",
    )
    parser.add_argument("--seed", type=int, default=COMMON_DEFAULTS["seed"], help="Random seed for reproducibility.")
    parser.add_argument("--lowercase", action="store_true", default=COMMON_DEFAULTS["lowercase"], help="Lowercase the dataset before training.")
    parser.add_argument(
        "--preserve-linebreaks",
        action="store_true",
        default=COMMON_DEFAULTS["preserve_linebreaks"],
        help="Treat line breaks as tokens to keep dialogue-style formatting.",
    )


def run_generation(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    jokes = load_jokes(
        args.dataset,
        encoding=args.encoding,
        fallback_encoding=args.fallback_encoding,
        errors=args.errors,
        lowercase=args.lowercase,
        preserve_linebreaks=args.preserve_linebreaks,
    )
    sequences = build_sequences(jokes, args.min_tokens, args.split_sentences, args.preserve_linebreaks)
    if not sequences:
        raise SystemExit("Dataset is empty after filtering; adjust --min-tokens or check the input file.")
    chain = build_chain(sequences, args.order)
    known_ngrams = collect_ngrams(sequences, args.order)

    for idx in range(args.count):
        joke, tokens = generate_joke(
            chain,
            args.order,
            args.max_length,
            rng,
            args.sentences_per_joke,
            args.carry_context,
        )
        uniqueness = uniqueness_score(tokens, known_ngrams, args.order)
        print(f"{idx + 1:02d}: {joke} (uniqueness: {uniqueness:.1f}%)")


def _prompt_value(label: str, default: Any, cast) -> Any:
    raw = input(f"{label} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return cast(raw)
    except Exception:
        print("Неверный ввод, используем значение по умолчанию.")
        return default


def _prompt_bool(label: str, default: bool) -> bool:
    raw = input(f"{label} [{ 'Y/n' if default else 'y/N' }]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "1", "true", "t", "д", "да", "+"}


def interactive_args() -> argparse.Namespace:
    args = argparse.Namespace()
    args.dataset = Path(_prompt_value("Dataset path", COMMON_DEFAULTS["dataset"], str))
    args.encoding = _prompt_value("Encoding", COMMON_DEFAULTS["encoding"], str)
    args.fallback_encoding = _prompt_value("Fallback encoding", COMMON_DEFAULTS["fallback_encoding"], str)
    args.errors = _prompt_value("Encoding errors mode", COMMON_DEFAULTS["errors"], str)
    args.order = _prompt_value("Markov order (n-gram)", COMMON_DEFAULTS["order"], int)
    args.min_tokens = _prompt_value("Min tokens per sequence", COMMON_DEFAULTS["min_tokens"], int)
    args.split_sentences = _prompt_bool("Split by sentences", COMMON_DEFAULTS["split_sentences"])
    args.count = _prompt_value("How many jokes to generate", COMMON_DEFAULTS["count"], int)
    args.max_length = _prompt_value("Max tokens per joke", COMMON_DEFAULTS["max_length"], int)
    args.sentences_per_joke = _prompt_value("Sentences per joke", COMMON_DEFAULTS["sentences_per_joke"], int)
    args.carry_context = _prompt_bool("Carry context between sentences", COMMON_DEFAULTS["carry_context"])
    seed_raw = input(f"Seed (пусто — без фиксации) [{COMMON_DEFAULTS['seed']}]: ").strip()
    args.seed = int(seed_raw) if seed_raw else None
    args.lowercase = _prompt_bool("Lowercase dataset", COMMON_DEFAULTS["lowercase"])
    args.preserve_linebreaks = _prompt_bool("Preserve line breaks", COMMON_DEFAULTS["preserve_linebreaks"])
    return args


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate jokes with explicit parameters.")
    add_common_arguments(generate_parser)

    subparsers.add_parser("interactive", help="Interactive prompts to configure and generate jokes.")
    return parser.parse_args()


def main() -> None:
    args = parse_cli()
    if args.command == "interactive":
        run_generation(interactive_args())
    else:
        run_generation(args)


if __name__ == "__main__":
    main()
