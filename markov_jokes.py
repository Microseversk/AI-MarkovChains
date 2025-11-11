#!/usr/bin/env python3
"""CLI tool to generate jokes using Markov chains built from a text corpus."""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Iterable, List, Sequence, Tuple

START_TOKEN = "<START>"
END_TOKEN = "<END>"
TOKEN_PATTERN = re.compile(r"[A-Za-z\u0410-\u044f\u0401\u04510-9']+|[.,!?;:()\u00ab\u00bb\"-]", re.UNICODE)
NO_SPACE_BEFORE = set(".,!?;:)]}\"\u00bb>")
NO_SPACE_AFTER = set("([{\u00ab<")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def iter_lines(path: Path, encoding: str, fallback_encoding: str | None, errors: str) -> Iterable[str]:
    """Yield lines from file, retrying with fallback encoding when needed."""
    try:
        with path.open("r", encoding=encoding, errors=errors) as handle:
            for line in handle:
                yield line
        return
    except UnicodeDecodeError:
        if not fallback_encoding:
            raise

    with path.open("r", encoding=fallback_encoding, errors=errors) as handle:
        for line in handle:
            yield line


def load_jokes(
    path: Path,
    encoding: str = "utf-8",
    fallback_encoding: str | None = None,
    errors: str = "strict",
    lowercase: bool = False,
) -> List[str]:
    """Read jokes separated by blank lines, return cleaned strings."""
    jokes: List[str] = []
    chunk: List[str] = []

    for raw_line in iter_lines(path, encoding, fallback_encoding, errors):
        line = raw_line.strip()
        if not line:
            if chunk:
                jokes.append(process_chunk(chunk, lowercase))
                chunk = []
            continue
        chunk.append(line)

    if chunk:
        jokes.append(process_chunk(chunk, lowercase))

    return jokes


def process_chunk(lines: List[str], lowercase: bool) -> str:
    """Join and normalize a single joke chunk."""
    text = " ".join(lines)
    return text.lower() if lowercase else text


def tokenize(text: str) -> List[str]:
    """Split text into tokens of words and punctuation."""
    return TOKEN_PATTERN.findall(text)


def split_into_sentences(text: str) -> List[str]:
    """Split paragraph into sentence-like segments."""
    parts = [segment.strip() for segment in SENTENCE_SPLIT_PATTERN.split(text) if segment.strip()]
    return parts or [text]


def build_chain(sequences: Iterable[Sequence[str]], order: int) -> DefaultDict[Tuple[str, ...], Counter]:
    """Create Markov transition counts for n-gram order."""
    if order < 1:
        raise ValueError("Order of the chain must be >= 1")

    chain: DefaultDict[Tuple[str, ...], Counter] = defaultdict(Counter)

    for seq in sequences:
        padded = [START_TOKEN] * order + list(seq) + [END_TOKEN]
        for idx in range(len(padded) - order):
            state = tuple(padded[idx : idx + order])
            nxt = padded[idx + order]
            chain[state][nxt] += 1

    return chain


def select_context_state(
    chain: DefaultDict[Tuple[str, ...], Counter],
    order: int,
    history: Sequence[str],
    carry_context: bool,
) -> Tuple[str, ...]:
    """Derive the next state after finishing a sentence."""
    if carry_context and len(history) >= order:
        candidate = tuple(history[-order:])
        options = chain.get(candidate)
        if options:
            non_terminal = sum(weight for token, weight in options.items() if token != END_TOKEN)
            if non_terminal:
                return candidate
    return tuple([START_TOKEN] * order)


def generate_joke(
    chain: DefaultDict[Tuple[str, ...], Counter],
    order: int,
    max_tokens: int,
    rng: random.Random,
    sentences_per_joke: int,
    carry_context: bool,
) -> str:
    """Sample a multi-sentence joke from transition counts."""
    state = tuple([START_TOKEN] * order)
    generated: List[str] = []
    tokens_used = 0
    sentences_done = 0

    while tokens_used < max_tokens and sentences_done < sentences_per_joke:
        options = chain.get(state)
        if not options:
            state = tuple([START_TOKEN] * order)
            options = chain.get(state)
            if not options:
                break
        tokens, weights = zip(*options.items())
        nxt = rng.choices(tokens, weights=weights, k=1)[0]
        if nxt == END_TOKEN:
            sentences_done += 1
            if sentences_done >= sentences_per_joke:
                break
            state = select_context_state(chain, order, generated, carry_context)
            continue
        generated.append(nxt)
        tokens_used += 1
        state = (*state[1:], nxt) if order > 1 else (nxt,)

    return untokenize(generated)


def untokenize(tokens: Sequence[str]) -> str:
    """Join tokens back into a readable string."""
    if not tokens:
        return ""

    pieces: List[str] = []
    for token in tokens:
        if not pieces:
            pieces.append(token)
            continue

        if token in NO_SPACE_BEFORE:
            pieces[-1] += token
        elif pieces[-1][-1] in NO_SPACE_AFTER:
            pieces[-1] += token
        else:
            pieces.append(f" {token}")

    return "".join(pieces)


def build_sequences(jokes: Sequence[str], min_tokens: int, sentence_mode: bool) -> List[List[str]]:
    """Convert jokes to token sequences, optionally splitting by sentences."""
    sequences: List[List[str]] = []
    for joke in jokes:
        fragments = split_into_sentences(joke) if sentence_mode else [joke]
        for fragment in fragments:
            tokens = tokenize(fragment)
            if len(tokens) >= min_tokens:
                sequences.append(tokens)
    return sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("full_jokes.txt"), help="Path to the source jokes file.")
    parser.add_argument("--encoding", default="utf-8", help="Primary encoding used to read the dataset.")
    parser.add_argument("--fallback-encoding", dest="fallback_encoding", default="cp1251", help="Fallback encoding.")
    parser.add_argument("--errors", default="ignore", help="Encoding error handling strategy.")
    parser.add_argument("--order", type=int, default=3, help="Order of the Markov chain (n-gram size).")
    parser.add_argument("--min-tokens", type=int, default=5, help="Skip sequences with fewer tokens.")
    parser.add_argument("--split-sentences", action="store_true", help="Train on individual sentences.")
    parser.add_argument("--count", type=int, default=5, help="Number of jokes to generate.")
    parser.add_argument("--max-length", type=int, default=60, help="Maximum tokens per generated joke.")
    parser.add_argument(
        "--sentences-per-joke",
        type=int,
        default=2,
        help="Target number of sentences per generated joke.",
    )
    parser.add_argument(
        "--no-context-carry",
        dest="carry_context",
        action="store_false",
        help="Reset state between sentences instead of reusing the last context.",
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase the dataset before training.")
    parser.set_defaults(carry_context=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    jokes = load_jokes(
        args.dataset,
        encoding=args.encoding,
        fallback_encoding=args.fallback_encoding,
        errors=args.errors,
        lowercase=args.lowercase,
    )
    sequences = build_sequences(jokes, args.min_tokens, args.split_sentences)
    if not sequences:
        raise SystemExit("Dataset is empty after filtering; adjust --min-tokens or check the input file.")
    chain = build_chain(sequences, args.order)

    for idx in range(args.count):
        joke = generate_joke(
            chain,
            args.order,
            args.max_length,
            rng,
            args.sentences_per_joke,
            args.carry_context,
        )
        print(f"{idx + 1:02d}: {joke}")


if __name__ == "__main__":
    main()
