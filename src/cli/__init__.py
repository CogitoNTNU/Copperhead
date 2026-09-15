"""Command-line tools for architecture export."""

import argparse
from pathlib import Path

from .flattening import flatten_yaml
from .graphing import graph_yaml


def _arguments(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("source", type=Path, help="Architecture YAML file")
    parser.add_argument("destination", type=Path, help="Output file")
    return parser.parse_args()


def graph() -> None:
    args = _arguments("Draw a network architecture.")
    graph_yaml(args.source, args.destination)


def flatten() -> None:
    args = _arguments("Resolve an architecture into a single YAML file.")
    flatten_yaml(args.source, args.destination)
