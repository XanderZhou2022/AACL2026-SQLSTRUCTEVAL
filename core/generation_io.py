"""Shared CLI I/O; exclusive creation protects previously sampled runs."""
import json
from pathlib import Path
from .chat_client import OpenAIChatClient

def positive_int(value):
    import argparse
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result

def write_records(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()

def generation_metadata(args, client):
    return {"model_name": args.model_name, "num_generations_requested": args.num_generations,
            "temperature": None if client.omit_temperature else getattr(args, "temperature", 1.0),
            "temperature_omitted": client.omit_temperature}

def parse_indices(value, size):
    if value is None:
        return list(range(size))
    result = set()
    for token in value.split(","):
        ends = token.strip().split("-")
        if len(ends) == 1:
            result.add(int(ends[0]))
        elif len(ends) == 2:
            start, end = map(int, ends)
            if end < start:
                raise ValueError("Index ranges must be ascending")
            result.update(range(start, end + 1))
        else:
            raise ValueError("Invalid index range")
    if not result or min(result) < 0 or max(result) >= size:
        raise ValueError("Indices must refer to entries in the dev file")
    return sorted(result)
