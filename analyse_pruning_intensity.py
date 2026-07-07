import argparse
import json


POSITIONS = {
    "a": (0, 0),
    "b": (0, 1),
    "c": (1, 0),
    "d": (1, 1),
}


def is_2x2_vector(value):
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(
            isinstance(row, list)
            and len(row) == 2
            and all(isinstance(item, (int, bool)) for item in row)
            for row in value
        )
    )


def iter_2x2_vectors(value):
    if is_2x2_vector(value):
        yield value
        return

    if isinstance(value, list):
        for item in value:
            yield from iter_2x2_vectors(item)


def analyze_intensity(path, key):
    ones = {name: 0 for name in POSITIONS}
    total = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            if key not in record:
                raise KeyError(f"Line {line_idx} does not contain key '{key}'")

            for vector in iter_2x2_vectors(record[key]):
                total += 1
                for name, (row, col) in POSITIONS.items():
                    ones[name] += int(vector[row][col] == 1)

    if total == 0:
        raise ValueError(f"No 2x2 vectors found under key '{key}' in {path}")

    return {
        name: {
            "ones": count,
            "total": total,
            "intensity": count / total,
        }
        for name, count in ones.items()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute position-wise intensity scores for 2x2 pruning vectors."
    )
    parser.add_argument(
        "--path",
        default="pruning_vectors_text.jsonl",
        help="Input JSONL file.",
    )
    parser.add_argument(
        "--key",
        default="keep",
        help="JSON key containing 2x2 vectors or nested lists of 2x2 vectors.",
    )
    args = parser.parse_args()

    result = analyze_intensity(args.path, args.key)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
