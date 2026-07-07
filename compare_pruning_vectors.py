import argparse
import json
from itertools import zip_longest

def compare_jsonl(vision_path, text_path, key):
    equal = 0
    unequal = 0
    total = 0

    with (
        open(vision_path, "r", encoding="utf-8") as vision_file,
        open(text_path, "r", encoding="utf-8") as text_file,
    ):
        for index, (vision_line, text_line) in enumerate(
            zip_longest(vision_file, text_file), start=1
        ):
            vision_object = json.loads(vision_line)
            text_object = json.loads(text_line)
            total += 1

            if vision_object[key] == text_object[key]:
                equal += 1
            else:
                unequal += 1

    equal_percentage = 100 * equal / total
    unequal_percentage = 100 * unequal / total

    print(f"Total corresponding objects: {total}")
    print(f"Equal {key} values: {equal} ({equal_percentage:.2f}%)")
    print(f"Unequal {key} values: {unequal} ({unequal_percentage:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description="Compare corresponding values in two JSONL files.")
    
    parser.add_argument("--vision", default="pruning_vectors_vision.jsonl", help="Path to the vision pruning-vector JSONL file.")
    parser.add_argument("--text", default="pruning_vectors_text.jsonl", help="Path to the text pruning-vector JSONL file.")
    parser.add_argument("--key", default="keep", help="JSON object key whose value should be compared.")
    
    args = parser.parse_args()
    compare_jsonl(args.vision, args.text, args.key)

if __name__ == "__main__":
    main()
