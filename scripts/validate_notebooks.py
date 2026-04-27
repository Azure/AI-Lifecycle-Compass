"""Validate that all .ipynb files are well-formed JSON."""

import glob
import json
import sys


def main() -> int:
    files = glob.glob("**/*.ipynb", recursive=True)
    errors = 0
    for f in files:
        try:
            with open(f) as fh:
                json.load(fh)
        except json.JSONDecodeError as e:
            print(f"{f}: {e}")
            errors += 1
    print(f"Checked {len(files)} notebooks, {errors} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
