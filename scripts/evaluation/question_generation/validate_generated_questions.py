#!/usr/bin/env python3
"""Lightweight validator for generated VQA questions.

Checks:
- Empty/invalid answers
- Basic question formatting
- Counts per question_type

Usage:
  python validate_generated_questions.py output/generated_questions.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python validate_generated_questions.py <generated_questions.json>")
        return 2

    path = sys.argv[1]
    with open(path, "r") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    print(f"Loaded {len(questions)} questions")

    type_counts = Counter()
    empty_answers = 0
    invalid_answers = 0
    bad_questions = 0
    per_image = defaultdict(int)

    for q in questions:
        qtype = q.get("question_type", "(missing)")
        type_counts[qtype] += 1
        per_image[q.get("image_id")] += 1

        ans = q.get("answer")
        if ans is None or str(ans).strip() == "":
            empty_answers += 1
        if str(ans).strip() == "__INVALID__":
            invalid_answers += 1

        qt = str(q.get("question", ""))
        if not qt or "?" not in qt:
            bad_questions += 1

    print("Counts by question_type:")
    for k, v in type_counts.most_common():
        print(f"  {k}: {v}")

    print(f"Empty answers: {empty_answers}")
    print(f"Invalid answers: {invalid_answers}")
    print(f"Bad question strings: {bad_questions}")

    # small sanity on image coverage
    if per_image:
        min_q = min(per_image.values())
        max_q = max(per_image.values())
        print(f"Questions per image: min={min_q} max={max_q} images={len(per_image)}")

    # Fail exit code on serious issues
    if empty_answers or invalid_answers:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
