#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_evaluation.py

Run multi-reviewer evaluation for structured dataset rows and merge the reviewer
outputs into a single result per row.

Input:
- a JSON file containing a list of row dictionaries, each with a required "id"

Output:
- a JSON file containing merged evaluation results, updated by row id
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

ALL_REVIEWER_NAMES = ("claude", "gemini", "grok")


def load_json(path: str) -> Any:
    """Load JSON from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    """Save JSON to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def validate_input_rows(rows: Any) -> List[Dict[str, Any]]:
    """Validate the input row list."""
    if not isinstance(rows, list):
        raise ValueError("Input JSON must be a list of dict rows.")

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {i} is not a dict.")

        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id.strip():
            raise ValueError(f"Row {i} is missing a valid non-empty string 'id'.")

    return rows


def validate_existing_results(results: Any) -> List[Dict[str, Any]]:
    """Validate an existing merged-output JSON structure."""
    if not isinstance(results, list):
        raise ValueError("Existing output JSON must be a list of dict rows.")

    for i, item in enumerate(results):
        if not isinstance(item, dict):
            raise ValueError(f"Existing output item {i} is not a dict.")

        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError(
                f"Existing output item {i} is missing a valid non-empty string 'id'."
            )

    return results


def load_existing_output(path: str) -> List[Dict[str, Any]]:
    """Load existing output JSON if present; otherwise return an empty list."""
    if not os.path.exists(path):
        return []
    return validate_existing_results(load_json(path))


def parse_reviewers_arg(reviewers_text: str) -> List[str]:
    """Parse and validate the --reviewers argument."""
    reviewers = [x.strip().lower() for x in reviewers_text.split(",") if x.strip()]
    if not reviewers:
        raise ValueError("At least one reviewer must be provided in --reviewers.")

    invalid = [x for x in reviewers if x not in ALL_REVIEWER_NAMES]
    if invalid:
        raise ValueError(
            f"Invalid reviewer(s): {invalid}. Allowed reviewers: {list(ALL_REVIEWER_NAMES)}"
        )

    seen = set()
    ordered: List[str] = []
    for name in reviewers:
        if name not in seen:
            seen.add(name)
            ordered.append(name)

    return ordered


def safe_round(x: float, ndigits: int = 2) -> float:
    """Round a numeric value safely."""
    return round(float(x), ndigits)


def compute_average(values: List[float]) -> float:
    """Compute the mean of a list of numbers."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def merge_unique_strings(items: List[str]) -> List[str]:
    """Merge strings while preserving order and removing duplicates."""
    seen = set()
    merged: List[str] = []
    for item in items:
        key = item.strip()
        if key and key not in seen:
            seen.add(key)
            merged.append(key)
    return merged


def choose_final_decision(avg_rating: Optional[float], avg_confidence: Optional[float]) -> str:
    """Choose the final decision from aggregate scores."""
    if avg_rating is None or avg_confidence is None:
        return "review_error"
    if avg_rating >= 8.0 and avg_confidence >= 3.0:
        return "accept"
    if avg_rating <= 4.0:
        return "reject"
    return "conditional_accept"


def is_successful_reviewer_output(result: Any) -> bool:
    """
    Return True only for reviewer outputs that should count in aggregation.

    Reviewer modules should set:
      review_status = "success" for valid model outputs
      review_status = "error"   for fallback/error outputs
    """
    return isinstance(result, dict) and result.get("review_status") == "success"


def choose_resolved_data_row(
    original_row: Dict[str, Any],
    reviewer_outputs: Dict[str, Dict[str, Any]],
    selected_reviewers: List[str],
) -> Dict[str, Any]:
    """
    Choose the resolved row from the highest-rated successful reviewer.
    Ties are broken by confidence. Falls back to the original row.
    """
    best_name: Optional[str] = None
    best_rating = -1
    best_confidence = -1

    for name in selected_reviewers:
        result = reviewer_outputs.get(name)
        if not is_successful_reviewer_output(result):
            continue

        rating = result.get("rating", -1)
        confidence = result.get("confidence", -1)

        if rating > best_rating:
            best_name = name
            best_rating = rating
            best_confidence = confidence
        elif rating == best_rating and confidence > best_confidence:
            best_name = name
            best_confidence = confidence

    if best_name is None:
        return copy.deepcopy(original_row)

    chosen = reviewer_outputs[best_name].get("resolved_data_row")
    if isinstance(chosen, dict):
        return copy.deepcopy(chosen)

    return copy.deepcopy(original_row)


def merge_reviewer_outputs(
    original_row: Dict[str, Any],
    row_id: str,
    row_position: int,
    reviewer_outputs: Dict[str, Dict[str, Any]],
    selected_reviewers: List[str],
) -> Dict[str, Any]:
    """Merge reviewer outputs into one final record."""
    ratings: List[int] = []
    confidences: List[int] = []
    all_concerns: List[str] = []
    all_solutions: List[str] = []
    successful_reviewers: List[str] = []
    failed_reviewers: List[str] = []

    for name in selected_reviewers:
        result = reviewer_outputs.get(name)
        if not isinstance(result, dict):
            failed_reviewers.append(name)
            continue

        if is_successful_reviewer_output(result):
            successful_reviewers.append(name)

            rating = result.get("rating")
            confidence = result.get("confidence")

            if isinstance(rating, int):
                ratings.append(rating)
            if isinstance(confidence, int):
                confidences.append(confidence)

            concerns = result.get("concerns", [])
            solutions = result.get("solutions", [])

            if isinstance(concerns, list):
                all_concerns.extend(x for x in concerns if isinstance(x, str))
            if isinstance(solutions, list):
                all_solutions.extend(x for x in solutions if isinstance(x, str))
        else:
            failed_reviewers.append(name)

    avg_rating: Optional[float] = None
    avg_confidence: Optional[float] = None

    if ratings:
        avg_rating = safe_round(compute_average(ratings), 2)
    if confidences:
        avg_confidence = safe_round(compute_average(confidences), 2)

    final_decision = choose_final_decision(avg_rating, avg_confidence)

    return {
        "row_position": row_position,
        "id": row_id,
        "selected_reviewers": selected_reviewers,
        "successful_reviewers": successful_reviewers,
        "failed_reviewers": failed_reviewers,
        "reviewer_outputs": reviewer_outputs,
        "avg_rating": avg_rating,
        "avg_confidence": avg_confidence,
        "final_decision": final_decision,
        "merged_concerns": merge_unique_strings(all_concerns),
        "merged_solutions": merge_unique_strings(all_solutions),
        "resolved_data_row": choose_resolved_data_row(
            original_row=original_row,
            reviewer_outputs=reviewer_outputs,
            selected_reviewers=selected_reviewers,
        ),
    }


def upsert_one_result_by_id(
    existing_results: List[Dict[str, Any]],
    new_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Insert or replace one merged result by id."""
    new_id = new_result.get("id")
    if not isinstance(new_id, str) or not new_id.strip():
        raise ValueError("New merged result is missing a valid non-empty string 'id'.")

    for i, item in enumerate(existing_results):
        if item.get("id") == new_id:
            existing_results[i] = new_result
            break
    else:
        existing_results.append(new_result)

    existing_results.sort(
        key=lambda x: (
            x.get("row_position", float("inf")),
            x.get("id", ""),
        )
    )
    return existing_results


def load_reviewer_function(name: str) -> Callable[..., Dict[str, Any]]:
    """Dynamically load the evaluation function for a reviewer."""
    module_name = f"data_reviewers.{name}"
    function_name = f"evaluate_with_{name}"

    module = importlib.import_module(module_name)
    fn = getattr(module, function_name, None)
    if fn is None or not callable(fn):
        raise RuntimeError(
            f"Reviewer module '{module_name}' does not define callable '{function_name}'."
        )
    return fn


def evaluate_one_row(
    row: Dict[str, Any],
    row_id: str,
    system_prompt_path: str,
    claude_model: str,
    gemini_model: str,
    grok_model: str,
    selected_reviewers: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Run the selected reviewers on one row.

    Reviewer-specific modules are responsible for building their own user prompts
    or request payloads from the input row.
    """
    model_by_reviewer = {
        "claude": claude_model,
        "gemini": gemini_model,
        "grok": grok_model,
    }

    reviewer_outputs: Dict[str, Dict[str, Any]] = {}

    for name in selected_reviewers:
        evaluate_fn = load_reviewer_function(name)
        reviewer_outputs[name] = evaluate_fn(
            row=row,
            row_id=row_id,
            system_prompt_path=system_prompt_path,
            model=model_by_reviewer[name],
        )

    return reviewer_outputs


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run multi-model data quality evaluation."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSON file containing a list of row dictionaries.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Path to merged output JSON file. "
            "If it already exists, finished rows are updated in place by id."
        ),
    )
    parser.add_argument(
        "--system-prompt",
        required=True,
        help="Path to system prompt file.",
    )
    parser.add_argument(
        "--reviewers",
        default="claude,gemini,grok",
        help="Comma-separated reviewers to run, for example: claude or claude,gemini",
    )
    parser.add_argument(
        "--claude-model",
        default="claude-haiku-4-5-20251001",
        help="Claude model name.",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-2.5-flash-lite",
        help="Gemini model name.",
    )
    parser.add_argument(
        "--grok-model",
        default="grok-4.20-0309-reasoning",
        help="Grok model name.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start row position (inclusive) in the input list.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="End row position (exclusive). Use -1 to evaluate through the end.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    selected_reviewers = parse_reviewers_arg(args.reviewers)
    rows = validate_input_rows(load_json(args.input))
    existing_results = load_existing_output(args.output)

    start = max(0, args.start)
    end = len(rows) if args.end == -1 else min(len(rows), args.end)

    if start >= end:
        raise ValueError(
            "Invalid slice range: start={0}, end={1}. "
            "Remember: --start is inclusive and --end is exclusive. "
            "To evaluate only the first row, use --start 0 --end 1.".format(start, end)
        )

    sliced_rows = rows[start:end]

    logger.info("Loaded %d rows from %s", len(rows), args.input)
    logger.info("Selected reviewers: %s", selected_reviewers)
    logger.info(
        "Evaluating rows in range [%d, %d) -> %d rows",
        start,
        end,
        len(sliced_rows),
    )

    for local_i, row in enumerate(sliced_rows):
        row_position = start + local_i
        row_id = row["id"]

        logger.info(
            "[%d/%d] Evaluating row position %d, id=%s",
            local_i + 1,
            len(sliced_rows),
            row_position,
            row_id,
        )

        reviewer_outputs = evaluate_one_row(
            row=row,
            row_id=row_id,
            system_prompt_path=args.system_prompt,
            claude_model=args.claude_model,
            gemini_model=args.gemini_model,
            grok_model=args.grok_model,
            selected_reviewers=selected_reviewers,
        )

        merged_result = merge_reviewer_outputs(
            original_row=row,
            row_id=row_id,
            row_position=row_position,
            reviewer_outputs=reviewer_outputs,
            selected_reviewers=selected_reviewers,
        )

        existing_results = upsert_one_result_by_id(existing_results, merged_result)
        save_json(args.output, existing_results)

        logger.info(
            "Saved row position %d, id=%s to %s",
            row_position,
            row_id,
            args.output,
        )

    logger.info("Finished evaluation for range [%d, %d)", start, end)


if __name__ == "__main__":
    main()