#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py

Pipeline entrypoint:
  1) relationship_extraction
  2) metadata_enrichment (optional)
  3) data_evaluation (optional)

Rules:
- If --enriched_out is not provided, Stage 2 is skipped.
- If --evaluation_out is not provided, Stage 3 is skipped.
- Stage 3 requires Stage 2 output.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from relationship_extraction import extract_relationships_from_md
from metadata_enrichment import add_metadata_from_paths
from data_evaluation import (
    evaluate_one_row,
    load_existing_output,
    merge_reviewer_outputs,
    parse_reviewers_arg,
    save_json,
    upsert_one_result_by_id,
    validate_input_rows,
)


def ensure_parent_dir(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str, obj: Any) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the QoS-QoE extraction, enrichment, and evaluation pipeline."
    )

    parser.add_argument("--md", required=True, help="Path to the paper markdown file.")
    parser.add_argument(
        "--relationship_prompt",
        required=True,
        help="Path to the relationship extraction prompt.",
    )
    parser.add_argument(
        "--relationships_out",
        required=True,
        help="Path to save stage 1 relationship extraction output.",
    )
    parser.add_argument(
        "--paper_csv",
        default="",
        help=(
            "Path to the paper metadata CSV used to attach year and venue during "
            "relationship extraction."
        ),
    )

    parser.add_argument(
        "--metadata_prompt",
        default="",
        help="Path to the metadata enrichment prompt. Required if --enriched_out is set.",
    )
    parser.add_argument(
        "--enriched_out",
        default="",
        help="Path to save stage 2 metadata enrichment output. If omitted, Stage 2 is skipped.",
    )

    parser.add_argument(
        "--evaluation_system_prompt",
        default="",
        help="Path to the evaluation system prompt. Required if --evaluation_out is set.",
    )
    parser.add_argument(
        "--evaluation_out",
        default="",
        help="Path to save stage 3 data evaluation output. If omitted, Stage 3 is skipped.",
    )

    parser.add_argument("--model", default="gpt-5.2", help="OpenAI model for stages 1 and 2.")
    parser.add_argument(
        "--reasoning_effort",
        default="high",
        choices=["low", "medium", "high"],
        help="Reasoning effort for stages 1 and 2.",
    )
    parser.add_argument("--max_images", type=int, default=24)
    parser.add_argument("--no_compact", action="store_true")

    parser.add_argument(
        "--reviewers",
        default="claude,gemini,grok",
        help="Comma-separated reviewers for stage 3.",
    )
    parser.add_argument("--claude-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash-lite")
    parser.add_argument("--grok-model", default="grok-4.20-0309-reasoning")

    return parser.parse_args()


def validate_stage_dependencies(args: argparse.Namespace) -> None:
    """Validate optional stage dependencies."""
    if args.enriched_out and not args.metadata_prompt:
        raise ValueError("--metadata_prompt is required when --enriched_out is provided.")

    if args.evaluation_out and not args.enriched_out:
        raise ValueError("--evaluation_out requires --enriched_out, because evaluation needs enriched rows.")

    if args.evaluation_out and not args.evaluation_system_prompt:
        raise ValueError("--evaluation_system_prompt is required when --evaluation_out is provided.")


def main() -> None:
    args = parse_args()
    validate_stage_dependencies(args)

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    md_path = os.path.abspath(args.md)
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"Markdown not found: {md_path}")

    compact = not args.no_compact
    client = OpenAI(api_key=openai_api_key)

    paper_csv = os.path.abspath(args.paper_csv) if args.paper_csv else None

    # -----------------------
    # Stage 1: relationships
    # -----------------------
    relationships = extract_relationships_from_md(
        md_path=md_path,
        prompt_file=os.path.abspath(args.relationship_prompt),
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_images=args.max_images,
        compact=compact,
        client=client,
        paper_csv=paper_csv,
    )
    write_json(os.path.abspath(args.relationships_out), relationships)
    print(f"[ok] wrote relationships: {args.relationships_out} (n={len(relationships)})")

    if not relationships:
        print("[skip] no relationships extracted; downstream stages skipped.")
        print("[done] pipeline complete")
        return

    # -----------------------
    # Stage 2: metadata enrichment
    # -----------------------
    if not args.enriched_out:
        print("[skip] --enriched_out not provided; skipping metadata enrichment and evaluation.")
        print("[done] pipeline complete")
        return

    enriched = add_metadata_from_paths(
        md_path=md_path,
        relationships=relationships,
        metadata_prompt_file=os.path.abspath(args.metadata_prompt),
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_images=args.max_images,
        compact=compact,
        client=client,
    )
    write_json(os.path.abspath(args.enriched_out), enriched)
    print(f"[ok] wrote enriched rows: {args.enriched_out} (n={len(enriched)})")

    # -----------------------
    # Stage 3: data evaluation
    # -----------------------
    if not args.evaluation_out:
        print("[skip] --evaluation_out not provided; skipping data evaluation.")
        print("[done] pipeline complete")
        return

    selected_reviewers = parse_reviewers_arg(args.reviewers)
    rows = validate_input_rows(enriched)
    existing_results = load_existing_output(os.path.abspath(args.evaluation_out))

    for row_position, row in enumerate(rows):
        row_id = row["id"]

        reviewer_outputs = evaluate_one_row(
            row=row,
            row_id=row_id,
            system_prompt_path=os.path.abspath(args.evaluation_system_prompt),
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
        save_json(os.path.abspath(args.evaluation_out), existing_results)

    print(f"[ok] wrote evaluation results: {args.evaluation_out} (n={len(existing_results)})")
    print("[done] pipeline complete")


if __name__ == "__main__":
    main()