#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
metadata_enrichment.py

Add paper-level metadata to each relationship extracted by
relationship_extraction.py.

Inputs:
- metadata prompt
- paper markdown (optionally compacted) with embedded images
- relationship records from relationship_extraction

Output:
A JSON list of dictionaries. Each dictionary contains:
- relationship identifiers and copied relationship fields
- paper-level metadata inferred from the paper
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from relationship_extraction import (
    build_content_from_markdown,
    compact_markdown,
    read_text_file,
)

# ---------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------

DATA_TYPES = ("equation", "table", "figure")

HISTORY_PAIR: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "metric": {"type": "string"},
        "value": {"anyOf": [{"type": "number"}, {"type": "string"}]},
    },
    "required": ["metric", "value"],
    "additionalProperties": False,
}

DEF_PAIR: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "metric": {"type": "string"},
        "definition": {"type": "string"},
    },
    "required": ["metric", "definition"],
    "additionalProperties": False,
}

HISTORY_POINT: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "time_s": {"type": "number"},
        "qos": {"type": "array", "minItems": 1, "items": HISTORY_PAIR},
        "qoe": {"type": "array", "minItems": 1, "items": HISTORY_PAIR},
    },
    "required": ["time_s", "qos", "qoe"],
    "additionalProperties": False,
}

ENRICHED_ONE: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "year": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "venue": {"type": "string"},
        "domain": {"type": "string"},
        "protocol": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "network_type": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "device_type": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "video_type": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "user_preference": {"type": "string"},
        "scenario": {"type": "string"},
        "history_log": {
            "type": "array",
            "minItems": 6,
            "maxItems": 12,
            "items": HISTORY_POINT,
        },
        "data_type": {"type": "string", "enum": list(DATA_TYPES)},
        "qos_parameter": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "qos_parameter_definition": {"type": "array", "items": DEF_PAIR, "minItems": 1},
        "qoe_parameter": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "qoe_parameter_definition": {"type": "array", "items": DEF_PAIR, "minItems": 1},
        "relationship": {"type": "string"},
        "description": {"type": "string"},
        "source": {"type": "string"},
    },
    "required": [
        "id",
        "year",
        "venue",
        "domain",
        "protocol",
        "network_type",
        "device_type",
        "video_type",
        "user_preference",
        "scenario",
        "history_log",
        "data_type",
        "qos_parameter",
        "qos_parameter_definition",
        "qoe_parameter",
        "qoe_parameter_definition",
        "relationship",
        "description",
        "source",
    ],
    "additionalProperties": False,
}

SCHEMA_ENRICHED_LIST: Dict[str, Any] = {
    "name": "qos_qoe_metadata_enrichment",
    "schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": ENRICHED_ONE,
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def safe_json_dumps(obj: Any) -> str:
    """Serialize JSON with indentation and UTF-8 characters preserved."""
    return json.dumps(obj, ensure_ascii=False, indent=2)


def ensure_list_of_dicts(value: Any) -> List[Dict[str, Any]]:
    """Return only dictionary items from a list-like value."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def get_year_venue_from_relationship(relationship: Dict[str, Any]) -> Dict[str, Any]:
    """Extract year and venue from a relationship record."""
    year = relationship.get("year", None)
    venue = relationship.get("venue", "")
    if venue is None:
        venue = ""
    return {"year": year, "venue": venue}


def get_id_from_relationship(relationship: Dict[str, Any]) -> str:
    """Extract the required id field from a relationship record."""
    value = relationship.get("id")
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    raise RuntimeError("Relationship is missing required 'id' field.")


def reorder_enriched_keys(item: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict with a stable preferred key order."""
    preferred = [
        "id",
        "year",
        "venue",
        "domain",
        "protocol",
        "network_type",
        "device_type",
        "video_type",
        "user_preference",
        "scenario",
        "history_log",
        "data_type",
        "qos_parameter",
        "qos_parameter_definition",
        "qoe_parameter",
        "qoe_parameter_definition",
        "relationship",
        "description",
        "source",
    ]

    ordered: Dict[str, Any] = {}
    for key in preferred:
        if key in item:
            ordered[key] = item[key]

    for key, value in item.items():
        if key not in ordered:
            ordered[key] = value

    return ordered


def ensure_str_list(value: Any) -> List[str]:
    """
    Coerce a value into a deduplicated list[str].

    - If given a string, wrap it as a one-item list
    - Strip whitespace
    - Drop empty entries
    - Preserve order
    """
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []

    if not isinstance(value, list):
        return []

    out: List[str] = []
    seen = set()
    for item in value:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if stripped and stripped not in seen:
            out.append(stripped)
            seen.add(stripped)
    return out


# ---------------------------------------------------------------------
# Core enrichment
# ---------------------------------------------------------------------

def add_metadata(
    *,
    client: OpenAI,
    metadata_prompt: str,
    paper_md: str,
    md_dir: str,
    relationships: List[Dict[str, Any]],
    model: str,
    reasoning_effort: str = "high",
    max_images: int = 24,
    compact: bool = True,
    instructions: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Enrich extracted relationships with paper-level metadata in one model call.

    Returns a list of dictionaries with metadata fields and copied relationship fields.
    """
    input_relationships = ensure_list_of_dicts(relationships)

    if compact:
        paper_md = compact_markdown(paper_md)

    relationships_block = (
        "\n\n--- RELATIONSHIPS (JSON FROM relationship_extraction) ---\n"
        + safe_json_dumps(input_relationships)
        + "\n"
    )

    content = build_content_from_markdown(
        prompt=metadata_prompt,
        md_payload=paper_md,
        md_dir=md_dir,
        max_images=max_images,
    )
    content.append({"type": "input_text", "text": relationships_block})

    if instructions is None:
        instructions = (
            "You are an experienced multimedia networking researcher and meticulous "
            "data curator specializing in video streaming. "
            "Follow the user-provided prompt exactly."
        )

    response = client.responses.create(
        model=model,
        reasoning={"effort": reasoning_effort},
        instructions=instructions,
        input=[{"role": "user", "content": content}],
        text={
            "format": {
                "type": "json_schema",
                "name": SCHEMA_ENRICHED_LIST["name"],
                "schema": SCHEMA_ENRICHED_LIST["schema"],
                "strict": True,
            }
        },
    )

    output_text = getattr(response, "output_text", None)
    if not output_text or not str(output_text).strip():
        raise RuntimeError("OpenAI response has no output_text; could not parse JSON.")

    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse model JSON output: {exc}\n---RAW---\n{output_text[:2000]}"
        ) from exc

    items = ensure_list_of_dicts(parsed.get("items", []))

    if len(items) != len(input_relationships):
        raise RuntimeError(
            f"Model returned {len(items)} item(s), expected {len(input_relationships)}."
        )

    for i, item in enumerate(items):
        item["protocol"] = ensure_str_list(item.get("protocol"))
        item["network_type"] = ensure_str_list(item.get("network_type"))
        item["device_type"] = ensure_str_list(item.get("device_type"))
        item["video_type"] = ensure_str_list(item.get("video_type"))

        if not item["protocol"]:
            raise RuntimeError("Model produced an empty protocol list.")
        if not item["network_type"]:
            raise RuntimeError("Model produced an empty network_type list.")
        if not item["device_type"]:
            raise RuntimeError("Model produced an empty device_type list.")
        if not item["video_type"]:
            raise RuntimeError("Model produced an empty video_type list.")

        item["id"] = get_id_from_relationship(input_relationships[i])

        year_venue = get_year_venue_from_relationship(input_relationships[i])
        item["year"] = year_venue["year"]
        item["venue"] = year_venue["venue"]

        items[i] = reorder_enriched_keys(item)

    return items


def add_metadata_from_paths(
    *,
    md_path: str,
    relationships: List[Dict[str, Any]],
    metadata_prompt_file: str,
    model: str = "gpt-5.2",
    reasoning_effort: str = "medium",
    max_images: int = 24,
    compact: bool = True,
    api_key: Optional[str] = None,
    client: Optional[OpenAI] = None,
    instructions: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Read the prompt and paper Markdown from disk, then enrich relationships.
    """
    md_path = os.path.abspath(md_path)
    md_dir = os.path.dirname(md_path)

    if not os.path.exists(metadata_prompt_file):
        raise FileNotFoundError(f"Metadata prompt file not found: {metadata_prompt_file}")
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    metadata_prompt = read_text_file(metadata_prompt_file).strip()
    if not metadata_prompt:
        raise ValueError(f"Metadata prompt file is empty: {metadata_prompt_file}")

    paper_md = read_text_file(md_path)

    if client is None:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Please set it in the environment or pass api_key."
            )
        client = OpenAI(api_key=api_key)

    return add_metadata(
        client=client,
        metadata_prompt=metadata_prompt,
        paper_md=paper_md,
        md_dir=md_dir,
        relationships=relationships,
        model=model,
        reasoning_effort=reasoning_effort,
        max_images=max_images,
        compact=compact,
        instructions=instructions,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Add metadata to relationship records extracted from a Markdown paper."
    )
    parser.add_argument("--md", required=True, help="Path to the Markdown paper")
    parser.add_argument("--rels", required=True, help="Path to the relationships JSON file")
    parser.add_argument(
        "--metadata_prompt_file",
        required=True,
        help="Path to the metadata prompt file",
    )
    parser.add_argument("--model", default="gpt-5.2", help="OpenAI model name")
    parser.add_argument("--out", default="enriched.json", help="Output JSON path")
    parser.add_argument("--max_images", type=int, default=24, help="Maximum number of images to attach")
    parser.add_argument(
        "--reasoning_effort",
        default="high",
        choices=["none", "low", "medium", "high"],
        help="Reasoning effort passed to the Responses API",
    )
    parser.add_argument(
        "--no_compact",
        action="store_true",
        help="Disable Markdown compaction",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    relationships_obj = json.loads(read_text_file(args.rels))
    if isinstance(relationships_obj, dict) and "relationships" in relationships_obj:
        relationships = relationships_obj["relationships"]
    else:
        relationships = relationships_obj

    enriched = add_metadata_from_paths(
        md_path=args.md,
        relationships=relationships,
        metadata_prompt_file=args.metadata_prompt_file,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_images=args.max_images,
        compact=not args.no_compact,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    print(f"[OK] Wrote {len(enriched)} enriched item(s) -> {args.out}")


if __name__ == "__main__":
    main()