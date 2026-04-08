#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
relationship_extraction.py

Extract QoS-QoE relationships from a paper in Markdown format using the
OpenAI Responses API with structured JSON output.

Public API:
    - extract_relationships(...)
    - extract_relationships_from_md(...)

Features:
    - Supports Markdown input with embedded local or remote images
    - Preserves figure/table/equation evidence
    - Optionally compacts Markdown by removing common non-essential sections
    - Optionally enriches extracted relationships with year and venue metadata
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import os
import re
import uuid
from typing import Any, Dict, Iterable, List, Optional

import requests
from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------

DATA_TYPES = ("equation", "table", "figure")

SCHEMA_CORE_ONE: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "data_type": {"type": "string", "enum": list(DATA_TYPES)},
        "qos_parameter": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "qos_parameter_definition": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string"},
                    "definition": {"type": "string"},
                },
                "required": ["metric", "definition"],
                "additionalProperties": False,
            },
        },
        "qoe_parameter": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "qoe_parameter_definition": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string"},
                    "definition": {"type": "string"},
                },
                "required": ["metric", "definition"],
                "additionalProperties": False,
            },
        },
        "relationship": {"type": "string"},
        "description": {"type": "string"},
        "source": {"type": "string"},
    },
    "required": [
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

SCHEMA_CORE_MANY: Dict[str, Any] = {
    "name": "qos_qoe_core_extraction",
    "schema": {
        "type": "object",
        "properties": {
            "relationships": {
                "type": "array",
                "items": SCHEMA_CORE_ONE,
            }
        },
        "required": ["relationships"],
        "additionalProperties": False,
    },
}

# ---------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------

IMG_MD_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
HEADER_RE = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")
PAPER_ID_RE = re.compile(r"(\d+)(?=\.md$)", re.IGNORECASE)

DROP_SECTION_RE = re.compile(
    r"^\s*#{1,6}\s+("
    r"abstract|conclusion|conclusions|related\s+work|references|"
    r"bibliography|acknowledg(e)?ments?|open\s+access"
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------
# File / metadata helpers
# ---------------------------------------------------------------------


def read_text_file(path: str) -> str:
    """Read a UTF-8 text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def paper_id_from_md_path(md_path: str) -> Optional[str]:
    """Extract a numeric paper id from a filename like '15.md'."""
    base = os.path.basename(md_path.strip())
    match = PAPER_ID_RE.search(base)
    return match.group(1) if match else None


def load_paper_meta_map(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load a mapping:
        paper_id -> {"year": int | None, "venue": str}

    Expected CSV columns include:
        Index, Year, Venue
    """
    csv_path = os.path.abspath(csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Paper metadata CSV not found: {csv_path}")

    meta: Dict[str, Dict[str, Any]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = str(row.get("Index", "")).strip()
            if not idx:
                continue

            year_raw = str(row.get("Year", "")).strip()
            venue = str(row.get("Venue", "")).strip()

            year_val: Optional[int] = None
            if year_raw:
                try:
                    year_val = int(float(year_raw))
                except (ValueError, TypeError):
                    year_val = None

            meta[idx] = {
                "year": year_val,
                "venue": venue,
            }

    return meta


def augment_source_with_paper_id(source: Any, paper_id: Optional[str]) -> Any:
    """
    Add paper id into the source string.

    Examples:
        'figure 13a; Title' -> 'figure 13a; [15] Title'
        'Title' -> '[15] Title'

    If the paper id is already present, the source is left unchanged.
    """
    if not paper_id or not isinstance(source, str):
        return source

    tag = f"[{paper_id}]"
    if tag in source:
        return source

    source = source.strip()
    if not source:
        return source

    if ";" in source:
        left, right = source.split(";", 1)
        right = right.lstrip()
        return f"{left.strip()}; {tag} {right}" if right else f"{left.strip()}; {tag}"

    return f"{tag} {source}"


def attach_year_venue(
    relationship: Dict[str, Any],
    paper_id: Optional[str],
    paper_meta_map: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Attach year and venue fields to a relationship dict."""
    if not isinstance(relationship, dict):
        return relationship

    year_val = None
    venue_val = ""

    if paper_id and paper_meta_map and paper_id in paper_meta_map:
        year_val = paper_meta_map[paper_id].get("year")
        venue_val = paper_meta_map[paper_id].get("venue", "") or ""

    relationship["year"] = year_val
    relationship["venue"] = venue_val
    return relationship


def reorder_relationship_keys(relationship: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict with a stable preferred key order."""
    if not isinstance(relationship, dict):
        return relationship

    preferred_order = [
        "id",
        "year",
        "venue",
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

    for key in preferred_order:
        if key in relationship:
            ordered[key] = relationship[key]

    for key, value in relationship.items():
        if key not in ordered:
            ordered[key] = value

    return ordered


# ---------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------


def file_to_data_url(path: str) -> Optional[str]:
    """Convert a local image file to a data URL."""
    extension = os.path.splitext(path)[1].lower()
    mime_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(extension)

    if not mime_type:
        return None

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def url_to_data_url(url: str, timeout_s: int = 20) -> Optional[str]:
    """Download an image URL and convert it to a data URL."""
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=timeout_s,
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "image/jpeg")
        content_type = content_type.split(";")[0].strip()
        if content_type not in {"image/png", "image/jpeg", "image/webp"}:
            content_type = "image/jpeg"

        encoded = base64.b64encode(response.content).decode("utf-8")
        return f"data:{content_type};base64,{encoded}"
    except Exception as exc:
        logger.warning("Failed to download image URL %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------
# Markdown processing
# ---------------------------------------------------------------------


def compact_markdown(md: str) -> str:
    """
    Remove common non-essential sections such as:
    Abstract, Conclusions, Related Work, References, etc.
    """
    lines = md.splitlines()
    drop_mask = [False] * len(lines)

    in_drop_section = False
    drop_level: Optional[int] = None

    def header_level(line: str) -> int:
        match = re.match(r"^\s*(#{1,6})\s+", line)
        return len(match.group(1)) if match else 0

    for i, line in enumerate(lines):
        if HEADER_RE.match(line):
            level = header_level(line)

            if DROP_SECTION_RE.match(line):
                in_drop_section = True
                drop_level = level
                drop_mask[i] = True
                continue

            if in_drop_section and drop_level is not None and level <= drop_level:
                in_drop_section = False
                drop_level = None

        if in_drop_section:
            drop_mask[i] = True

    kept_lines = [line for i, line in enumerate(lines) if not drop_mask[i]]
    return "\n".join(kept_lines)


def build_content_from_markdown(
    prompt: str,
    md_payload: str,
    md_dir: str,
    max_images: int = 24,
) -> List[Dict[str, Any]]:
    """
    Build multimodal Responses API content from Markdown.

    The returned content contains:
    - input_text blocks for text
    - input_image blocks for embedded images
    """
    pieces: List[Dict[str, Any]] = []
    pieces.append(
        {
            "type": "input_text",
            "text": prompt + "\n\n--- PAPER (MARKDOWN) ---\n",
        }
    )

    attached_images = 0
    last_pos = 0

    for match in IMG_MD_RE.finditer(md_payload):
        before = md_payload[last_pos:match.start()]
        if before:
            pieces.append({"type": "input_text", "text": before})

        raw_path = match.group(1).strip().strip('"').strip("'")
        added_image = False

        if attached_images < max_images:
            if raw_path.startswith(("http://", "https://")):
                data_url = url_to_data_url(raw_path)
                if data_url:
                    pieces.append({"type": "input_image", "image_url": data_url})
                    attached_images += 1
                    added_image = True
            else:
                local_path = raw_path
                if not os.path.isabs(local_path):
                    local_path = os.path.abspath(os.path.join(md_dir, local_path))

                if os.path.exists(local_path):
                    data_url = file_to_data_url(local_path)
                    if data_url:
                        pieces.append({"type": "input_image", "image_url": data_url})
                        attached_images += 1
                        added_image = True

        if not added_image:
            pieces.append({"type": "input_text", "text": match.group(0)})

        last_pos = match.end()

    tail = md_payload[last_pos:]
    if tail:
        pieces.append({"type": "input_text", "text": tail})

    content: List[Dict[str, Any]] = []
    text_buffer: List[str] = []

    for piece in pieces:
        if piece["type"] == "input_text":
            text_buffer.append(piece.get("text", ""))
        else:
            if text_buffer:
                content.append({"type": "input_text", "text": "".join(text_buffer)})
                text_buffer = []
            content.append(piece)

    if text_buffer:
        content.append({"type": "input_text", "text": "".join(text_buffer)})

    return [
        item
        for item in content
        if not (item["type"] == "input_text" and not item.get("text", "").strip())
    ]


# ---------------------------------------------------------------------
# Relationship normalization / dedup
# ---------------------------------------------------------------------


def dedup_list_preserve_order(values: Any) -> List[str]:
    """Deduplicate a list of values while preserving order."""
    if not isinstance(values, list):
        return []

    out: List[str] = []
    seen = set()

    for value in values:
        s = str(value).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)

    return out


def pairs_to_map(pairs: Any) -> Dict[str, str]:
    """
    Convert:
        [{"metric": "...", "definition": "..."}, ...]
    into:
        {"...": "..."}
    """
    out: Dict[str, str] = {}
    if not isinstance(pairs, list):
        return out

    for item in pairs:
        if not isinstance(item, dict):
            continue
        key = str(item.get("metric", item.get("parameter", ""))).strip()
        if not key:
            continue

        value = item.get("definition", "")
        out[key] = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    return out


def coerce_pairs_to_exact_params(params: List[str], pairs: Any) -> List[Dict[str, str]]:
    """
    Ensure the returned definition list exactly matches params:
    - same spelling
    - same order
    - missing definitions filled with ""
    - extra definitions dropped
    """
    params = dedup_list_preserve_order(params)
    pair_map = pairs_to_map(pairs)

    return [{"metric": param, "definition": pair_map.get(param, "")} for param in params]


def normalize_and_validate_arrays_and_defs(relationship: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort normalization for backward compatibility:
    - Wrap string qos/qoe parameter values in a list
    - Convert string or dict definitions into standard list-of-dict form
    """
    if isinstance(relationship.get("qos_parameter"), str):
        relationship["qos_parameter"] = [relationship["qos_parameter"].strip()]

    if isinstance(relationship.get("qoe_parameter"), str):
        relationship["qoe_parameter"] = [relationship["qoe_parameter"].strip()]

    qos_params = relationship.get("qos_parameter") or []
    qos_defs = relationship.get("qos_parameter_definition")
    if isinstance(qos_defs, str):
        if len(qos_params) == 1 and isinstance(qos_params[0], str) and qos_params[0].strip():
            relationship["qos_parameter_definition"] = [
                {"metric": qos_params[0].strip(), "definition": qos_defs}
            ]
    elif isinstance(qos_defs, dict):
        relationship["qos_parameter_definition"] = [
            {
                "metric": str(k).strip(),
                "definition": v if isinstance(v, str) else json.dumps(v, ensure_ascii=False),
            }
            for k, v in qos_defs.items()
            if str(k).strip()
        ]

    qoe_params = relationship.get("qoe_parameter") or []
    qoe_defs = relationship.get("qoe_parameter_definition")
    if isinstance(qoe_defs, str):
        if len(qoe_params) == 1 and isinstance(qoe_params[0], str) and qoe_params[0].strip():
            relationship["qoe_parameter_definition"] = [
                {"metric": qoe_params[0].strip(), "definition": qoe_defs}
            ]
    elif isinstance(qoe_defs, dict):
        relationship["qoe_parameter_definition"] = [
            {
                "metric": str(k).strip(),
                "definition": v if isinstance(v, str) else json.dumps(v, ensure_ascii=False),
            }
            for k, v in qoe_defs.items()
            if str(k).strip()
        ]

    return relationship


def dedup_relationships(relationships: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate relationships using a content-based key."""
    def norm_list(value: Any) -> str:
        if not isinstance(value, list):
            return ""
        return ",".join(str(item).strip() for item in value)

    def norm_pairs(value: Any) -> str:
        if not isinstance(value, list):
            return ""

        normalized = []
        for item in value:
            if not isinstance(item, dict):
                continue
            metric = str(item.get("metric", "")).strip()
            definition = item.get("definition", "")
            if not isinstance(definition, str):
                definition = json.dumps(definition, ensure_ascii=False)
            if metric:
                normalized.append((metric, definition))

        normalized.sort(key=lambda x: x[0])
        return json.dumps(normalized, ensure_ascii=False)

    def make_key(relationship: Dict[str, Any]) -> str:
        return "|".join(
            [
                str(relationship.get("data_type", "")).strip(),
                norm_list(relationship.get("qos_parameter", [])),
                norm_pairs(relationship.get("qos_parameter_definition", [])),
                norm_list(relationship.get("qoe_parameter", [])),
                norm_pairs(relationship.get("qoe_parameter_definition", [])),
                str(relationship.get("relationship", "")).strip(),
                str(relationship.get("source", "")).strip(),
            ]
        )

    seen = set()
    out: List[Dict[str, Any]] = []

    for relationship in relationships:
        if not isinstance(relationship, dict):
            continue

        key = make_key(relationship)
        if key in seen:
            continue

        seen.add(key)
        out.append(relationship)

    return out


def add_uuid(relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add a fresh UUID to each relationship under the 'id' key."""
    generated_ids = set()

    for relationship in relationships:
        if not isinstance(relationship, dict):
            continue

        while True:
            new_id = str(uuid.uuid4())
            if new_id not in generated_ids:
                generated_ids.add(new_id)
                relationship["id"] = new_id
                break

    return relationships


# ---------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------


def extract_relationships(
    *,
    client: OpenAI,
    prompt: str,
    paper_md: str,
    md_dir: str,
    model: str,
    reasoning_effort: str = "high",
    max_images: int = 24,
    compact: bool = True,
    instructions: Optional[str] = None,
    paper_id: Optional[str] = None,
    paper_meta_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract QoS-QoE relationships from paper Markdown in a single OpenAI call.

    Returns a deduplicated list of relationship dictionaries, each with a UUID id.
    """
    if compact:
        paper_md = compact_markdown(paper_md)

    content = build_content_from_markdown(
        prompt=prompt,
        md_payload=paper_md,
        md_dir=md_dir,
        max_images=max_images,
    )

    if instructions is None:
        instructions = (
            "You are an experienced multimedia networking researcher and meticulous "
            "data curator specializing in video streaming. Extract ALL and ONLY "
            "QoS-QoE relationships directly supported by the provided paper evidence "
            "(equations, tables, figures). Do not fabricate. Return only JSON "
            "matching the given schema."
        )

    response = client.responses.create(
        model=model,
        reasoning={"effort": reasoning_effort},
        instructions=instructions,
        input=[{"role": "user", "content": content}],
        text={
            "format": {
                "type": "json_schema",
                "name": SCHEMA_CORE_MANY["name"],
                "schema": SCHEMA_CORE_MANY["schema"],
                "strict": True,
            }
        },
    )

    output_text = getattr(response, "output_text", None)
    if not output_text or not str(output_text).strip():
        raise RuntimeError("OpenAI response has no output_text; could not parse JSON output.")

    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse model JSON output: {exc}\n---RAW---\n{output_text[:2000]}"
        ) from exc

    relationships = parsed.get("relationships", [])
    if not isinstance(relationships, list):
        relationships = []

    cleaned: List[Dict[str, Any]] = []

    for relationship in relationships:
        if not isinstance(relationship, dict):
            continue

        relationship = normalize_and_validate_arrays_and_defs(relationship)
        relationship["qos_parameter"] = dedup_list_preserve_order(
            relationship.get("qos_parameter")
        )
        relationship["qoe_parameter"] = dedup_list_preserve_order(
            relationship.get("qoe_parameter")
        )
        relationship["qos_parameter_definition"] = coerce_pairs_to_exact_params(
            relationship["qos_parameter"],
            relationship.get("qos_parameter_definition"),
        )
        relationship["qoe_parameter_definition"] = coerce_pairs_to_exact_params(
            relationship["qoe_parameter"],
            relationship.get("qoe_parameter_definition"),
        )

        relationship["source"] = augment_source_with_paper_id(
            relationship.get("source"),
            paper_id,
        )
        relationship = attach_year_venue(
            relationship,
            paper_id,
            paper_meta_map,
        )
        relationship = reorder_relationship_keys(relationship)
        cleaned.append(relationship)

    cleaned = dedup_relationships(cleaned)
    cleaned = add_uuid(cleaned)
    cleaned = [reorder_relationship_keys(item) for item in cleaned]

    return cleaned


def extract_relationships_from_md(
    *,
    md_path: str,
    prompt_file: str,
    model: str = "gpt-5.2",
    reasoning_effort: str = "high",
    max_images: int = 24,
    compact: bool = True,
    api_key: Optional[str] = None,
    client: Optional[OpenAI] = None,
    instructions: Optional[str] = None,
    paper_csv: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper that reads files, creates an OpenAI client if needed,
    and returns extracted relationships.
    """
    md_path = os.path.abspath(md_path)
    md_dir = os.path.dirname(md_path)

    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    prompt = read_text_file(prompt_file).strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {prompt_file}")

    paper_md = read_text_file(md_path)
    detected_paper_id = paper_id_from_md_path(md_path)

    paper_meta_map = None
    if paper_csv:
        paper_meta_map = load_paper_meta_map(paper_csv)

    if client is None:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Please set it in the environment or pass api_key."
            )
        client = OpenAI(api_key=api_key)

    return extract_relationships(
        client=client,
        prompt=prompt,
        paper_md=paper_md,
        md_dir=md_dir,
        model=model,
        reasoning_effort=reasoning_effort,
        max_images=max_images,
        compact=compact,
        instructions=instructions,
        paper_id=detected_paper_id,
        paper_meta_map=paper_meta_map,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Extract QoS-QoE relationships from a Markdown paper."
    )
    parser.add_argument("--md", required=True, help="Path to the Markdown paper")
    parser.add_argument("--prompt_file", required=True, help="Path to the extraction prompt file")
    parser.add_argument("--model", default="gpt-5.2", help="OpenAI model name")
    parser.add_argument("--out", default="relationships_core.json", help="Output JSON path")
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
    parser.add_argument(
        "--paper_csv",
        default=None,
        help="Optional CSV file with paper metadata (Index, Year, Venue)",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    relationships = extract_relationships_from_md(
        md_path=args.md,
        prompt_file=args.prompt_file,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_images=args.max_images,
        compact=not args.no_compact,
        paper_csv=args.paper_csv,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(relationships, f, indent=2, ensure_ascii=False)

    print(f"[OK] Extracted {len(relationships)} relationship(s) -> {args.out}")


if __name__ == "__main__":
    main()