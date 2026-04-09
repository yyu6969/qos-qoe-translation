#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
claude.py

Claude-based evaluator for one structured JSON data row.

This version uses a stricter Anthropic tool schema and stronger Python-side
validation so the model output is closer to OpenAI-style structured extraction.

Expected returned format:
{
  "id": str,
  "review_status": "success" | "error",
  "rating": int,              # 1-10
  "confidence": int,          # 1-5
  "concerns": [str, ...],
  "solutions": [str, ...],    # one-to-one aligned with concerns
  "resolved_data_row": {...}
}
"""

from __future__ import annotations

import copy
import json
import os
import time
from typing import Any, Dict, List, Optional

from anthropic import Anthropic

TOOL_NAME = "submit_data_evaluation"

DATA_EVALUATION_USER_PROMPT = """You are a careful QoS-QoE data reviewer for a multimedia systems dataset.

Evaluate the following JSON row for correctness, consistency, completeness, and schema alignment.

Requirements:
- rating must be an integer from 1 to 10
- confidence must be an integer from 1 to 5
- concerns must be a list of concrete issues
- solutions must be a list of fixes aligned one-to-one with concerns
- resolved_data_row must be the full resolved row
- keep corrections conservative and grounded in the given row
- do not invent unsupported facts
- preserve the same top-level fields as the input row
- keep the row id unchanged
- do not add extra fields outside the required tool schema

JSON row:
{row_json}
"""


def load_prompt(prompt_path: str) -> str:
    """Load and return a prompt file as stripped text."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_user_prompt(row: Dict[str, Any]) -> str:
    """Build the row-specific user prompt."""
    row_json = json.dumps(row, ensure_ascii=False, indent=2)
    return DATA_EVALUATION_USER_PROMPT.format(row_json=row_json)


def infer_json_schema_from_value(value: Any) -> Dict[str, Any]:
    """
    Build a strict JSON schema from an example value.

    This is used to constrain resolved_data_row to look like the input row:
    - objects: exact keys only, additionalProperties=False
    - arrays: item schema inferred from example items when possible
    - scalars: exact JSON type
    """
    if value is None:
        return {"type": "null"}

    if isinstance(value, bool):
        return {"type": "boolean"}

    if isinstance(value, int) and not isinstance(value, bool):
        return {"type": "integer"}

    if isinstance(value, float):
        return {"type": "number"}

    if isinstance(value, str):
        return {"type": "string"}

    if isinstance(value, list):
        if not value:
            return {
                "type": "array",
                "items": {},
            }

        item_schemas = [infer_json_schema_from_value(item) for item in value]
        first_schema = json.dumps(item_schemas[0], sort_keys=True)
        homogeneous = all(json.dumps(s, sort_keys=True) == first_schema for s in item_schemas)

        if homogeneous:
            return {
                "type": "array",
                "items": item_schemas[0],
            }

        return {
            "type": "array",
            "items": {
                "anyOf": item_schemas,
            },
        }

    if isinstance(value, dict):
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for key, subvalue in value.items():
            if not isinstance(key, str):
                continue
            properties[key] = infer_json_schema_from_value(subvalue)
            required.append(key)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    return {}


def get_tool_schema(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the Anthropic tool schema for structured evaluation output."""
    resolved_row_schema = infer_json_schema_from_value(row)

    return [
        {
            "name": TOOL_NAME,
            "description": "Return the structured data quality evaluation for one JSON row.",
            "input_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "rating": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Overall data quality rating from 1 to 10.",
                    },
                    "confidence": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "Confidence in the evaluation from 1 to 5.",
                    },
                    "concerns": {
                        "type": "array",
                        "minItems": 0,
                        "items": {
                            "type": "string",
                            "minLength": 1,
                        },
                        "description": "List of concrete data quality issues.",
                    },
                    "solutions": {
                        "type": "array",
                        "minItems": 0,
                        "items": {
                            "type": "string",
                            "minLength": 1,
                        },
                        "description": "List of fixes aligned one-to-one with concerns.",
                    },
                    "resolved_data_row": {
                        **resolved_row_schema,
                        "description": (
                            "Full resolved version of the input row. "
                            "It must preserve the same top-level structure and keep the same id."
                        ),
                    },
                },
                "required": [
                    "rating",
                    "confidence",
                    "concerns",
                    "solutions",
                    "resolved_data_row",
                ],
            },
        }
    ]


def same_json_type(a: Any, b: Any) -> bool:
    """Check whether two values have the same JSON type."""
    def json_type(x: Any) -> str:
        if x is None:
            return "null"
        if isinstance(x, bool):
            return "boolean"
        if isinstance(x, int) and not isinstance(x, bool):
            return "integer"
        if isinstance(x, float):
            return "number"
        if isinstance(x, str):
            return "string"
        if isinstance(x, list):
            return "array"
        if isinstance(x, dict):
            return "object"
        return "unknown"

    return json_type(a) == json_type(b)


def validate_resolved_row_structure(
    original: Any,
    resolved: Any,
    path: str = "resolved_data_row",
) -> None:
    """
    Validate that resolved_data_row preserves structure and JSON types.

    This is intentionally conservative:
    - object keys must match exactly
    - JSON types must match
    - recursion applies to nested values
    """
    if not same_json_type(original, resolved):
        raise ValueError(
            f"{path} has different JSON type than the original row "
            f"(original={type(original).__name__}, resolved={type(resolved).__name__})."
        )

    if isinstance(original, dict):
        original_keys = set(original.keys())
        resolved_keys = set(resolved.keys())

        if original_keys != resolved_keys:
            missing = sorted(original_keys - resolved_keys)
            extra = sorted(resolved_keys - original_keys)
            raise ValueError(
                f"{path} keys do not match original row. Missing={missing}, Extra={extra}"
            )

        for key in original:
            validate_resolved_row_structure(
                original[key],
                resolved[key],
                path=f"{path}.{key}",
            )

    elif isinstance(original, list):
        if not isinstance(resolved, list):
            raise ValueError(f"{path} must be a list.")

        # If original list is empty, there is nothing more to check.
        if not original:
            return

        # Use the first original item as the reference shape for all resolved items.
        reference = original[0]
        for i, item in enumerate(resolved):
            validate_resolved_row_structure(
                reference,
                item,
                path=f"{path}[{i}]",
            )


def validate_output(
    result: Dict[str, Any],
    original_row: Dict[str, Any],
    row_id: str,
) -> Dict[str, Any]:
    """Validate Claude structured output."""
    required_keys = [
        "rating",
        "confidence",
        "concerns",
        "solutions",
        "resolved_data_row",
    ]

    if not isinstance(result, dict):
        raise ValueError("Claude output is not a dict.")

    extra_keys = set(result.keys()) - set(required_keys)
    if extra_keys:
        raise ValueError(f"Claude output contains unexpected keys: {sorted(extra_keys)}")

    for key in required_keys:
        if key not in result:
            raise ValueError(f"Missing required key: {key}")

    if not isinstance(result["rating"], int) or not (1 <= result["rating"] <= 10):
        raise ValueError("rating must be an integer between 1 and 10.")

    if not isinstance(result["confidence"], int) or not (1 <= result["confidence"] <= 5):
        raise ValueError("confidence must be an integer between 1 and 5.")

    if not isinstance(result["concerns"], list) or not all(
        isinstance(x, str) and x.strip() for x in result["concerns"]
    ):
        raise ValueError("concerns must be a list of non-empty strings.")

    if not isinstance(result["solutions"], list) or not all(
        isinstance(x, str) and x.strip() for x in result["solutions"]
    ):
        raise ValueError("solutions must be a list of non-empty strings.")

    if len(result["concerns"]) != len(result["solutions"]):
        raise ValueError("concerns and solutions must have the same length.")

    if not isinstance(result["resolved_data_row"], dict):
        raise ValueError("resolved_data_row must be a dict.")

    validate_resolved_row_structure(
        original=original_row,
        resolved=result["resolved_data_row"],
        path="resolved_data_row",
    )

    resolved_id = result["resolved_data_row"].get("id")
    if resolved_id != row_id:
        raise ValueError(
            f"resolved_data_row.id must equal the original row id. "
            f"Expected {row_id!r}, got {resolved_id!r}."
        )

    return result


def make_fallback_result(
    row: Dict[str, Any],
    row_id: str,
    error_message: str,
) -> Dict[str, Any]:
    """Return a fallback result when Claude fails after all retries."""
    return {
        "id": row_id,
        "review_status": "error",
        "rating": 1,
        "confidence": 1,
        "concerns": [
            f"Claude evaluation failed due to API, parsing, or output validation error: {error_message}"
        ],
        "solutions": [
            "Retry this row manually or inspect the prompt, schema, and row content."
        ],
        "resolved_data_row": {},
    }


def extract_tool_result(response: Any) -> Dict[str, Any]:
    """Extract the expected tool output from a Claude response."""
    for block in response.content:
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == TOOL_NAME
        ):
            if isinstance(block.input, dict):
                return block.input
            raise ValueError("Claude returned a tool call, but the tool input is not a dict.")

    raise ValueError("Claude did not return the expected tool output.")


def evaluate_with_claude(
    row: Dict[str, Any],
    row_id: str,
    system_prompt_path: str,
    model: str = "claude-opus-4-1-20250805",
    temperature: float = 0.0,
    max_tokens: int = 4000,
    max_retries: int = 3,
    sleep_seconds: float = 1.0,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate one row with Claude and return standardized structured output.

    Parameters:
    - row: original input row
    - row_id: row identifier
    - system_prompt_path: path to system prompt file
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")

    if not isinstance(row, dict):
        raise ValueError("row must be a dict.")

    if not isinstance(row_id, str) or not row_id.strip():
        raise ValueError("row_id must be a non-empty string.")

    system_prompt = load_prompt(system_prompt_path)
    user_prompt = build_user_prompt(row)

    client = Anthropic(api_key=api_key)
    tools = get_tool_schema(row)

    last_error: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
                tools=tools,
                tool_choice={
                    "type": "tool",
                    "name": TOOL_NAME,
                },
                temperature=temperature,
                max_tokens=max_tokens,
            )

            tool_result = extract_tool_result(response)
            validated = validate_output(
                result=tool_result,
                original_row=row,
                row_id=row_id,
            )
            validated["review_status"] = "success"
            validated["id"] = row_id
            return validated

        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries:
                time.sleep(sleep_seconds)

    return make_fallback_result(
        row=row,
        row_id=row_id,
        error_message=last_error or "Unknown error",
    )