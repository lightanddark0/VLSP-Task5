"""OpenAI-compatible request formatting and sanitized response handling."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from spartqa.data import answer_choices, validate_answer


def response_format(payload: dict[str, Any]) -> dict[str, Any]:
    task = payload["q_type"]
    choices = answer_choices(payload)
    answer_schema = {
        "type": "array",
        "items": {"type": "integer" if task in {"FR", "CO"} else "string", "enum": choices},
        "minItems": 0 if task == "FB" else 1,
        "maxItems": 1 if task in {"YN", "CO"} else len(choices),
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "spatial_answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"reasoning": {"type": "string"}, "answer": answer_schema},
                "required": ["reasoning", "answer"],
                "additionalProperties": False,
            },
        },
    }


def request_answer(client: Any, example: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": config["prompt"]},
            {"role": "user", "content": json.dumps(example["payload"], ensure_ascii=False)},
        ],
        temperature=config["temperature"],
        max_completion_tokens=config["max_output_tokens"],
        response_format=response_format(example["payload"]),
    )
    record = {
        "key": example["key"],
        "response_id": response.id,
        "response_model": response.model,
        "usage": response.usage.model_dump() if response.usage else {},
    }
    choice = response.choices[0]
    if choice.finish_reason != "stop" or choice.message.refusal:
        return {**record, "error": "RefusedOrIncompleteResponse"}
    try:
        result = json.loads(choice.message.content or "")
        validate_answer(result["answer"], example["payload"])
        if not isinstance(result["reasoning"], str):
            raise ValueError("reasoning must be text")
    except (ValueError, KeyError, TypeError):
        return {**record, "error": "InvalidResponse"}
    return {**record, "answer": result["answer"], "reasoning": result["reasoning"]}


def api_error_details(error: Exception) -> dict[str, Any]:
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        body = body.get("error", body)
    if not isinstance(body, dict):
        body = {}
    details = {"status_code": getattr(error, "status_code", None)}
    for field in ("message", "code", "param", "type"):
        value = body.get(field)
        if isinstance(value, (str, int)):
            text = str(value)
            for name, secret in os.environ.items():
                if secret and (name.endswith("API_KEY") or name.endswith("TOKEN")):
                    text = text.replace(secret, "[REDACTED]")
            text = re.sub(r"\b(?:sk|sess)-[A-Za-z0-9_-]+", "[REDACTED]", text)
            text = re.sub(r"(?i)Bearer\s+[^\s\"']+", "Bearer [REDACTED]", text)
            details[field] = text[:1000]
    return details