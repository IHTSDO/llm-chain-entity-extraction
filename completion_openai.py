import os
import json
import hashlib
import sys
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

MODEL = os.environ.get("OPENAI_MODEL") or "gpt-5-mini"
TEMPERATURE = 0
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
CACHE_PATH = os.path.join(CACHE_DIR, "openai_responses_cache.json")
# Temperature 0.2 is recommended for deterministic and focused output that is
# 'more likely to be correct and efficient'. Instead, 0 is used for consistency.

_CACHE = None
DEBUG_CACHE = os.environ.get("OPENAI_DEBUG_CACHE", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
EMPTY_SENTINEL = "__EMPTY__"

def _normalize_for_cache(value):
    if isinstance(value, dict):
        return {k: _normalize_for_cache(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_normalize_for_cache(item) for item in value]
    return value

def _load_cache():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                # Drop low-value cache entries (empty strings) to reduce false "hits".
                cleaned = {
                    k: v for k, v in loaded.items()
                    if not (isinstance(v, str) and not v.strip())
                }
                _CACHE = cleaned
                if len(cleaned) != len(loaded):
                    _save_cache()
            else:
                _CACHE = {}
    except FileNotFoundError:
        _CACHE = {}
    except json.JSONDecodeError:
        _CACHE = {}
    return _CACHE

def _save_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp_path = f"{CACHE_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_load_cache(), f, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    os.replace(tmp_path, CACHE_PATH)

def _cache_key(kind, model, prompts, kwargs, schema=None):
    payload = {
        "kind": kind,
        "model": model,
        "prompts": _normalize_for_cache(prompts),
        "kwargs": _normalize_for_cache(kwargs),
    }
    if schema is not None:
        payload["schema"] = _normalize_for_cache(schema)
    payload_str = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

def _cache_get(key):
    value = _load_cache().get(key)
    if value == EMPTY_SENTINEL:
        if DEBUG_CACHE:
            print(f"[cache] HIT-EMPTY key={key[:12]}", file=sys.stderr)
        return ""
    if DEBUG_CACHE:
        print(f"[cache] {'HIT' if value is not None else 'MISS'} key={key[:12]}", file=sys.stderr)
    return value

def _cache_set(key, value):
    if isinstance(value, str) and not value.strip():
        value = EMPTY_SENTINEL
    cache = _load_cache()
    cache[key] = value
    if DEBUG_CACHE:
        label = "STORE-EMPTY" if value == EMPTY_SENTINEL else "STORE"
        print(f"[cache] {label} key={key[:12]}", file=sys.stderr)
    _save_cache()

def _responses_create_with_fallback(model, responses_input, responses_kwargs):
    try:
        return client.responses.create(
            model=model,
            input=responses_input,
            **responses_kwargs,
        )
    except Exception:
        # Safety fallback: retry once without reasoning hints.
        if "reasoning" in responses_kwargs:
            retry_kwargs = dict(responses_kwargs)
            retry_kwargs.pop("reasoning", None)
            return client.responses.create(
                model=model,
                input=responses_input,
                **retry_kwargs,
            )
        raise

def _coerce_to_schema(parsed, schema):
    if isinstance(parsed, (dict, list)):
        return parsed
    if not isinstance(schema, dict):
        return None
    if schema.get("type") != "object":
        return None

    required = schema.get("required", [])
    properties = schema.get("properties", {})
    if len(required) != 1:
        return None
    key = required[0]
    prop = properties.get(key, {})

    if prop.get("type") == "integer":
        try:
            value = int(parsed)
        except (TypeError, ValueError):
            return None
        minimum = prop.get("minimum")
        maximum = prop.get("maximum")
        if minimum is not None and value < minimum:
            return None
        if maximum is not None and value > maximum:
            return None
        return {key: value}

    if prop.get("type") == "string":
        return {key: str(parsed)}

    return None

def _to_responses_input(prompts):
    responses_input = []
    for prompt in prompts:
        role = prompt["role"]
        content = prompt["content"]
        if isinstance(content, list):
            responses_input.append({"role": role, "content": content})
            continue

        content_type = "output_text" if role == "assistant" else "input_text"
        responses_input.append(
            {
                "role": role,
                "content": [{"type": content_type, "text": content}],
            }
        )
    return responses_input


def _extract_text_from_responses_api(response):
    # Prefer the SDK helper when available.
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    collected = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "output_text":
                collected.append(getattr(content, "text", ""))
    return "".join(collected).strip()


def create_chat_completion(prompts, model=MODEL, temperature=TEMPERATURE, **kwargs):
    # Keep compatibility with existing calls in the project.
    use_cache = kwargs.pop("use_cache", True)
    max_output_tokens = kwargs.pop("max_output_tokens", None)
    legacy_max_tokens = kwargs.pop("max_tokens", None)
    if max_output_tokens is None:
        max_output_tokens = legacy_max_tokens

    responses_kwargs = dict(kwargs)
    if max_output_tokens is not None:
        # Responses API en modelos recientes (p. ej. gpt-5-mini) requiere >= 16.
        if max_output_tokens < 16:
            max_output_tokens = 16
        responses_kwargs["max_output_tokens"] = max_output_tokens
    # Some newer models (e.g. gpt-5-mini) reject temperature.
    if not str(model).startswith("gpt-5"):
        responses_kwargs["temperature"] = temperature
    else:
        responses_kwargs.setdefault("reasoning", {"effort": "minimal"})

    responses_input = _to_responses_input(prompts)
    key = None
    if use_cache:
        key = _cache_key(
            kind="text",
            model=model,
            prompts=responses_input,
            kwargs=responses_kwargs,
        )
        cached = _cache_get(key)
        if isinstance(cached, str):
            return cached

    response = _responses_create_with_fallback(
        model=model,
        responses_input=responses_input,
        responses_kwargs=responses_kwargs,
    )
    text = _extract_text_from_responses_api(response)
    if use_cache and key is not None:
        _cache_set(key, text)
    return text


def create_chat_completion_json(
    prompts,
    schema,
    schema_name="structured_output",
    model=MODEL,
    temperature=TEMPERATURE,
    **kwargs,
):
    """Return JSON decoded output constrained by a JSON schema when supported."""
    # Keep compatibility with existing calls in the project.
    use_cache = kwargs.pop("use_cache", True)
    max_output_tokens = kwargs.pop("max_output_tokens", None)
    legacy_max_tokens = kwargs.pop("max_tokens", None)
    if max_output_tokens is None:
        max_output_tokens = legacy_max_tokens

    responses_kwargs = dict(kwargs)
    if max_output_tokens is not None:
        if max_output_tokens < 16:
            max_output_tokens = 16
        responses_kwargs["max_output_tokens"] = max_output_tokens
    if not str(model).startswith("gpt-5"):
        responses_kwargs["temperature"] = temperature
    else:
        responses_kwargs.setdefault("reasoning", {"effort": "minimal"})

    responses_input = _to_responses_input(prompts)
    # Prefer native schema-constrained generation.
    responses_kwargs["text"] = {
        "format": {
            "type": "json_schema",
            "name": schema_name,
            "schema": schema,
            "strict": True,
        }
    }

    key = None
    if use_cache:
        key = _cache_key(
            kind="json",
            model=model,
            prompts=responses_input,
            kwargs=responses_kwargs,
            schema=schema,
        )
        cached = _cache_get(key)
        if isinstance(cached, (dict, list)):
            return cached

    try:
        response = _responses_create_with_fallback(
            model=model,
            responses_input=responses_input,
            responses_kwargs=responses_kwargs,
        )
        parsed_raw = json.loads(_extract_text_from_responses_api(response))
        parsed = _coerce_to_schema(parsed_raw, schema) or parsed_raw
        if use_cache and key is not None and isinstance(parsed, (dict, list)):
            _cache_set(key, parsed)
        return parsed
    except Exception:
        # Fallback to plain completion; caller may handle parsing/retries.
        text = create_chat_completion(prompts, model=model, temperature=temperature, **kwargs)
        parsed_raw = json.loads(text)
        parsed = _coerce_to_schema(parsed_raw, schema) or parsed_raw
        if use_cache and key is not None and isinstance(parsed, (dict, list)):
            _cache_set(key, parsed)
        return parsed
