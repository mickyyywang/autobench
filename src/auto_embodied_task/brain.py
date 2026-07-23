from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
import os
import sys
import time
from typing import Any, Callable, Protocol
from urllib.parse import quote

from openai import APITimeoutError, AuthenticationError, OpenAI, RateLimitError
import requests


@dataclass(frozen=True)
class BrainPolicyConfig:
    provider: str = "qwen"
    model: str = "qwen-vl-plus"
    api_key: str | None = None
    api_key_env: str | None = None
    api_base_url: str | None = None
    timeout_seconds: int = 120
    temperature: float = 0.0
    max_attempts: int = 1
    retry_backoff_seconds: float = 5.0
    retry_max_seconds: float = 60.0
    api_style: str = "chat_completions"
    max_output_tokens: int = 2048
    json_response_format: bool = True
    google_thinking_level: str | None = "low"


@dataclass(frozen=True)
class BrainRequest:
    messages: list[dict[str, Any]]
    summary: dict[str, Any] = field(default_factory=dict)


class ObservationAdapter(Protocol):
    def build_request(self, **kwargs: Any) -> BrainRequest:
        ...

    def parse_response(self, text: str) -> Any:
        ...


class BrainPolicy:
    def __init__(
        self,
        config: BrainPolicyConfig,
        *,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        providers = {
            "openai",
            "qwen",
            "compatible",
            "mr_openai",
            "mr_anthropic",
            "mr_google",
        }
        if config.provider not in providers:
            raise ValueError(f"Unknown provider {config.provider!r}; use one of {sorted(providers)}")
        api_styles = {
            "chat_completions",
            "responses",
            "anthropic_messages",
            "gemini_generate_content",
        }
        if config.api_style not in api_styles:
            raise ValueError(f"Unknown api_style {config.api_style!r}; use one of {sorted(api_styles)}")
        required_style = {
            "mr_anthropic": "anthropic_messages",
            "mr_google": "gemini_generate_content",
        }.get(config.provider)
        if required_style is not None and config.api_style != required_style:
            raise ValueError(f"provider {config.provider!r} requires api_style={required_style!r}")
        if config.provider not in {"mr_anthropic", "mr_google"} and config.api_style not in {
            "chat_completions",
            "responses",
        }:
            raise ValueError(
                f"provider {config.provider!r} only supports chat_completions or responses"
            )
        if config.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if config.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if config.google_thinking_level not in {None, "minimal", "low", "medium", "high"}:
            raise ValueError(
                "google_thinking_level must be minimal, low, medium, high, or None"
            )
        self.config = config
        self.last_response_metadata: dict[str, Any] | None = None
        env_name = config.api_key_env or (
            "DASHSCOPE_API_KEY"
            if config.provider == "qwen"
            else "MR_API_KEY"
            if config.provider.startswith("mr_")
            else "OPENAI_API_KEY"
        )
        api_key = config.api_key or os.environ.get(env_name)
        if not api_key:
            raise RuntimeError(f"Missing API key. Set {env_name} or pass api_key.")
        base_url = config.api_base_url
        if base_url is None and config.provider == "qwen":
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        elif base_url is None and config.provider == "compatible":
            base_url = os.environ.get("OPENAI_BASE_URL")
        elif base_url is None and config.provider == "mr_openai":
            base_url = "https://routify.alibaba-inc.com/protocol/openai/v1"
        elif base_url is None and config.provider == "mr_anthropic":
            base_url = "https://routify.alibaba-inc.com/protocol/anthropic/v1/messages"
        elif base_url is None and config.provider == "mr_google":
            base_url = "https://routify.alibaba-inc.com/protocol/vertex/v1beta"
        self.api_key = api_key
        self.base_url = base_url
        if config.provider in {"mr_anthropic", "mr_google"}:
            self.client = client_factory() if client_factory is not None else requests.Session()
        else:
            self.client = (client_factory or OpenAI)(
                api_key=api_key,
                base_url=base_url,
                timeout=config.timeout_seconds,
            )

    def complete(self, request: BrainRequest) -> str:
        self.last_response_metadata = None
        if self.config.api_style == "anthropic_messages":
            return self._complete_anthropic(request)
        if self.config.api_style == "gemini_generate_content":
            return self._complete_google(request)
        if self.config.api_style == "responses":
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "input": _responses_input(request.messages),
                "max_output_tokens": self.config.max_output_tokens,
            }
        else:
            kwargs = {
                "model": self.config.model,
                "messages": request.messages,
                "temperature": self.config.temperature,
            }
            if self.config.json_response_format:
                kwargs["response_format"] = {"type": "json_object"}
            if self.config.provider == "qwen":
                kwargs["extra_body"] = {"enable_thinking": False}
        completion = None
        response_attempts: list[dict[str, Any]] = []
        responses_output_token_cap = (
            self.config.max_output_tokens * 2
            if self.config.api_style == "responses"
            else self.config.max_output_tokens
        )
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                if self.config.api_style == "responses":
                    completion = self.client.responses.create(**kwargs)
                else:
                    completion = self.client.chat.completions.create(**kwargs)
            except RateLimitError as exc:
                if attempt >= self.config.max_attempts:
                    raise RuntimeError(f"{self.config.provider} brain API request failed: {exc}") from exc
                delay = min(
                    self.config.retry_backoff_seconds * (2 ** (attempt - 1)),
                    self.config.retry_max_seconds,
                )
                print(
                    f"{self.config.provider} brain API throttled; retrying "
                    f"attempt {attempt + 1}/{self.config.max_attempts} in {delay:g}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
                completion = None
                continue
            except (TimeoutError, APITimeoutError) as exc:
                if attempt >= self.config.max_attempts:
                    raise RuntimeError(
                        f"{self.config.provider} brain API request timed out after "
                        f"{self.config.timeout_seconds} seconds on attempt {attempt}/"
                        f"{self.config.max_attempts}."
                    ) from exc
                delay = min(
                    self.config.retry_backoff_seconds * (2 ** (attempt - 1)),
                    self.config.retry_max_seconds,
                )
                print(
                    f"{self.config.provider} brain API request timed out; retrying "
                    f"attempt {attempt + 1}/{self.config.max_attempts} in {delay:g}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
                completion = None
                continue
            except AuthenticationError as exc:
                transient_ak_lookup_failure = (
                    self.config.provider == "mr_openai"
                    and "AK查询失败" in str(exc)
                )
                if not transient_ak_lookup_failure or attempt >= self.config.max_attempts:
                    raise RuntimeError(
                        f"{self.config.provider} brain API request failed: {exc}"
                    ) from exc
                delay = min(
                    self.config.retry_backoff_seconds * (2 ** (attempt - 1)),
                    self.config.retry_max_seconds,
                )
                print(
                    f"{self.config.provider} gateway AK lookup failed; retrying "
                    f"attempt {attempt + 1}/{self.config.max_attempts} in {delay:g}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
                completion = None
                continue
            except Exception as exc:
                raise RuntimeError(f"{self.config.provider} brain API request failed: {exc}") from exc
            if self.config.api_style != "responses":
                break

            content = _responses_output_text(completion)
            attempt_metadata = _responses_attempt_metadata(
                completion,
                attempt=attempt,
                requested_max_output_tokens=int(kwargs["max_output_tokens"]),
                content_length=len(content),
            )
            response_attempts.append(attempt_metadata)
            self.last_response_metadata = {
                "provider": self.config.provider,
                "model": self.config.model,
                "attempts": response_attempts,
            }
            if content:
                return content
            if attempt >= self.config.max_attempts:
                diagnostic = json.dumps(
                    response_attempts[-1],
                    ensure_ascii=False,
                    sort_keys=True,
                )
                raise RuntimeError(
                    f"{self.config.provider} Responses API returned empty output text "
                    f"after {attempt} attempt(s); last response metadata: {diagnostic}"
                )
            incomplete_details = attempt_metadata.get("incomplete_details")
            incomplete_reason = (
                incomplete_details.get("reason")
                if isinstance(incomplete_details, dict)
                else None
            )
            if incomplete_reason == "max_output_tokens":
                current_max_output_tokens = int(kwargs["max_output_tokens"])
                kwargs["max_output_tokens"] = min(
                    current_max_output_tokens * 2,
                    responses_output_token_cap,
                )
            delay = min(
                self.config.retry_backoff_seconds * (2 ** (attempt - 1)),
                self.config.retry_max_seconds,
            )
            print(
                f"{self.config.provider} Responses API returned empty output text; "
                f"retrying attempt {attempt + 1}/{self.config.max_attempts} in {delay:g}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
            completion = None
        if completion is None:
            raise RuntimeError(f"{self.config.provider} brain API request did not complete")
        if not completion.choices:
            raise RuntimeError(f"{self.config.provider} brain API returned no choices")
        content = completion.choices[0].message.content
        if not content:
            raise RuntimeError(f"{self.config.provider} brain API returned empty content")
        return content

    def _complete_anthropic(self, request: BrainRequest) -> str:
        system, messages = _anthropic_messages(request.messages)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_output_tokens,
            "stream": False,
            "messages": messages,
        }
        if system:
            payload["system"] = system
        response = self._post_json(
            str(self.base_url),
            payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        texts = [
            str(item.get("text") or "")
            for item in response.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        content = "\n".join(text for text in texts if text)
        if not content:
            raise RuntimeError("MR Anthropic API returned empty text content")
        return content

    def _complete_google(self, request: BrainRequest) -> str:
        system, contents = _google_contents(request.messages)
        endpoint = _google_endpoint(str(self.base_url), self.config.model)
        max_output_tokens = self.config.max_output_tokens
        attempts: list[dict[str, Any]] = []
        final_content = ""
        final_response: dict[str, Any] = {}
        for semantic_attempt in range(1, 3):
            generation_config: dict[str, Any] = {
                "temperature": self.config.temperature,
                "maxOutputTokens": max_output_tokens,
                "responseMimeType": "application/json",
                "responseJsonSchema": _google_response_json_schema(),
            }
            if (
                self.config.google_thinking_level is not None
                and self.config.model.lower().startswith("gemini-3")
            ):
                generation_config["thinkingConfig"] = {
                    "thinkingLevel": self.config.google_thinking_level
                }
            payload: dict[str, Any] = {
                "contents": contents,
                "generationConfig": generation_config,
            }
            if system:
                payload["systemInstruction"] = {"parts": [{"text": system}]}
            response = self._post_json(
                endpoint,
                payload,
                headers={"x-goog-api-key": self.api_key.strip()},
            )
            final_response = response
            candidates = [
                candidate
                for candidate in response.get("candidates", [])
                if isinstance(candidate, dict)
            ]
            finish_reasons = [
                str(candidate.get("finishReason") or "")
                for candidate in candidates
                if candidate.get("finishReason") is not None
            ]
            finish_messages = [
                str(candidate.get("finishMessage") or "")
                for candidate in candidates
                if candidate.get("finishMessage") is not None
            ]
            texts: list[str] = []
            for candidate in candidates:
                content = candidate.get("content")
                if not isinstance(content, dict):
                    continue
                for part in content.get("parts", []):
                    if (
                        isinstance(part, dict)
                        and not part.get("thought")
                        and isinstance(part.get("text"), str)
                    ):
                        texts.append(part["text"])
            final_content = "\n".join(text for text in texts if text)
            attempts.append(
                {
                    "attempt": semantic_attempt,
                    "max_output_tokens": max_output_tokens,
                    "finish_reasons": finish_reasons,
                    "finish_messages": finish_messages,
                    "usage_metadata": response.get("usageMetadata"),
                    "content_length": len(final_content),
                }
            )
            self.last_response_metadata = {
                "provider": self.config.provider,
                "model": self.config.model,
                "attempts": attempts,
            }
            hit_token_limit = any(
                reason.strip().upper() == "MAX_TOKENS" for reason in finish_reasons
            )
            retry_tokens = min(max_output_tokens * 2, 8192)
            if semantic_attempt == 1 and hit_token_limit and retry_tokens > max_output_tokens:
                max_output_tokens = retry_tokens
                continue
            break
        if not final_content:
            feedback = final_response.get("promptFeedback")
            detail = f": {feedback}" if feedback else ""
            raise RuntimeError(f"MR Google API returned empty text content{detail}")
        return final_content

    def _post_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        request_headers = {"Content-Type": "application/json", **headers}
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                response = self.client.post(
                    endpoint,
                    json=payload,
                    headers=request_headers,
                    timeout=self.config.timeout_seconds,
                )
            except requests.Timeout as exc:
                raise RuntimeError(
                    f"{self.config.provider} brain API request timed out after "
                    f"{self.config.timeout_seconds} seconds."
                ) from exc
            except requests.RequestException as exc:
                raise RuntimeError(f"{self.config.provider} brain API request failed: {exc}") from exc

            status_code = int(getattr(response, "status_code", 0) or 0)
            retryable = status_code == 429 or status_code >= 500
            if retryable and attempt < self.config.max_attempts:
                delay = min(
                    self.config.retry_backoff_seconds * (2 ** (attempt - 1)),
                    self.config.retry_max_seconds,
                )
                print(
                    f"{self.config.provider} brain API returned HTTP {status_code}; retrying "
                    f"attempt {attempt + 1}/{self.config.max_attempts} in {delay:g}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
                continue
            if status_code < 200 or status_code >= 300:
                detail = str(getattr(response, "text", ""))[:2000]
                raise RuntimeError(
                    f"{self.config.provider} brain API request failed with HTTP "
                    f"{status_code}: {detail}"
                )
            try:
                parsed = response.json()
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{self.config.provider} brain API returned invalid JSON"
                ) from exc
            if not isinstance(parsed, dict):
                raise RuntimeError(f"{self.config.provider} brain API returned non-object JSON")
            return parsed
        raise RuntimeError(f"{self.config.provider} brain API request did not complete")


def _responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        raw_content = message.get("content")
        if isinstance(raw_content, str):
            content = [{"type": "input_text", "text": raw_content}]
        elif isinstance(raw_content, list):
            content = []
            for item in raw_content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    content.append({"type": "input_text", "text": str(item.get("text") or "")})
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url")
                    if isinstance(image_url, dict):
                        image_url = image_url.get("url")
                    if image_url:
                        content.append({"type": "input_image", "image_url": str(image_url)})
        else:
            content = []
        converted.append({"role": str(message.get("role") or "user"), "content": content})
    return converted


def _anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        raw_content = message.get("content")
        if role == "system":
            system_text = _content_text(raw_content)
            if system_text:
                system_parts.append(system_text)
            continue
        content: list[dict[str, Any]] = []
        for item in _content_items(raw_content):
            if item.get("type") == "text":
                content.append({"type": "text", "text": str(item.get("text") or "")})
            elif item.get("type") == "image_url":
                image_url = _image_url(item)
                if not image_url:
                    continue
                parsed = _parse_data_url(image_url)
                if parsed is None:
                    source = {"type": "url", "url": image_url}
                else:
                    media_type, data = parsed
                    source = {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    }
                content.append({"type": "image", "source": source})
        converted.append(
            {
                "role": "assistant" if role == "assistant" else "user",
                "content": content,
            }
        )
    return "\n".join(system_parts) or None, converted


def _google_contents(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        raw_content = message.get("content")
        if role == "system":
            system_text = _content_text(raw_content)
            if system_text:
                system_parts.append(system_text)
            continue
        parts: list[dict[str, Any]] = []
        for item in _content_items(raw_content):
            if item.get("type") == "text":
                parts.append({"text": str(item.get("text") or "")})
            elif item.get("type") == "image_url":
                image_url = _image_url(item)
                if not image_url:
                    continue
                parsed = _parse_data_url(image_url)
                if parsed is None:
                    parts.append({"fileData": {"fileUri": image_url}})
                else:
                    media_type, data = parsed
                    parts.append({"inlineData": {"mimeType": media_type, "data": data}})
        converted.append(
            {
                "role": "model" if role == "assistant" else "user",
                "parts": parts,
            }
        )
    return "\n".join(system_parts) or None, converted


def _content_items(raw_content: Any) -> list[dict[str, Any]]:
    if isinstance(raw_content, str):
        return [{"type": "text", "text": raw_content}]
    if isinstance(raw_content, list):
        return [item for item in raw_content if isinstance(item, dict)]
    return []


def _content_text(raw_content: Any) -> str:
    return "\n".join(
        str(item.get("text") or "")
        for item in _content_items(raw_content)
        if item.get("type") == "text" and item.get("text")
    )


def _image_url(item: dict[str, Any]) -> str | None:
    image_url = item.get("image_url")
    if isinstance(image_url, dict):
        image_url = image_url.get("url")
    return str(image_url) if image_url else None


def _parse_data_url(url: str) -> tuple[str, str] | None:
    if not url.startswith("data:") or "," not in url:
        return None
    header, data = url.split(",", 1)
    if ";base64" not in header:
        return None
    media_type = header[5:].split(";", 1)[0] or "application/octet-stream"
    try:
        base64.b64decode(data, validate=True)
    except ValueError as exc:
        raise ValueError("invalid base64 image data URL") from exc
    return media_type, data


def _google_endpoint(base_url: str, model: str) -> str:
    base = base_url.rstrip("/")
    encoded_model = quote(model, safe="")
    if "{model}" in base:
        return base.replace("{model}", encoded_model)
    if base.endswith(":generateContent"):
        return base
    return f"{base}/models/{encoded_model}:generateContent"


def _google_response_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reason": {"type": "string"},
            "recovery": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "required": {"type": "boolean"},
                    "failed_action": {"type": ["string", "null"]},
                    "failed_node_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["required", "failed_action", "failed_node_ids"],
            },
            "action": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "object": {"type": ["string", "null"]},
                    "target": {"type": ["string", "null"]},
                    "node_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["name", "object", "target", "node_ids"],
            },
        },
        "required": ["reason", "recovery", "action"],
    }


def _responses_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text:
        return output_text
    for output in getattr(response, "output", None) or []:
        for item in getattr(output, "content", None) or []:
            text = getattr(item, "text", None)
            if isinstance(text, str) and text:
                return text
    return ""


def _responses_attempt_metadata(
    response: Any,
    *,
    attempt: int,
    requested_max_output_tokens: int,
    content_length: int,
) -> dict[str, Any]:
    outputs = list(getattr(response, "output", None) or [])
    output_types: list[str] = []
    content_types: list[str] = []
    for output in outputs:
        output_type = getattr(output, "type", None)
        if output_type is not None:
            output_types.append(str(output_type))
        for item in getattr(output, "content", None) or []:
            content_type = getattr(item, "type", None)
            if content_type is not None:
                content_types.append(str(content_type))
    return {
        "attempt": attempt,
        "requested_max_output_tokens": requested_max_output_tokens,
        "response_id": getattr(response, "id", None),
        "status": getattr(response, "status", None),
        "incomplete_details": _response_metadata_value(
            getattr(response, "incomplete_details", None)
        ),
        "error": _response_metadata_value(getattr(response, "error", None)),
        "usage": _response_metadata_value(getattr(response, "usage", None)),
        "output_count": len(outputs),
        "output_types": output_types,
        "content_types": content_types,
        "content_length": content_length,
    }


def _response_metadata_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _response_metadata_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_response_metadata_value(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _response_metadata_value(model_dump(mode="json"))
    if hasattr(value, "__dict__"):
        return _response_metadata_value(vars(value))
    return str(value)


class BrainHarness:
    def __init__(self, policy: BrainPolicy, adapter: ObservationAdapter) -> None:
        self.policy = policy
        self.adapter = adapter

    def build_request(self, **kwargs: Any) -> BrainRequest:
        return self.adapter.build_request(**kwargs)

    def decide(self, **kwargs: Any) -> Any:
        request = self.build_request(**kwargs)
        return self.decide_request(request)

    def decide_request(self, request: BrainRequest) -> Any:
        return self.adapter.parse_response(self.policy.complete(request))

    @property
    def last_response_metadata(self) -> dict[str, Any] | None:
        metadata = getattr(self.policy, "last_response_metadata", None)
        return metadata if isinstance(metadata, dict) else None
