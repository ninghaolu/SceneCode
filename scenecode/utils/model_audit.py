"""Opt-in request auditing and model enforcement for reproducible pilots.

Only JSON bodies are recorded; authentication headers are never serialized.
Image payloads are stored once by SHA256. Files are unique across workers/retries.
"""
import contextvars
import hashlib
import json
import os
import re
import time
import traceback
import uuid
from pathlib import Path

import httpx

audit_context = contextvars.ContextVar("model_audit_context", default={})


def _externalize(value, root):
    if isinstance(value, str) and value.startswith("data:image/"):
        digest = hashlib.sha256(value.encode()).hexdigest()
        path = root / "images" / (digest + ".data-url")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(value)
        return {"image_sha256": digest, "path": str(path)}
    if isinstance(value, dict):
        return {k: _externalize(v, root) for k, v in value.items()
                if k not in {"reasoning_content", "api_key", "authorization"}}
    if isinstance(value, list):
        return [_externalize(v, root) for v in value]
    return value


def _request(request):
    if request.method != "POST":
        return
    try:
        body = json.loads(request.content)
    except (ValueError, UnicodeDecodeError):
        return
    model = body.get("model", "")
    expected = os.getenv("SCENECODE_REQUIRED_MODEL")
    if expected and model != expected:
        raise RuntimeError(f"Unintended model requested: {model!r}; expected {expected!r}")
    if model.startswith("qwen"):
        body.pop("service_tier", None)
        # DashScope rejects forced tool selection while thinking is enabled.
        # Preserve the required tool (critics must observe the scene first),
        # disabling thinking for that request only. The next auto-selection
        # request uses the configured reasoning level again.
        choice = body.get("tool_choice")
        forced_tool = choice == "required" or isinstance(choice, dict)
        effort = os.getenv("SCENECODE_QWEN_REASONING", "none")
        if forced_tool and effort != "none":
            request.extensions["scenecode_reasoning_override"] = "forced_tool_choice_requires_non_thinking"
            effort = "none"
        if request.url.path.endswith("/responses"):
            body["reasoning"] = {"effort": effort}
            body.setdefault("max_output_tokens", int(os.getenv("SCENECODE_MAX_TOKENS", "16384")))
            # DashScope accepts text.format but may not constrain decoding.
            # Supply the schema in the prompt too; the Agents SDK still validates
            # the returned JSON against its Pydantic output type.
            fmt = body.get("text", {}).get("format", {})
            if fmt.get("type") == "json_schema":
                body["instructions"] = (body.get("instructions") or "") + (
                    "\nReturn ONLY a JSON object matching this exact JSON Schema. "
                    "No markdown fences, no extra keys, and include every required field:\n"
                    + json.dumps(fmt["schema"]))
        else:
            body["enable_thinking"] = effort != "none"
            if body["enable_thinking"]:
                body["thinking_budget"] = int(os.getenv("SCENECODE_THINKING_BUDGET", "2048"))
            else:
                body.pop("thinking_budget", None)
            body["max_tokens"] = min(body.get("max_tokens", 16384), int(os.getenv("SCENECODE_MAX_TOKENS", "16384")))
        payload = json.dumps(body).encode()
        request._content = payload
        request.stream = httpx.ByteStream(payload)
        request.headers["content-length"] = str(len(payload))
    request.extensions["scenecode_started"] = time.monotonic()
    request.extensions["scenecode_model"] = model
    root = os.getenv("SCENECODE_AUDIT_DIR")
    if root:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{time.time_ns()}_{os.getpid()}_{uuid.uuid4().hex}.json"
        frames = [f for f in traceback.extract_stack()
                  if "/scenecode/" in f.filename and not f.filename.endswith("model_audit.py")]
        component = "outer_agent"
        for frame in frames:
            filename = Path(frame.filename).name
            if filename == "planner_checker.py":
                component = "plan_check"
            elif filename == "planner.py":
                component = "plan_revision" if "revis" in frame.name else "object_plan"
            elif filename == "part_constructor.py":
                component = "code_repair" if "fix" in frame.name else "part_code"
            elif filename == "urdf_pipeline_v1.py":
                component = "urdf_generation"
            elif filename in ("mesh_physics_analyzer.py", "articulated_physics_analyzer.py"):
                component = "physics_analysis"
            elif filename == "router.py":
                component = "asset_validation" if "validat" in frame.name else "asset_routing"
            elif filename == "turn_trimming_session.py":
                component = "memory_summary"
        record = {"requested_model": model, "api": request.url.path,
                  "component": component,
                  "reasoning_override": request.extensions.get("scenecode_reasoning_override"),
                  "context": audit_context.get(), "started_unix": time.time(),
                  "call_stack": [f"{f.filename}:{f.lineno}:{f.name}" for f in frames],
                  "request": _externalize(body, root)}
        path.write_text(json.dumps(record, indent=2))
        request.extensions["scenecode_audit_path"] = path


def _response(response):
    request = response.request
    requested = request.extensions.get("scenecode_model")
    if requested is None:
        return
    try:
        body = response.json()
    except ValueError:
        body = {"unparsed_response": response.text[:2000]}
    path = request.extensions.get("scenecode_audit_path")
    if path:
        record = json.loads(path.read_text())
        record.update(status=response.status_code,
                      api_seconds=time.monotonic() - request.extensions["scenecode_started"],
                      returned_model=body.get("model"),
                      response=_externalize(body, path.parent), billing=None)
        path.write_text(json.dumps(record, indent=2))
    if os.getenv("SCENECODE_REQUIRED_MODEL") and response.is_success and body.get("model") != requested:
        raise RuntimeError(f"Unexpected returned model: {body.get('model')!r}; requested {requested!r}")
    # Some Qwen Responses replies ignore the requested JSON-only format. Remove
    # only a complete Markdown wrapper around syntactically valid JSON. Keep the
    # original response above for audit; SDK schema validation still runs normally.
    if requested.startswith("qwen") and response.is_success:
        request_body = json.loads(request.content)
        if request_body.get("text", {}).get("format", {}).get("type") == "json_schema":
            changed = False
            for item in body.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") != "output_text":
                        continue
                    match = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n\s*```\s*", content.get("text", ""), re.DOTALL)
                    if match:
                        try:
                            json.loads(match.group(1))
                        except ValueError:
                            continue
                        content["text"] = match.group(1)
                        changed = True
            if changed:
                payload = json.dumps(body).encode()
                response._content = payload
                response.headers["content-length"] = str(len(payload))
                if path:
                    record["normalization"] = "removed_markdown_json_fence"
                    path.write_text(json.dumps(record, indent=2))


def audited_http_client(*, asynchronous=False):
    """Return SDK kwargs only when a pilot explicitly enables auditing/pinning."""
    if not (os.getenv("SCENECODE_AUDIT_DIR") or os.getenv("SCENECODE_REQUIRED_MODEL")):
        return {}
    if asynchronous:
        async def before(request):
            _request(request)

        async def after(response):
            await response.aread()
            _response(response)
        client = httpx.AsyncClient(event_hooks={"request": [before], "response": [after]}, timeout=600)
    else:
        def after(response):
            response.read()
            _response(response)
        client = httpx.Client(event_hooks={"request": [_request], "response": [after]}, timeout=600)
    return {"http_client": client}
