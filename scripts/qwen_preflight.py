"""Probe the actual model capabilities before allocating a generation run."""
import argparse
import asyncio
import base64
import concurrent.futures
import json
import os
from pathlib import Path

from scenecode.utils.model_audit import audited_http_client, _externalize


async def sdk_tool_roundtrip(model):
    """Exercise the actual critic lifecycle: forced observation, then verdict."""
    from agents import Agent, ModelSettings, SQLiteSession, function_tool
    from pydantic import BaseModel
    from scenecode.utils.openai import create_run_config, close_cached_async_openai_clients
    from scenecode.agent_utils.bounded_critic import run_bounded_critic

    calls = []

    @function_tool
    def observe_scene() -> str:
        """Read the measured scene; call this before giving a verdict."""
        calls.append("observe_scene")
        return json.dumps({"category": "table", "measured_gap_m": 0.04, "allowed_gap_m": 0.002})

    class Verdict(BaseModel):
        category: str
        acceptable: bool
        gap_m: float

    agent = Agent(name="preflight_critic", model=model, tools=[observe_scene],
                  instructions="First observe_scene, then return the measured category and gap. "
                               "Accept only when measured_gap_m <= allowed_gap_m. Do not invent measurements.",
                  model_settings=ModelSettings(tool_choice="observe_scene", max_tokens=2048),
                  output_type=Verdict)
    session = SQLiteSession("preflight_critic")
    try:
        # Exercise the production critic's format finalization too. Preserve the
        # existing four-call upper bound: three tool turns plus one final verdict.
        result = await run_bounded_critic(
            agent=agent, instruction="Check this scene.", session=session,
            max_turns=3, run_config=create_run_config())
        verdict = result.final_output
        passed = bool(calls) and verdict.category == "table" and not verdict.acceptable and abs(verdict.gap_m - .04) < 1e-8
        return {"passed": passed, "tool_calls": calls, "verdict": verdict.model_dump()}
    finally:
        session.close()
        await close_cached_async_openai_clients()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = os.environ.get("SCENECODE_QWEN_MODEL", "qwen3.7-plus")
    base = os.environ["OPENAI_API_BASE"].rstrip("/")
    image = "data:image/png;base64," + base64.b64encode(args.image.read_bytes()).decode()
    schema = {"type": "object", "properties": {"category": {"type": "string"}},
              "required": ["category"], "additionalProperties": False}
    tests = {
        "vision": ("chat/completions", {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "What furniture is this? Answer in three words."},
            {"type": "image_url", "image_url": {"url": image}}]}],
            "enable_thinking": False, "max_tokens": 64}),
        "chat_tool": ("chat/completions", {"messages": [{"role": "user", "content": "Call classify with category table."}],
            "tools": [{"type": "function", "function": {"name": "classify", "parameters": schema}}],
            "tool_choice": {"type": "function", "function": {"name": "classify"}},
            "enable_thinking": False, "max_tokens": 128}),
        "summary": ("responses", {"instructions": "Reply concisely.",
            "input": "Summarize: A sofa and table were placed in a living room."}),
        "structured": ("responses", {"input": "Classify a coffee table in JSON.",
            "text": {"format": {"type": "json_schema", "name": "category", "strict": True, "schema": schema}}}),
        "responses_tool": ("responses", {"input": "Call classify with category table.",
            "tools": [{"type": "function", "name": "classify", "parameters": schema, "strict": True}],
            "tool_choice": {"type": "function", "name": "classify"}, "parallel_tool_calls": False}),
    }

    def check(item):
        name, (api, body) = item
        if api == "responses":
            body.update(reasoning={"effort": "none"}, max_output_tokens=512)
        try:
            import httpx
            with audited_http_client().get("http_client", httpx.Client()) as client:
                r = client.post(base + "/" + api, headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
                                json={"model": model, **body}, timeout=120)
            data = r.json()
            passed = r.status_code == 200 and data.get("model") == model
            if passed and name == "vision":
                passed = bool(data["choices"][0]["message"].get("content"))
            if passed and api == "responses":
                passed = data.get("status") == "completed"
            if passed and name == "chat_tool":
                calls = data["choices"][0]["message"].get("tool_calls", [])
                passed = any(c["function"]["name"] == "classify" and
                             json.loads(c["function"]["arguments"]) == {"category": "table"} for c in calls)
            if passed and name == "responses_tool":
                passed = any(x["type"] == "function_call" and x["name"] == "classify" and
                             json.loads(x["arguments"]) == {"category": "table"} for x in data.get("output", []))
            if passed and name == "structured":
                content = "".join(c.get("text", "") for x in data.get("output", []) for c in x.get("content", []))
                verdict = json.loads(content)
                passed = set(verdict) == {"category"} and isinstance(verdict["category"], str)
            print(name, r.status_code, "PASS" if passed else "FAIL", flush=True)
            return name, {"passed": passed, "status": r.status_code, "response": data}
        except Exception as exc:
            return name, {"passed": False, "error": str(exc), "response": locals().get("data")}

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        result = dict(pool.map(check, tests.items()))
    if all(x["passed"] for x in result.values()):
        try:
            result["sdk_tool_roundtrip"] = asyncio.run(sdk_tool_roundtrip(model))
        except Exception as exc:
            result["sdk_tool_roundtrip"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        print("sdk_tool_roundtrip", "PASS" if result["sdk_tool_roundtrip"]["passed"] else "FAIL", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_externalize({"model": model, "api_base": base,
        "reasoning": os.getenv("SCENECODE_QWEN_REASONING", "none"),
        "forced_tool_reasoning": "none", "checks": result}, args.output.parent), indent=2))
    if not all(x["passed"] for x in result.values()):
        raise SystemExit("API preflight failed; inspect " + str(args.output))


if __name__ == "__main__":
    main()
