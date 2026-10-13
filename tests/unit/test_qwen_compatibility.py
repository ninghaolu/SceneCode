"""Ensure compatibility preserves forced observation and subsequent reasoning."""
import json

import httpx
import pytest

from scenecode.utils.model_audit import _request


@pytest.mark.parametrize("api", ["responses", "chat/completions"])
@pytest.mark.parametrize("choice", ["required", {"type": "function", "name": "observe_scene"}])
def test_forced_tool_keeps_selection_without_thinking_and_next_call_restores_it(monkeypatch, api, choice):
    monkeypatch.setenv("SCENECODE_QWEN_REASONING", "low")
    monkeypatch.setenv("SCENECODE_REQUIRED_MODEL", "qwen3.7-plus")
    monkeypatch.delenv("SCENECODE_AUDIT_DIR", raising=False)
    request = httpx.Request("POST", "https://example.test/" + api, json={
        "model": "qwen3.7-plus", "tool_choice": choice, "thinking_budget": 2048})
    _request(request)
    body = json.loads(request.content)
    assert body["tool_choice"] == choice
    if api == "responses":
        assert body["reasoning"]["effort"] == "none"
    else:
        assert body["enable_thinking"] is False
        assert "thinking_budget" not in body
    assert request.extensions["scenecode_reasoning_override"]

    following = httpx.Request("POST", "https://example.test/" + api, json={
        "model": "qwen3.7-plus", "tool_choice": "auto"})
    _request(following)
    body = json.loads(following.content)
    assert body["tool_choice"] == "auto"
    if api == "responses":
        assert body["reasoning"]["effort"] == "low"
    else:
        assert body["enable_thinking"] is True
