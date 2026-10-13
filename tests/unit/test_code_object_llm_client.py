"""Request compatibility for the Code_Object OpenAI client."""

import unittest

from types import SimpleNamespace
from unittest.mock import AsyncMock

from scenecode.code_object.agent_framework.utils.llm_client import OpenAIClient


class TestCodeObjectOpenAIClient(unittest.IsolatedAsyncioTestCase):
    def make_client(self, model):
        client = OpenAIClient(api_key="test-key", model=model, max_tokens=4096)
        create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ok", tool_calls=None)
                    )
                ]
            )
        )
        client._client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        ))
        return client, create

    async def test_gpt5_image_and_tool_requests_use_compatible_parameters(self):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Build this object"},
            {"type": "image_url", "image_url": {"url": "https://example.com/chair.png"}},
        ]}]
        tools = [{"type": "function", "function": {"name": "build_object"}}]
        for model in ("gpt-5.4-mini", "gpt-5.4", "gpt-5-mini"):
            with self.subTest(model=model):
                client, create = self.make_client(model)
                result = await client.chat(messages, tools=tools, tool_choice="auto")
                self.assertEqual(result, "ok")
                create.assert_awaited_once_with(
                    model=model, messages=messages, max_completion_tokens=4096,
                    tools=tools, tool_choice="auto",
                )

    async def test_model_and_token_budget_overrides(self):
        client, create = self.make_client("gpt-4o")
        messages = [{"role": "user", "content": "hello"}]
        await client.chat(messages, model="gpt-5.4-mini", max_tokens=1024)
        self.assertEqual(create.call_args.kwargs["max_completion_tokens"], 1024)
        self.assertNotIn("temperature", create.call_args.kwargs)
        await client.chat(
            messages, model="gpt-5.4-mini", max_tokens=1024,
            max_completion_tokens=2048,
        )
        self.assertEqual(create.call_args.kwargs["max_completion_tokens"], 2048)

    async def test_other_models_keep_existing_parameters(self):
        messages = [{"role": "user", "content": "hello"}]
        for model in ("gpt-4o", "gemini-3.1-pro-preview"):
            with self.subTest(model=model):
                client, create = self.make_client(model)
                await client.chat(messages, temperature=0.2, max_tokens=512)
                create.assert_awaited_once_with(
                    model=model, messages=messages, temperature=0.2, max_tokens=512,
                )


if __name__ == "__main__":
    unittest.main()
