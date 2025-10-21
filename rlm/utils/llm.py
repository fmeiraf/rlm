"""
OpenAI Client wrapper specifically for GPT-5 models.
"""

import os
from typing import Optional

import litellm
from dotenv import load_dotenv
from langfuse import observe
from openai import OpenAI

load_dotenv()


class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        self.model = model
        self.client = OpenAI(api_key=self.api_key)

        # Implement cost tracking logic here.

    def completion(
        self,
        messages: list[dict[str, str]] | str,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, dict):
                messages = [messages]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_tokens,
                **kwargs,
            )
            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")


class LiteLLMClient:
    def __init__(self, model: str = "openai/gpt-5"):
        self.model = model
        self.client = litellm.completion
        self.aclient = litellm.acompletion

    @observe(name="litellm.completion", as_type="generation")
    def completion(
        self,
        messages: list[dict[str, str]] | str,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> str:
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, dict):
                messages = [messages]

            response = self.client(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")

    @observe(name="litellm.completion", as_type="generation")
    async def acompletion(
        self,
        messages: list[dict[str, str]] | str,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> str:
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, dict):
                messages = [messages]

            response = await self.aclient(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")
