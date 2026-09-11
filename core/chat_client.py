"""Optional OpenAI-compatible adapter. Credentials never enter generation records."""
import os
from urllib.parse import urlparse

class OpenAIChatClient:
    def __init__(self):
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("Set OPENAI_API_KEY in your environment before generation.")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        parsed = urlparse(base_url)
        if parsed.scheme != "https" or parsed.username or parsed.password or parsed.query:
            raise SystemExit("OPENAI_BASE_URL must use HTTPS and contain no credentials or query parameters.")
        self.client = OpenAI(api_key=key, base_url=base_url, timeout=120, max_retries=2)
        self.omit_temperature = os.environ.get("STRUCTEVAL_OMIT_TEMPERATURE", "0") == "1"

    def chat_completions(self, model, messages, n, temperature):
        # Single-output requests also work with providers that do not support n > 1.
        outputs = []
        for _ in range(n):
            kwargs = {"model": model, "messages": [{"role": m.role, "content": m.content} for m in messages]}
            if not self.omit_temperature:
                kwargs["temperature"] = temperature
            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                # SDK errors can contain endpoint details; never persist raw API errors.
                raise RuntimeError(f"Generation request failed ({type(exc).__name__}); check model, endpoint and parameter support.") from None
            content = response.choices[0].message.content if response.choices else None
            if content is None:
                raise RuntimeError("Provider returned no text completion.")
            outputs.append(content.strip())
        return outputs
