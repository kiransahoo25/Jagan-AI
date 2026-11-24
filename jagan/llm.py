from openai import OpenAI

from .config import Config


class LLM:
    """Provider-agnostic wrapper. Points at any OpenAI-compatible endpoint
    (OpenAI, Ollama, Groq, Together, vLLM, LM Studio)."""

    def __init__(self, cfg: Config):
        self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        self.model = cfg.model

    def chat(self, messages: list[dict], tools: list[dict] | None = None):
        kwargs = dict(model=self.model, messages=messages, temperature=0.2)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return self.client.chat.completions.create(**kwargs).choices[0].message