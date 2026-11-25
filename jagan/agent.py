import json

from .config import Config
from .llm import LLM
from .memory import Memory
from .tools.registry import Registry

SYSTEM_PROMPT = """You are Jagan, a terse, competent engineering and research assistant.
You operate in the user's project directory and have tools for files, shell, git,
web, arXiv, and semantic search over indexed docs/code.

Rules:
- Be concise. No preamble, no filler.
- Prefer tools over guessing. Read files before editing them.
- For multi-step work, state a 1-2 line plan, then execute.
- You cannot ask the user to confirm; the harness handles confirmations for
  dangerous tools. Just call them.
- Cite paths and line numbers when discussing code.
- For research, use arxiv_search + fetch_url; summarize with citations.
- Persist durable project facts (build commands, architecture, conventions)
  via `remember` so future sessions retain them.

Project root: {root}
"""


class Agent:
    def __init__(self, llm: LLM, reg: Registry, mem: Memory, cfg: Config):
        self.llm, self.reg, self.mem, self.cfg = llm, reg, mem, cfg
        self.messages: list[dict] = []
        self.reset()

    def reset(self):
        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(root=self.cfg.project_root),
            }
        ]

    def _confirm(self, name: str, args: dict) -> bool:
        if not self.cfg.confirm or not self.reg.is_dangerous(name):
            return True
        from rich.console import Console
        from rich.prompt import Confirm

        Console().print(f"[yellow]WARN {name}({json.dumps(args)[:300]})[/yellow]")
        return Confirm.ask("run?", default=False)

    def step(self, user_input, on_tool=lambda n, a: None) -> str:
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(25):
            msg = self.llm.chat(self.messages, tools=self.reg.schemas())
            self.messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                return msg.content or "(no response)"

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                on_tool(name, args)

                if not self._confirm(name, args):
                    result = "user declined"
                else:
                    try:
                        result = self.reg.call(name, args)
                    except Exception as e:
                        result = f"error: {type(e).__name__}: {e}"

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result[:20000],
                    }
                )

        return "(hit tool-iteration limit; refine the request)"