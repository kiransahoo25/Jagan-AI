import json

from rich.console import Console
from rich.markdown import Markdown

from .agent import Agent
from .config import load_config
from .llm import LLM
from .memory import Memory
from .rag import VectorStore, ingest, make_embedder
from .tools import register_all
from .tools.registry import Registry

HELP = """\
Commands:
  /index <dir>   index a directory for semantic search
  /tools         list available tools
  /reset         clear conversation
  /help          this message
  /exit          quit
"""


def main():
    cfg = load_config()
    console = Console()

    llm = LLM(cfg)
    mem = Memory(cfg.home / "memory.db")
    embed = make_embedder(cfg)
    store = VectorStore(cfg.home / "vectors.db", embed)

    reg = Registry()
    register_all(reg, cfg, mem, store)
    agent = Agent(llm, reg, mem, cfg)

    console.print(
        f"[bold cyan]Jagan[/bold cyan] ready  -  model=[dim]{cfg.model}[/dim]  -  "
        f"root=[dim]{cfg.project_root}[/dim]  -  confirm={'on' if cfg.confirm else 'off'}"
    )
    console.print("[dim]type /help for commands[/dim]")

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not line:
            continue
        if line in ("/exit", "/quit"):
            break
        if line == "/help":
            console.print(HELP)
            continue
        if line == "/reset":
            agent.reset()
            console.print("reset")
            continue
        if line == "/tools":
            for t in reg.tools.values():
                console.print(f"  [cyan]{t.name}[/cyan] - {t.description}")
            continue
        if line.startswith("/index "):
            ingest((cfg.project_root / line[7:]).resolve(), store, console.print)
            continue

        def on_tool(name, args):
            console.print(f"[dim]>> {name}({json.dumps(args)[:140]})[/dim]")

        try:
            out = agent.step(line, on_tool)
            console.print(Markdown(out))
        except Exception as e:
            console.print(f"[red]error: {e}[/red]")