import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    model: str
    base_url: str | None
    api_key: str
    home: Path
    project_root: Path
    confirm: bool
    embed_backend: str   # "openai" | "local"
    embed_model: str


def load_config() -> Config:
    home = Path(os.getenv("JAGAN_HOME", Path.home() / ".jagan")).expanduser()
    home.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("JAGAN_API_KEY") or os.getenv("OPENAI_API_KEY") or "ollama"
    embed_backend = os.getenv(
        "JAGAN_EMBED_BACKEND",
        "openai" if api_key != "ollama" else "local",
    )

    return Config(
        model=os.getenv("JAGAN_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("JAGAN_BASE_URL") or None,
        api_key=api_key,
        home=home,
        project_root=Path.cwd(),
        confirm=os.getenv("JAGAN_CONFIRM", "1") != "0",
        embed_backend=embed_backend,
        embed_model=os.getenv(
            "JAGAN_EMBED_MODEL",
            "text-embedding-3-small"
            if embed_backend == "openai"
            else "all-MiniLM-L6-v2",
        ),
    )