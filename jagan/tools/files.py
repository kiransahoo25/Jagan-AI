from pathlib import Path

from .registry import Registry


def register(reg: Registry, cfg):
    root = cfg.project_root

    def _resolve(p: str) -> Path:
        path = Path(p)
        return (path if path.is_absolute() else root / path).resolve()

    @reg.tool(
        "read_file",
        "Read a file with line numbers.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start": {"type": "integer"},
                "end": {"type": "integer"},
            },
            "required": ["path"],
        },
    )
    def read_file(path, start=None, end=None):
        p = _resolve(path)
        lines = p.read_text(errors="ignore").splitlines()
        s = (start - 1) if start else 0
        e = end if end else len(lines)
        body = "\n".join(
            f"{i + 1:5d}  {l}" for i, l in enumerate(lines[s:e], start=s)
        )
        return body[:50000] or "(empty)"

    @reg.tool(
        "write_file",
        "Create or overwrite a file.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        dangerous=True,
    )
    def write_file(path, content):
        p = _resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {len(content)} bytes to {p}"

    @reg.tool(
        "edit_file",
        "Replace an exact string in a file. `old` must appear exactly once.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old": {"type": "string"},
                "new": {"type": "string"},
            },
            "required": ["path", "old", "new"],
        },
        dangerous=True,
    )
    def edit_file(path, old, new):
        p = _resolve(path)
        text = p.read_text()
        n = text.count(old)
        if n != 1:
            return f"error: `old` appears {n} times (must be exactly 1)"
        p.write_text(text.replace(old, new))
        return f"edited {p}"

    @reg.tool(
        "list_dir",
        "List a directory.",
        {"type": "object", "properties": {"path": {"type": "string"}}},
    )
    def list_dir(path="."):
        p = _resolve(path)
        out = []
        for e in sorted(p.iterdir()):
            out.append(("dir  " if e.is_dir() else "file ") + e.name)
        return "\n".join(out) or "(empty)"

    @reg.tool(
        "glob",
        "Find files by glob pattern, e.g. '**/*.py'.",
        {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    )
    def glob(pattern):
        hits = [str(p.relative_to(root)) for p in root.glob(pattern)]
        return "\n".join(hits[:200]) or "(no matches)"