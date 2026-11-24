import shlex
import subprocess

from .registry import Registry


def register(reg: Registry, cfg):
    def run(cmd: str) -> str:
        r = subprocess.run(
            cmd, shell=True, cwd=cfg.project_root, capture_output=True, text=True
        )
        return (r.stdout + r.stderr)[:20000] or "(no output)"

    @reg.tool("git_status", "Short git status.", {})
    def git_status():
        return run("git status --short --branch")

    @reg.tool(
        "git_diff",
        "Git diff.",
        {"type": "object", "properties": {"staged": {"type": "boolean"}}},
    )
    def git_diff(staged=False):
        return run("git diff --staged" if staged else "git diff")

    @reg.tool(
        "git_log",
        "Recent commits.",
        {"type": "object", "properties": {"n": {"type": "integer"}}},
    )
    def git_log(n=10):
        return run(f"git log --oneline -n {n}")

    @reg.tool(
        "git_commit",
        "Stage all changes and commit.",
        {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        dangerous=True,
    )
    def git_commit(message):
        run("git add -A")
        return run(f"git commit -m {shlex.quote(message)}")