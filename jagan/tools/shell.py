import subprocess

from .registry import Registry


def register(reg: Registry, cfg):
    @reg.tool(
        "run_shell",
        "Run a shell command in the project root. Returns exit code + output.",
        {
            "type": "object",
            "properties": {
                "cmd": {"type": "string"},
                "timeout": {"type": "integer"},
            },
            "required": ["cmd"],
        },
        dangerous=True,
    )
    def run_shell(cmd, timeout=120):
        try:
            r = subprocess.run(
                cmd,
                shell=True,
                cwd=cfg.project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return f"exit={r.returncode}\n{(r.stdout + r.stderr)[:20000]}"
        except subprocess.TimeoutExpired:
            return "error: timed out"