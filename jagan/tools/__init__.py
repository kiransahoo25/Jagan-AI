from . import files, git, memory_tool, research, shell, web


def register_all(reg, cfg, mem, store):
    files.register(reg, cfg)
    shell.register(reg, cfg)
    git.register(reg, cfg)
    web.register(reg)
    research.register(reg, cfg, store)
    memory_tool.register(reg, mem)