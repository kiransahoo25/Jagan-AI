from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable
    dangerous: bool = False

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Registry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def tool(
        self,
        name: str,
        description: str,
        parameters: dict | None = None,
        dangerous: bool = False,
    ):
        def deco(fn):
            self.tools[name] = Tool(
                name,
                description,
                parameters or {"type": "object", "properties": {}},
                fn,
                dangerous,
            )
            return fn

        return deco

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def call(self, name: str, args: dict) -> str:
        return str(self.tools[name].fn(**args))

    def is_dangerous(self, name: str) -> bool:
        return self.tools[name].dangerous