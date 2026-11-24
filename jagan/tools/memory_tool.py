from .registry import Registry


def register(reg: Registry, mem):
    @reg.tool(
        "remember",
        "Store a durable fact across sessions.",
        {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
                "namespace": {"type": "string"},
            },
            "required": ["key", "value"],
        },
    )
    def remember(key, value, namespace="global"):
        return mem.remember(key, value, namespace)

    @reg.tool(
        "recall",
        "Retrieve facts. Omit key to list the namespace.",
        {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "namespace": {"type": "string"},
            },
        },
    )
    def recall(key=None, namespace="global"):
        return mem.recall(key, namespace)