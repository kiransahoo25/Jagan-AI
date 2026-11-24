import urllib.parse
import xml.etree.ElementTree as ET

import requests

from ..rag import ingest
from .registry import Registry


def register(reg: Registry, cfg, store):
    @reg.tool(
        "arxiv_search",
        "Search arXiv. Returns titles, authors, ids, abstracts.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        },
    )
    def arxiv_search(query, max_results=5):
        url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"search_query": f"all:{query}", "max_results": max_results}
        )
        r = requests.get(url, timeout=20)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.text)

        out = []
        for e in root.findall("a:entry", ns):
            title = " ".join(e.find("a:title", ns).text.split())
            authors = ", ".join(
                a.find("a:name", ns).text for a in e.findall("a:author", ns)
            )
            link = e.find("a:id", ns).text
            summary = " ".join(e.find("a:summary", ns).text.split())[:600]
            out.append(f"**{title}**\n{authors}\n{link}\n{summary}")

        return "\n\n".join(out) or "no results"

    @reg.tool(
        "search_docs",
        "Semantic search over indexed local docs/code. Returns path:line + snippet.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer"},
            },
            "required": ["query"],
        },
    )
    def search_docs(query, k=6):
        hits = store.search(query, k)
        return (
            "\n\n".join(f"[{p}:{s}]\n{t[:500]}" for _, p, s, t in hits)
            or "no hits"
        )

    @reg.tool(
        "index_path",
        "Index a directory for semantic search.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )
    def index_path(path):
        ingest((cfg.project_root / path).resolve(), store)
        return f"indexed {path}"