import requests
from bs4 import BeautifulSoup

from .registry import Registry


def register(reg: Registry):
    @reg.tool(
        "fetch_url",
        "Fetch a URL and return readable text.",
        {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    )
    def fetch_url(url):
        r = requests.get(url, timeout=20, headers={"User-Agent": "Jagan/0.1"})
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "aside"]):
            tag.decompose()
        text = " ".join(soup.get_text(" ").split())
        return text[:20000]