"""Fetch reference pages supplied by the user and extract their visible text."""
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "DeepTrace/0.1 (plagiarism research tool)"}


def fetch_text(url: str, timeout: int = 10) -> str:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    return " ".join(soup.get_text(" ").split())


def fetch_many(urls: list[str]) -> dict[str, str]:
    """Fetch several URLs, skipping any that fail."""
    pages = {}
    for url in urls:
        try:
            pages[url] = fetch_text(url)
        except requests.RequestException:
            continue
    return pages
