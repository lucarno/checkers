import requests
from bs4 import BeautifulSoup


def scrape_guidelines(url: str) -> str:
    """Fetch a journal guidelines page and extract clean text content."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
        tag.decompose()

    # Try to find the main content area
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", {"role": "main"})
        or soup.find("div", class_=lambda c: c and "content" in c.lower() if c else False)
    )

    target = main if main else soup.body if soup.body else soup

    text = target.get_text(separator="\n", strip=True)

    # Collapse excessive blank lines
    lines = []
    blank_count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            blank_count += 1
            if blank_count <= 2:
                lines.append("")
        else:
            blank_count = 0
            lines.append(stripped)

    cleaned = "\n".join(lines).strip()

    # Truncate if extremely long (keep first ~15k chars for LLM context)
    if len(cleaned) > 15000:
        cleaned = cleaned[:15000] + "\n\n[Content truncated for processing...]"

    return cleaned
