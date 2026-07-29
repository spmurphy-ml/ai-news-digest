"""
AI News Digest — a LangGraph workflow.

Pulls recent AI news from a set of RSS feeds, summarizes each item,
assigns a category, and renders a dated markdown digest.

Graph:
    fetch -> filter -> summarize -> categorize -> render

Run:
    python main.py
    python main.py --days 3 --max-items 15
"""

import argparse
import os
from datetime import datetime, timedelta, timezone
from typing import TypedDict

import feedparser
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

load_dotenv()

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Multiple sources on purpose. Feeds move and break; one dead source
# should degrade the output, not kill the run.
FEEDS = [
    ("The Batch", "https://www.deeplearning.ai/the-batch/rss/"),
    ("Hugging Face", "https://huggingface.co/blog/feed.xml"),
    ("Google AI", "https://blog.google/technology/ai/rss/"),
    ("arXiv cs.AI", "http://export.arxiv.org/rss/cs.AI"),
]

CATEGORIES = [
    "Research",
    "Product & Tooling",
    "Policy & Governance",
    "Industry & Funding",
    "Other",
]


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

class DigestState(TypedDict):
    """Shared state passed between nodes. Each node reads it and returns
    a partial update, which LangGraph merges in."""
    days: int
    max_items: int
    items: list[dict]      # raw entries from the feeds
    kept: list[dict]       # entries that survived the filter
    summarized: list[dict] # entries with a 'summary' key added
    categorized: list[dict]# entries with a 'category' key added
    digest: str            # final rendered markdown
    errors: list[str]      # feeds that failed, for the report footer


# --------------------------------------------------------------------------
# LLM setup (optional)
# --------------------------------------------------------------------------

def get_llm():
    """Return a chat model, or None if no API key is configured.

    The workflow is designed to run either way — without a key it falls
    back to extractive summaries so you can still see the graph work.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model="claude-sonnet-4-6", max_tokens=300)


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------

def fetch(state: DigestState) -> dict:
    """Pull entries from every configured feed."""
    items, errors = [], []

    for source, url in FEEDS:
        try:
            parsed = feedparser.parse(url)
            if parsed.bozo and not parsed.entries:
                errors.append(f"{source}: no entries returned")
                continue
            for entry in parsed.entries:
                items.append({
                    "source": source,
                    "title": entry.get("title", "Untitled"),
                    "link": entry.get("link", ""),
                    "published": entry.get("published_parsed")
                                 or entry.get("updated_parsed"),
                    "raw": entry.get("summary", "") or entry.get("description", ""),
                })
        except Exception as exc:                      # noqa: BLE001
            errors.append(f"{source}: {exc}")

    print(f"[fetch]      {len(items)} items from {len(FEEDS)} feeds "
          f"({len(errors)} failed)")
    return {"items": items, "errors": errors}


def filter_recent(state: DigestState) -> dict:
    """Keep only items published within the window, newest first."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=state["days"])
    kept = []

    for item in state["items"]:
        published = item.get("published")
        if not published:
            continue
        when = datetime(*published[:6], tzinfo=timezone.utc)
        if when >= cutoff:
            item["date"] = when
            kept.append(item)

    kept.sort(key=lambda i: i["date"], reverse=True)
    kept = kept[: state["max_items"]]

    print(f"[filter]     {len(kept)} items in the last {state['days']} days")
    return {"kept": kept}


def summarize(state: DigestState) -> dict:
    """Two sentences per item. Uses an LLM when available, otherwise
    falls back to the first chunk of the feed's own description."""
    llm = get_llm()
    out = []

    for item in state["kept"]:
        text = _strip_html(item["raw"])[:2000]

        if llm and text:
            prompt = (
                "Summarize this AI news item in exactly two sentences. "
                "Be concrete and factual. No preamble.\n\n"
                f"Title: {item['title']}\n\n{text}"
            )
            try:
                item["summary"] = llm.invoke(prompt).content.strip()
            except Exception:                          # noqa: BLE001
                item["summary"] = _extractive(text)
        else:
            item["summary"] = _extractive(text)

        out.append(item)

    mode = "LLM" if llm else "extractive (no API key set)"
    print(f"[summarize]  {len(out)} summaries — {mode}")
    return {"summarized": out}


def categorize(state: DigestState) -> dict:
    """Assign each item to one of CATEGORIES."""
    llm = get_llm()
    out = []

    for item in state["summarized"]:
        if llm:
            prompt = (
                "Classify this AI news item into exactly one category. "
                f"Reply with only the category name.\n\n"
                f"Categories: {', '.join(CATEGORIES)}\n\n"
                f"Title: {item['title']}\nSummary: {item['summary']}"
            )
            try:
                guess = llm.invoke(prompt).content.strip()
                item["category"] = guess if guess in CATEGORIES else "Other"
            except Exception:                          # noqa: BLE001
                item["category"] = _keyword_category(item)
        else:
            item["category"] = _keyword_category(item)

        out.append(item)

    print(f"[categorize] {len(out)} items classified")
    return {"categorized": out}


def render(state: DigestState) -> dict:
    """Group by category and write the markdown digest."""
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [f"# AI News Digest — {today}", ""]
    lines.append(f"*{len(state['categorized'])} items from the last "
                 f"{state['days']} days.*")
    lines.append("")

    for category in CATEGORIES:
        group = [i for i in state["categorized"] if i["category"] == category]
        if not group:
            continue
        lines.append(f"## {category}")
        lines.append("")
        for item in group:
            lines.append(f"### [{item['title']}]({item['link']})")
            lines.append(f"*{item['source']} — "
                         f"{item['date'].strftime('%b %d')}*")
            lines.append("")
            lines.append(item["summary"])
            lines.append("")

    if state["errors"]:
        lines.append("---")
        lines.append("")
        lines.append("**Feeds that failed this run:**")
        for err in state["errors"]:
            lines.append(f"- {err}")
        lines.append("")

    digest = "\n".join(lines)
    filename = f"digest-{today}.md"
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(digest)

    print(f"[render]     wrote {filename}")
    return {"digest": digest}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _strip_html(text: str) -> str:
    import re
    return re.sub(r"<[^>]+>", " ", text).replace("&nbsp;", " ").strip()


def _extractive(text: str) -> str:
    """Cheap fallback: first two sentences."""
    if not text:
        return "_No description available._"
    parts = text.replace("\n", " ").split(". ")
    return ". ".join(parts[:2]).strip().rstrip(".") + "."


def _keyword_category(item: dict) -> str:
    """Cheap fallback classifier."""
    blob = f"{item['title']} {item.get('summary', '')}".lower()
    if any(w in blob for w in ("arxiv", "paper", "benchmark", "model card")):
        return "Research"
    if any(w in blob for w in ("regulation", "policy", "eu ai act", "safety")):
        return "Policy & Governance"
    if any(w in blob for w in ("raises", "funding", "acquire", "series")):
        return "Industry & Funding"
    if any(w in blob for w in ("release", "launch", "api", "sdk", "tool")):
        return "Product & Tooling"
    return "Other"


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(DigestState)

    graph.add_node("fetch", fetch)
    graph.add_node("filter", filter_recent)
    graph.add_node("summarize", summarize)
    graph.add_node("categorize", categorize)
    graph.add_node("render", render)

    graph.set_entry_point("fetch")
    graph.add_edge("fetch", "filter")
    graph.add_edge("filter", "summarize")
    graph.add_edge("summarize", "categorize")
    graph.add_edge("categorize", "render")
    graph.add_edge("render", END)

    return graph.compile()


def main():
    parser = argparse.ArgumentParser(description="Build an AI news digest.")
    parser.add_argument("--days", type=int, default=7,
                        help="how far back to look (default: 7)")
    parser.add_argument("--max-items", type=int, default=12,
                        help="cap on items in the digest (default: 12)")
    args = parser.parse_args()

    app = build_graph()
    app.invoke({
        "days": args.days,
        "max_items": args.max_items,
        "items": [],
        "kept": [],
        "summarized": [],
        "categorized": [],
        "digest": "",
        "errors": [],
    })


if __name__ == "__main__":
    main()
