"""
Doubt Scouts positioning scraper v2.

Two modes:
  - depth=shallow  (default) -- single-page homepage diagnosis, same as v1.
  - depth=deep -- BFS crawl up to 100 pages, extract customer stories via
    Claude API, produce a superconsumer report.

Deep mode is async: POST /analyze returns immediately with a job_id.
Poll GET /job/{job_id} for the result.

Deploy to Render, Fly.io, Railway, or any Docker host.
"""

from __future__ import annotations

import os
import re
import time
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field as dc_field
from typing import Any
from urllib.parse import urlparse, urljoin, urldefrag, urlencode, parse_qs

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Doubt Scouts Positioning Scraper", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("DOUBT_SCOUTS_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HYPE_WORDS = [
    "disrupt", "disruption", "disruptive",
    "game-changer", "game changing", "game-changing",
    "cutting-edge", "cutting edge",
    "next-generation", "next generation", "next-gen",
    "best-in-class", "best in class",
    "world-class", "world class",
    "seamless", "seamlessly",
    "unlock", "unlocking",
    "synergy", "synergies",
    "thought leadership", "thought leader",
    "revolutionary", "revolutionize",
    "empower", "empowering",
    "leverage", "leveraging",
    "robust", "scalable", "holistic",
    "state-of-the-art", "state of the art",
    "end-to-end", "end to end",
    "turnkey", "bleeding-edge", "bleeding edge",
    "ai-powered", "ai powered",
    "mission-critical", "mission critical",
    "best-of-breed", "best of breed",
    "frictionless",
]

GENERIC_PHRASES = [
    "helps companies", "helps teams", "helps businesses",
    "grow faster", "scale faster",
    "all-in-one", "all in one",
    "platform for", "powerful platform", "modern platform",
    "operating system for", "the future of",
    "one place", "everything you need",
    "built for scale", "built for speed", "purpose-built", "trusted by",
]

POV_SIGNALS = [
    r"\bunlike\b", r"\binstead of\b", r"\bthe old way\b",
    r"\bstatus quo\b", r"\byesterday's\b", r"\byesterdays\b",
    r"\bthe problem with\b", r"\bmost \w+ still\b",
    r"\bbelieve\b", r"\bpoint of view\b", r"\bcategory\b",
]

MISSIONARY_SIGNALS = [
    r"\bcommunity\b", r"\bmovement\b", r"\bmanifesto\b",
    r"\bbelievers\b", r"\bjoin us\b", r"\bwe believe\b",
]

# URL path fragments that signal customer-related pages
CUSTOMER_PATH_SIGNALS = [
    "customer", "case-stud", "case_stud", "stories", "story",
    "testimonial", "success", "community", "partner", "about",
    "review", "spotlight", "showcase",
]

# High-priority paths to crawl first
PRIORITY_PATHS = [
    "/customers", "/case-studies", "/case-study", "/stories",
    "/testimonials", "/success-stories", "/about", "/community",
    "/partners", "/reviews", "/showcase",
]

SKIP_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
    ".css", ".js", ".zip", ".tar", ".gz", ".mp4", ".mp3",
    ".woff", ".woff2", ".ttf", ".eot", ".ico", ".xml", ".json",
}

# ---------------------------------------------------------------------------
# Pydantic models -- shallow analysis (v1 compat)
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    url: str = Field(..., description="The website URL to analyze.")
    depth: str = Field("shallow", description="'shallow' or 'deep'")


class Extracted(BaseModel):
    final_url: str
    title: str | None
    meta_description: str | None
    og_title: str | None
    og_description: str | None
    h1: str | None
    h2s: list[str]
    hero_subhead: str | None
    cta_texts: list[str]
    nav_items: list[str]
    first_paragraph: str | None
    word_count: int


class Diagnosis(BaseModel):
    named_problem: bool
    named_enemy: bool
    pov_strength: int
    hype_word_hits: list[str]
    generic_phrase_hits: list[str]
    languaging_consistency: bool
    missionary_signals: bool
    score: int
    verdict: str
    sharpest_roast: str


# ---------------------------------------------------------------------------
# Pydantic models -- deep analysis (v2)
# ---------------------------------------------------------------------------

class CustomerStory(BaseModel):
    source_url: str
    company_name: str | None = None
    vertical: str | None = None
    quote: str | None = None
    quoted_person: str | None = None
    quoted_title: str | None = None
    outcome: str | None = None
    language_echoes_vendor: bool = False


class VerticalCluster(BaseModel):
    vertical: str
    count: int
    companies: list[str]
    common_outcomes: list[str]


class SuperconsumerReport(BaseModel):
    total_stories_found: int
    verticals: list[VerticalCluster]
    language_echo_rate: float
    missionary_signals_in_customers: bool
    strongest_vertical: str | None
    superconsumer_verdict: str
    raw_stories: list[CustomerStory]


class CrawlMeta(BaseModel):
    pages_crawled: int
    customer_pages_found: int
    crawl_time_seconds: float


class DeepAnalysis(BaseModel):
    crawl_meta: CrawlMeta
    superconsumer_report: SuperconsumerReport


class AnalyzeResponse(BaseModel):
    ok: bool
    extracted: Extracted
    diagnosis: Diagnosis
    depth: str = "shallow"
    job_id: str | None = None
    deep_analysis: DeepAnalysis | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: AnalyzeResponse | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Job store (in-memory, TTL-based cleanup)
# ---------------------------------------------------------------------------

@dataclass
class Job:
    status: str
    created_at: float
    result: AnalyzeResponse | None = None
    error: str | None = None

_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()
JOB_TTL = 900  # 15 minutes


def _cleanup_jobs() -> None:
    now = time.time()
    expired = [jid for jid, j in _jobs.items() if now - j.created_at > JOB_TTL]
    for jid in expired:
        del _jobs[jid]


# ---------------------------------------------------------------------------
# Helpers -- fetching and extraction (v1, unchanged)
# ---------------------------------------------------------------------------

def normalize_url(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        raise HTTPException(status_code=400, detail="empty url")
    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw
    parsed = urlparse(raw)
    if not parsed.netloc:
        raise HTTPException(status_code=400, detail="invalid url")
    return raw


def _http_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }


def fetch(url: str, timeout: float = 12.0) -> tuple[str, str]:
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers=_http_headers()) as client:
            r = client.get(url)
            r.raise_for_status()
            return str(r.url), r.text
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"fetch failed: {e}")


def fetch_safe(url: str, timeout: float = 8.0) -> tuple[str, str] | None:
    """Like fetch() but returns None on failure instead of raising."""
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers=_http_headers()) as client:
            r = client.get(url)
            r.raise_for_status()
            content_type = r.headers.get("content-type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return None
            return str(r.url), r.text
    except Exception:
        return None


def first_text(el) -> str | None:
    if not el:
        return None
    t = el.get_text(" ", strip=True)
    return t or None


def extract(html: str, final_url: str) -> Extracted:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()

    title = first_text(soup.title) if soup.title else None

    def meta(name: str) -> str | None:
        el = soup.find("meta", attrs={"name": name})
        if el and el.get("content"):
            return el["content"].strip()
        return None

    def og(prop: str) -> str | None:
        el = soup.find("meta", attrs={"property": prop})
        if el and el.get("content"):
            return el["content"].strip()
        return None

    h1_el = soup.find("h1")
    h1 = first_text(h1_el)
    h2s = [first_text(h) for h in soup.find_all("h2")[:8]]
    h2s = [h for h in h2s if h]

    hero_sub = None
    if h1_el:
        sib = h1_el.find_next(["p", "h2", "h3", "div"])
        if sib:
            hero_sub = first_text(sib)
            if hero_sub and len(hero_sub) > 300:
                hero_sub = hero_sub[:300].rsplit(" ", 1)[0] + "..."

    ctas = []
    for b in soup.find_all(["button", "a"]):
        txt = first_text(b)
        if not txt:
            continue
        if 2 <= len(txt) <= 40 and any(k in txt.lower() for k in [
            "start", "get started", "book", "demo", "try", "sign up", "signup",
            "contact", "talk to", "see how", "learn more", "watch", "schedule",
            "request", "join", "buy", "download", "explore"
        ]):
            ctas.append(txt)
    ctas = list(dict.fromkeys(ctas))[:10]

    nav_items: list[str] = []
    nav = soup.find("nav")
    if nav:
        for a in nav.find_all("a")[:15]:
            t = first_text(a)
            if t and 2 <= len(t) <= 30:
                nav_items.append(t)
    nav_items = list(dict.fromkeys(nav_items))

    first_p = None
    for p in soup.find_all("p"):
        t = first_text(p)
        if t and len(t) > 40:
            first_p = t[:500]
            break

    body_text = soup.get_text(" ", strip=True)
    word_count = len(body_text.split())

    soup.decompose()  # free memory

    return Extracted(
        final_url=final_url,
        title=title,
        meta_description=meta("description") or og("og:description"),
        og_title=og("og:title"),
        og_description=og("og:description"),
        h1=h1, h2s=h2s,
        hero_subhead=hero_sub,
        cta_texts=ctas,
        nav_items=nav_items,
        first_paragraph=first_p,
        word_count=word_count,
    )


def diagnose(e: Extracted) -> Diagnosis:
    corpus = " ".join(filter(None, [
        e.title, e.meta_description, e.og_title, e.og_description,
        e.h1, e.hero_subhead, e.first_paragraph, " ".join(e.h2s),
    ])).lower()

    hype_hits = sorted({w for w in HYPE_WORDS if w in corpus})
    generic_hits = sorted({g for g in GENERIC_PHRASES if g in corpus})
    pov_strength = min(sum(1 for pat in POV_SIGNALS if re.search(pat, corpus)), 3)

    named_enemy = any(re.search(p, corpus) for p in [
        r"\bunlike\b", r"\binstead of\b", r"\bthe old way\b",
        r"\bstatus quo\b", r"\bthe problem with\b",
    ])

    problem_markers = [
        "problem", "broken", "painful", "stuck", "frustrat",
        "tired of", "sick of", "still ", "every ", "most ",
    ]
    named_problem = any(m in (e.h1 or "").lower() + " " + (e.hero_subhead or "").lower()
                        for m in problem_markers)

    languaging_consistent = False
    if e.h1 and e.meta_description:
        h1_words = {w for w in re.findall(r"[a-z]{5,}", e.h1.lower())}
        meta_words = {w for w in re.findall(r"[a-z]{5,}", e.meta_description.lower())}
        languaging_consistent = len(h1_words & meta_words) >= 2

    missionary = any(re.search(p, corpus) for p in MISSIONARY_SIGNALS)

    score = 50
    score += pov_strength * 8
    score += 10 if named_problem else -10
    score += 10 if named_enemy else 0
    score -= min(len(hype_hits) * 4, 24)
    score -= min(len(generic_hits) * 4, 20)
    score += 6 if languaging_consistent else -4
    score += 5 if missionary else 0
    score = max(0, min(100, score))

    parts: list[str] = []
    if named_problem:
        parts.append("The hero actually names a problem, which most homepages skip.")
    else:
        parts.append("The hero names a solution, not a problem. That is the first tell.")
    if hype_hits:
        sample = ", ".join(hype_hits[:3])
        parts.append(f"Hype words detected: {sample}. Every one of those is a sentence your competitor also uses.")
    if generic_hits:
        parts.append(f"Generic phrases found: {', '.join(generic_hits[:3])}. Pasteable onto any B2B site without breaking anything.")
    if named_enemy:
        parts.append("There is a named enemy or status quo, which is a rare and good sign.")
    else:
        parts.append("No named enemy. The status quo is walking free.")
    if pov_strength >= 2:
        parts.append("Point of view is showing up in the copy. Not a manifesto yet, but a pulse.")
    else:
        parts.append("No point of view surfaced. The page is describing features, not defending a belief.")
    verdict = " ".join(parts)

    if score >= 75:
        roast = "This one has real positioning. I would not change much, I would just amplify it."
    elif score >= 55:
        roast = "The bones are there, the language is still too polite to fight for them."
    elif score >= 35:
        roast = "Your homepage reads like a board deck that was asked to be a buyer story and refused."
    else:
        roast = "This is wallpaper. Paste it onto three competitors and no one would notice."

    return Diagnosis(
        named_problem=named_problem,
        named_enemy=named_enemy,
        pov_strength=pov_strength,
        hype_word_hits=hype_hits,
        generic_phrase_hits=generic_hits,
        languaging_consistency=languaging_consistent,
        missionary_signals=missionary,
        score=score,
        verdict=verdict,
        sharpest_roast=roast,
    )


# ---------------------------------------------------------------------------
# Deep crawl engine
# ---------------------------------------------------------------------------

def _canonical_url(url: str) -> str:
    """Strip fragment, sort query params for dedup."""
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    if parsed.query:
        params = parse_qs(parsed.query, keep_blank_values=True)
        sorted_q = urlencode(sorted(params.items()), doseq=True)
        url = parsed._replace(query=sorted_q).geturl()
    return url.rstrip("/")


def _same_domain(url: str, base_netloc: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc == base_netloc or parsed.netloc == ""


def _skip_url(url: str) -> bool:
    parsed = urlparse(url)
    path_lower = parsed.path.lower()
    return any(path_lower.endswith(ext) for ext in SKIP_EXTENSIONS)


def _is_customer_page(url: str, text: str) -> bool:
    """Heuristic: is this page about customers/testimonials?"""
    path_lower = urlparse(url).path.lower()
    if any(sig in path_lower for sig in CUSTOMER_PATH_SIGNALS):
        return True
    # Check body for testimonial signals
    text_lower = text[:5000].lower()
    signals = 0
    for marker in ["said", "according to", "testimonial", "case study",
                    "blockquote", "customer stor", "success stor",
                    '"we ', "'we ", "helped us", "allowed us",
                    "vp ", "ceo ", "director", "head of", "cto "]:
        if marker in text_lower:
            signals += 1
    return signals >= 3


def _trim_page_text(html: str, max_chars: int = 8000) -> str:
    """Extract clean text from HTML, capped to max_chars."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    soup.decompose()
    return text[:max_chars]


def crawl_site(start_url: str, max_pages: int = 100) -> tuple[list[str], list[tuple[str, str]]]:
    """
    BFS crawl from start_url, same domain only.

    Returns:
        all_urls: list of all crawled URLs
        customer_pages: list of (url, trimmed_text) for customer-related pages
    """
    parsed_start = urlparse(start_url)
    base_netloc = parsed_start.netloc
    base_scheme = parsed_start.scheme

    visited: set[str] = set()
    all_urls: list[str] = []
    customer_pages: list[tuple[str, str]] = []

    # Seed the queue: start URL + priority paths
    queue: deque[str] = deque()
    # Priority paths first
    for path in PRIORITY_PATHS:
        priority_url = f"{base_scheme}://{base_netloc}{path}"
        canon = _canonical_url(priority_url)
        if canon not in visited:
            queue.append(priority_url)
            visited.add(canon)
    # Homepage at the front (if not already added via priority)
    canon_start = _canonical_url(start_url)
    if canon_start not in visited:
        queue.appendleft(start_url)
        visited.add(canon_start)
    else:
        queue.appendleft(start_url)

    pages_crawled = 0
    while queue and pages_crawled < max_pages:
        url = queue.popleft()
        result = fetch_safe(url, timeout=8.0)
        if result is None:
            continue

        final_url, html = result
        pages_crawled += 1
        all_urls.append(final_url)

        # Check if this is a customer page
        trimmed = _trim_page_text(html)
        if _is_customer_page(final_url, trimmed):
            customer_pages.append((final_url, trimmed))

        # Extract links for BFS
        try:
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                abs_url = urljoin(final_url, href)
                canon = _canonical_url(abs_url)
                if canon in visited:
                    continue
                if not _same_domain(abs_url, base_netloc):
                    continue
                if _skip_url(abs_url):
                    continue
                visited.add(canon)
                # Prioritize customer-signal URLs
                path_lower = urlparse(abs_url).path.lower()
                if any(sig in path_lower for sig in CUSTOMER_PATH_SIGNALS):
                    queue.appendleft(abs_url)
                else:
                    queue.append(abs_url)
            soup.decompose()
        except Exception:
            pass

        # Polite delay
        time.sleep(0.2)

    return all_urls, customer_pages


# ---------------------------------------------------------------------------
# Claude API -- customer story extraction
# ---------------------------------------------------------------------------

def _get_anthropic_client():
    """Lazy import to avoid crash when anthropic is not needed."""
    import anthropic
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


EXTRACTION_PROMPT = """You are extracting structured customer testimonial data from a webpage.
Given the following text from {url}, extract ALL customer stories, testimonials, quotes, and case study references you can find.

For each one, return a JSON object with these fields:
- company_name: the customer company name (string or null)
- vertical: the industry/vertical of the customer, e.g. "fintech", "healthcare", "e-commerce", "devtools" (string or null)
- quote: the exact testimonial quote if present (string or null)
- quoted_person: name of the person quoted (string or null)
- quoted_title: their job title and company (string or null)
- outcome: any quantified result or outcome mentioned, e.g. "reduced churn by 40%" (string or null)

Return ONLY a valid JSON array of objects. If no testimonials found, return [].
Do not include any explanation or markdown, just the JSON array.

Page text:
{text}"""


def extract_customer_stories(
    customer_pages: list[tuple[str, str]],
    max_pages: int = 20,
) -> list[CustomerStory]:
    """Send customer pages to Claude for structured extraction."""
    if not ANTHROPIC_API_KEY:
        return []

    client = _get_anthropic_client()
    pages_to_process = customer_pages[:max_pages]
    all_stories: list[CustomerStory] = []

    # Process in batches of 3 using threads for speed
    def _extract_one(url: str, text: str) -> list[CustomerStory]:
        try:
            prompt = EXTRACTION_PROMPT.format(url=url, text=text[:6000])
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            # Parse JSON, handle potential markdown wrapping
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            import json
            items = json.loads(raw)
            stories = []
            for item in items:
                if isinstance(item, dict):
                    stories.append(CustomerStory(
                        source_url=url,
                        company_name=item.get("company_name"),
                        vertical=item.get("vertical"),
                        quote=item.get("quote"),
                        quoted_person=item.get("quoted_person"),
                        quoted_title=item.get("quoted_title"),
                        outcome=item.get("outcome"),
                    ))
            return stories
        except Exception:
            return []

    # Threaded extraction, 3 concurrent
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(_extract_one, url, text) for url, text in pages_to_process]
        for f in futures:
            all_stories.extend(f.result())

    return all_stories


# ---------------------------------------------------------------------------
# Superconsumer analysis
# ---------------------------------------------------------------------------

def _vendor_keywords(extracted: Extracted) -> set[str]:
    """Extract meaningful words from vendor's own copy for echo detection."""
    corpus = " ".join(filter(None, [
        extracted.h1, extracted.meta_description,
        extracted.hero_subhead, " ".join(extracted.h2s),
        " ".join(extracted.cta_texts),
    ])).lower()
    # Words 4+ chars, excluding common stopwords
    stopwords = {"that", "this", "with", "from", "your", "they", "their",
                 "have", "been", "will", "more", "about", "than", "into",
                 "also", "what", "when", "which", "were", "would", "could",
                 "should", "some", "them", "then", "these", "those"}
    words = set(re.findall(r"[a-z]{4,}", corpus))
    return words - stopwords


def build_superconsumer_report(
    stories: list[CustomerStory],
    extracted: Extracted,
) -> SuperconsumerReport:
    """Analyze customer stories for superconsumer signals."""
    vendor_words = _vendor_keywords(extracted)

    # Language echo detection
    echo_count = 0
    for story in stories:
        if story.quote:
            quote_words = set(re.findall(r"[a-z]{4,}", story.quote.lower()))
            overlap = len(quote_words & vendor_words)
            total = len(quote_words) if quote_words else 1
            if overlap / total > 0.12:
                story.language_echoes_vendor = True
                echo_count += 1

    echo_rate = echo_count / len(stories) if stories else 0.0

    # Missionary signals in customer language
    missionary_in_customers = False
    for story in stories:
        if story.quote:
            for pat in MISSIONARY_SIGNALS:
                if re.search(pat, story.quote.lower()):
                    missionary_in_customers = True
                    break

    # Vertical clustering
    vertical_map: dict[str, list[CustomerStory]] = {}
    for story in stories:
        v = (story.vertical or "unknown").lower().strip()
        vertical_map.setdefault(v, []).append(story)

    clusters = []
    for vert, vert_stories in sorted(vertical_map.items(), key=lambda x: -len(x[1])):
        companies = list({s.company_name for s in vert_stories if s.company_name})
        outcomes = list({s.outcome for s in vert_stories if s.outcome})[:5]
        clusters.append(VerticalCluster(
            vertical=vert,
            count=len(vert_stories),
            companies=companies,
            common_outcomes=outcomes,
        ))

    strongest = clusters[0].vertical if clusters and clusters[0].count >= 2 else None

    # Verdict
    if not stories:
        verdict = (
            "No public customer stories detected. Either this company is pre-traction, "
            "or they are hiding their best proof. Both are red flags for category design. "
            "A company that has superconsumers puts them on stage. This company has an empty stage."
        )
    elif strongest and clusters[0].count >= 3 and echo_rate > 0.25:
        verdict = (
            f"Superconsumer cluster detected in {strongest}. {clusters[0].count} stories, "
            f"and {echo_rate:.0%} of customers are echoing the vendor's own language. "
            "That is the hallmark of category design that is working. "
            "The question is whether the company knows it and is leaning into it, "
            "or stumbled into it by accident."
        )
    elif strongest and clusters[0].count >= 3:
        verdict = (
            f"There is a vertical cluster in {strongest} ({clusters[0].count} stories), "
            "but customers are not echoing the company's language. "
            "The superconsumers might exist, but the company has not given them the words yet. "
            "That is a languaging problem, not a product problem."
        )
    elif stories and echo_rate > 0.3:
        verdict = (
            "The few customers who speak publicly are parroting the company's language. "
            "Good sign, but the sample is too small to call it a movement. "
            "There are believers, just not enough of them on display."
        )
    else:
        verdict = (
            "Customer stories exist but scatter across verticals with no clustering. "
            "No superconsumer pattern is forming. The company is selling to anyone who will buy. "
            "That is the opposite of category design."
        )

    return SuperconsumerReport(
        total_stories_found=len(stories),
        verticals=clusters,
        language_echo_rate=round(echo_rate, 3),
        missionary_signals_in_customers=missionary_in_customers,
        strongest_vertical=strongest,
        superconsumer_verdict=verdict,
        raw_stories=stories,
    )


# ---------------------------------------------------------------------------
# Background deep crawl runner
# ---------------------------------------------------------------------------

def _run_deep_crawl(job_id: str, url: str, shallow_result: AnalyzeResponse) -> None:
    """Runs in a background thread."""
    try:
        start = time.time()
        all_urls, customer_pages = crawl_site(url, max_pages=100)
        stories = extract_customer_stories(customer_pages)
        report = build_superconsumer_report(stories, shallow_result.extracted)
        elapsed = time.time() - start

        deep = DeepAnalysis(
            crawl_meta=CrawlMeta(
                pages_crawled=len(all_urls),
                customer_pages_found=len(customer_pages),
                crawl_time_seconds=round(elapsed, 1),
            ),
            superconsumer_report=report,
        )
        result = shallow_result.model_copy(update={
            "deep_analysis": deep,
            "depth": "deep",
        })
        with _jobs_lock:
            _jobs[job_id].status = "complete"
            _jobs[job_id].result = result
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id].status = "failed"
            _jobs[job_id].error = str(e)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def check_auth(x_api_key: str | None) -> None:
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "doubt-scouts-positioning-scraper",
        "version": "2.0",
        "endpoints": {
            "POST /analyze": "Positioning diagnosis. depth=shallow (default) or depth=deep.",
            "GET /job/{job_id}": "Poll for deep analysis results.",
            "GET /health": "Health check.",
        },
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest, x_api_key: str | None = Header(default=None)) -> AnalyzeResponse:
    check_auth(x_api_key)
    url = normalize_url(req.url)

    # Always run shallow analysis synchronously
    final_url, html = fetch(url)
    extracted = extract(html, final_url)
    diagnosis = diagnose(extracted)

    shallow_result = AnalyzeResponse(
        ok=True,
        extracted=extracted,
        diagnosis=diagnosis,
        depth="shallow",
    )

    if req.depth == "deep":
        if not ANTHROPIC_API_KEY:
            raise HTTPException(
                status_code=400,
                detail="deep analysis requires ANTHROPIC_API_KEY to be configured on the server",
            )
        # Kick off background job
        job_id = str(uuid.uuid4())
        with _jobs_lock:
            _cleanup_jobs()
            _jobs[job_id] = Job(status="processing", created_at=time.time())
        shallow_result.job_id = job_id
        shallow_result.depth = "deep"

        thread = threading.Thread(
            target=_run_deep_crawl,
            args=(job_id, url, shallow_result),
            daemon=True,
        )
        thread.start()

    return shallow_result


@app.get("/job/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str, x_api_key: str | None = Header(default=None)) -> JobStatusResponse:
    check_auth(x_api_key)
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found or expired")
    return JobStatusResponse(
        job_id=job_id,
        status=job.status,
        result=job.result,
        error=job.error,
    )
