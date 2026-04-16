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

import logging
import os
import re
import time
import threading
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scraper")
from collections import deque
from dataclasses import dataclass, field as dc_field
from typing import Any
from urllib.parse import urlparse, urljoin, urldefrag, urlencode, parse_qs

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Doubt Scouts Positioning Scraper", version="2.5")

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
    "review", "spotlight", "showcase", "proof", "evidence",
    "result", "impact", "wall-of-love", "love", "trust",
    "who-uses", "used-by", "powered-by", "built-with",
    "logo", "brand", "client", "portfolio", "roster",
    "quote", "feedback", "endorsement", "reference",
    "video", "webinar", "interview",
]

# High-priority paths to crawl first
PRIORITY_PATHS = [
    "/customers", "/case-studies", "/case-study", "/stories",
    "/testimonials", "/success-stories", "/about", "/community",
    "/partners", "/reviews", "/showcase", "/proof", "/results",
    "/wall-of-love", "/love", "/trust", "/clients", "/logos",
    "/who-uses", "/customers/all", "/resources/case-studies",
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
    depth: str = Field("shallow", description="'shallow', 'quick', or 'deep'")
    compact: bool = Field(True, description="Strip raw_stories and trim category analysis for voice agent consumption. Set false for full data.")


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
    evidence_type: str = "unknown"  # quote, case_study, logo, metric, video, blurb, trust_list
    language_echoes_vendor: bool = False
    # Category design fields (extracted per-story, aggregated later)
    belief_shift_from: str | None = None  # what they believed before
    belief_shift_to: str | None = None    # what they believe now
    commitment_signal: str | None = None  # identity-level language ("can't go back", "changed how I think")
    from_to_from: str | None = None       # old world description
    from_to_to: str | None = None         # new world description
    adjacent_products: list[str] = Field(default_factory=list)  # other tools/products mentioned
    # Superconsumer scoring (computed after extraction, not from LLM)
    superconsumer_score: int = 0  # 0-3: 0=logo/metric only, 1=has quote, 2=has worldview shift, 3=true superconsumer
    is_superconsumer: bool = False  # True if score >= 2


def score_superconsumer(story: CustomerStory) -> CustomerStory:
    """Score a story on a 0-3 superconsumer scale.
    0 = logo, metric, or blurb with no signal (just a customer)
    1 = has a quote or named person but no worldview evidence (satisfied customer)
    2 = has belief shift OR from/to OR commitment signal (showing worldview change)
    3 = has multiple signals: belief shift + commitment, or all three (true superconsumer)
    """
    score = 0
    signals = 0

    # Level 1: has a real quote or named person (not just a logo)
    if story.quote and len(story.quote) > 20:
        score = max(score, 1)
    if story.quoted_person:
        score = max(score, 1)

    # Count worldview signals
    if story.belief_shift_from and story.belief_shift_to:
        signals += 1
    if story.commitment_signal and len(story.commitment_signal) > 10:
        signals += 1
    if story.from_to_from and story.from_to_to:
        signals += 1

    # Level 2: at least one worldview signal
    if signals >= 1:
        score = max(score, 2)

    # Level 3: multiple worldview signals (true superconsumer)
    if signals >= 2:
        score = 3

    story.superconsumer_score = score
    story.is_superconsumer = score >= 2
    return story


class VerticalCluster(BaseModel):
    vertical: str
    count: int
    companies: list[str]
    common_outcomes: list[str]


class BuyerPersona(BaseModel):
    """Aggregated role/title pattern from customer stories."""
    role_pattern: str  # e.g. "VP of Sales", "Director", "Head of"
    count: int
    examples: list[str]  # e.g. ["Paul Santarelli, Chief Sales Officer at Pitchbook"]


class SuperconsumerReport(BaseModel):
    total_stories_found: int
    superconsumer_count: int = 0  # stories with score >= 2
    regular_customer_count: int = 0  # stories with score < 2
    superconsumer_ratio: str = "0/0"  # "3 of 12" format for voice narration
    superconsumer_names: list[str] = Field(default_factory=list)  # company names of true superconsumers
    verticals: list[VerticalCluster]
    buyer_personas: list[BuyerPersona]
    language_echo_rate: float
    missionary_signals_in_customers: bool
    strongest_vertical: str | None
    strongest_buyer_persona: str | None
    superconsumer_verdict: str
    raw_stories: list[CustomerStory]


class CrawlMeta(BaseModel):
    pages_crawled: int
    customer_pages_found: int
    crawl_time_seconds: float


class BeliefShift(BaseModel):
    from_belief: str
    to_belief: str
    frequency: int = 1  # how many stories share this pattern
    example_companies: list[str] = Field(default_factory=list)


class FromToNarrative(BaseModel):
    from_world: str
    to_world: str
    frequency: int = 1
    example_companies: list[str] = Field(default_factory=list)


class CategoryDesignAnalysis(BaseModel):
    belief_shifts: list[BeliefShift] = Field(default_factory=list)
    commitment_signals: list[dict] = Field(default_factory=list)  # [{signal, company, person}]
    from_to_narratives: list[FromToNarrative] = Field(default_factory=list)
    adjacent_categories: list[dict] = Field(default_factory=list)  # [{product, mentioned_by}]


class IdealSuperconsumer(BaseModel):
    ideal_belief_before: str = ""
    ideal_belief_after: str = ""
    ideal_commitment_language: str = ""
    ideal_from_to: str = ""
    what_makes_them_super: str = ""


class SuperconsumerGap(BaseModel):
    ideal_profile: IdealSuperconsumer
    actual_superconsumer_count: int = 0
    actual_total_stories: int = 0
    gap_verdict: str = ""  # the money line Scout narrates


class DeepAnalysis(BaseModel):
    crawl_meta: CrawlMeta
    superconsumer_report: SuperconsumerReport
    category_analysis: CategoryDesignAnalysis | None = None
    superconsumer_gap: SuperconsumerGap | None = None


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

# ---------------------------------------------------------------------------
# Quick analysis cache (in-memory, TTL-based)
# ---------------------------------------------------------------------------

@dataclass
class CachedResult:
    result: AnalyzeResponse
    created_at: float

_quick_cache: dict[str, CachedResult] = {}
_cache_lock = threading.Lock()
CACHE_TTL = 3600  # 1 hour


def _get_cached_quick(url: str) -> AnalyzeResponse | None:
    """Return cached quick/deep result for URL, or None if expired/missing."""
    with _cache_lock:
        entry = _quick_cache.get(url)
        if entry and (time.time() - entry.created_at) < CACHE_TTL:
            logger.info(f"Cache hit for {url}")
            return entry.result
        if entry:
            del _quick_cache[url]
    return None


def _set_cached_quick(url: str, result: AnalyzeResponse) -> None:
    with _cache_lock:
        _quick_cache[url] = CachedResult(result=result, created_at=time.time())
        # Evict old entries if cache grows too large
        if len(_quick_cache) > 100:
            oldest = min(_quick_cache, key=lambda k: _quick_cache[k].created_at)
            del _quick_cache[oldest]


def _cleanup_jobs() -> None:
    now = time.time()
    expired = [jid for jid, j in _jobs.items() if now - j.created_at > JOB_TTL]
    for jid in expired:
        del _jobs[jid]


# ---------------------------------------------------------------------------
# Background crawl store (pre-warm customer analysis while homepage returns)
# ---------------------------------------------------------------------------

@dataclass
class BackgroundCrawl:
    status: str  # "running", "complete", "failed"
    created_at: float
    stories: list | None = None
    customer_pages_found: int = 0
    pages_crawled: int = 0
    crawl_time: float = 0.0
    error: str | None = None

_bg_crawls: dict[str, BackgroundCrawl] = {}
_bg_lock = threading.Lock()
BG_CRAWL_TTL = 600  # 10 minutes

def _start_bg_crawl(url: str) -> None:
    """Kick off customer page crawl in background. Idempotent -- skips if already running."""
    canon = _canonical_url(url)
    with _bg_lock:
        existing = _bg_crawls.get(canon)
        if existing and existing.status == "running":
            return  # already running
        if existing and existing.status == "complete" and (time.time() - existing.created_at) < BG_CRAWL_TTL:
            return  # fresh result already cached
        _bg_crawls[canon] = BackgroundCrawl(status="running", created_at=time.time())

    def _do_crawl():
        try:
            start = time.time()
            all_urls, customer_pages = crawl_priority_only(url)
            stories = extract_customer_stories(customer_pages)
            elapsed = time.time() - start
            with _bg_lock:
                entry = _bg_crawls.get(canon)
                if entry:
                    entry.status = "complete"
                    entry.stories = stories
                    entry.customer_pages_found = len(customer_pages)
                    entry.pages_crawled = len(all_urls)
                    entry.crawl_time = round(elapsed, 1)
        except Exception as e:
            logger.error(f"Background crawl failed for {url}: {e}")
            with _bg_lock:
                entry = _bg_crawls.get(canon)
                if entry:
                    entry.status = "failed"
                    entry.error = str(e)

    thread = threading.Thread(target=_do_crawl, daemon=True)
    thread.start()

def _get_bg_crawl(url: str, timeout: float = 30.0) -> BackgroundCrawl | None:
    """Wait for background crawl to finish (up to timeout seconds). Returns result or None."""
    canon = _canonical_url(url)
    deadline = time.time() + timeout
    while time.time() < deadline:
        with _bg_lock:
            entry = _bg_crawls.get(canon)
            if not entry:
                return None
            if entry.status in ("complete", "failed"):
                return entry
        time.sleep(0.5)
    # Timeout -- return whatever we have
    with _bg_lock:
        return _bg_crawls.get(canon)


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


def _is_customer_page(url: str, html: str) -> bool:
    """Heuristic: is this page about customers/testimonials?
    Checks URL path, body text, AND HTML structure (alt tags, data attrs, schema)."""
    path_lower = urlparse(url).path.lower()
    if any(sig in path_lower for sig in CUSTOMER_PATH_SIGNALS):
        return True

    html_lower = html[:30000].lower()

    # Check for structured signals in raw HTML (catches JS-rendered logo walls)
    html_signals = 0
    for marker in [
        "customer-logo", "logo-wall", "logo-grid", "logo-strip", "logo-carousel",
        "testimonial", "blockquote", "customer-quote", "quote-card",
        "case-study", "casestudy", "success-story",
        '"customer"', "'customer'", "data-customer", "data-company",
        "schema.org/review", "schema.org/testimonial",
        "trust-badge", "social-proof", "proof-section",
        "wall-of-love", "customer-card", "client-logo",
    ]:
        if marker in html_lower:
            html_signals += 1
    if html_signals >= 2:
        return True

    # Check body text for testimonial language
    text_lower = html_lower[:10000]
    text_signals = 0
    for marker in [
        "said", "according to", "testimonial", "case study",
        "customer stor", "success stor",
        '"we ', "'we ", "helped us", "allowed us", "enabled us",
        "switched from", "moved from", "replaced",
        "vp ", "ceo ", "director", "head of", "cto ", "coo ",
        "chief ", "founder", "co-founder", "manager",
        "reduced", "increased", "improved", "saved",
        "% ", "roi", "revenue", "growth",
        "trusted by", "used by", "loved by", "chosen by",
        "customers include", "our customers", "who uses",
    ]:
        if marker in text_lower:
            text_signals += 1
    if text_signals >= 3:
        return True

    # Check for logo image alt tags (even on JS-heavy pages, img tags are often in SSR HTML)
    alt_company_count = 0
    for match in re.finditer(r'alt=["\']([^"\']{3,60})["\']', html[:50000]):
        alt = match.group(1).lower()
        if any(w in alt for w in ["logo", "customer", "client", "partner", "brand", "company"]):
            alt_company_count += 1
    if alt_company_count >= 3:
        return True

    return False


def _extract_html_signals(html: str) -> dict:
    """Extract customer signals directly from HTML structure.
    Catches logo walls, alt tags, data attributes, and structured data
    even when the visible text is thin (JS-rendered sites)."""
    signals = {
        "logo_companies": [],
        "alt_companies": [],
        "structured_mentions": [],
        "blockquote_texts": [],
        "meta_customers": [],
    }

    soup = BeautifulSoup(html, "html.parser")

    # 1. Alt tags on images -- logo walls
    for img in soup.find_all("img", alt=True):
        alt = img.get("alt", "").strip()
        if not alt or len(alt) < 2 or len(alt) > 80:
            continue
        alt_lower = alt.lower()
        # Skip generic alts
        if alt_lower in ("logo", "image", "icon", "photo", "picture", "avatar", "hero"):
            continue
        # Check if it looks like a company name (in a logo context)
        parent_classes = " ".join(img.parent.get("class", [])).lower() if img.parent else ""
        grandparent_classes = " ".join(img.parent.parent.get("class", [])).lower() if img.parent and img.parent.parent else ""
        context = parent_classes + " " + grandparent_classes
        if any(w in context for w in ["logo", "customer", "client", "partner", "brand", "trust", "proof", "carousel", "grid", "wall"]):
            # Strip " logo", " Logo" suffix
            clean = re.sub(r'\s*(logo|icon|image|badge)\s*$', '', alt, flags=re.IGNORECASE).strip()
            if clean and len(clean) > 1:
                signals["logo_companies"].append(clean)
        elif any(w in alt_lower for w in ["logo", "customer"]):
            clean = re.sub(r'\s*(logo|icon|image|badge)\s*$', '', alt, flags=re.IGNORECASE).strip()
            if clean and len(clean) > 1:
                signals["alt_companies"].append(clean)

    # 2. Blockquotes and quote elements
    for bq in soup.find_all(["blockquote", "q"]):
        text = bq.get_text(" ", strip=True)
        if text and len(text) > 20:
            signals["blockquote_texts"].append(text[:500])

    # 3. Elements with testimonial/quote classes
    for el in soup.find_all(class_=re.compile(r"testimonial|quote|review|feedback", re.I)):
        text = el.get_text(" ", strip=True)
        if text and len(text) > 20:
            signals["blockquote_texts"].append(text[:500])

    # 4. Schema.org / JSON-LD structured data
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            import json
            data = json.loads(script.string or "")
            items = data if isinstance(data, list) else [data]
            for item in items:
                t = item.get("@type", "")
                if t in ("Review", "Testimonial", "Recommendation"):
                    signals["structured_mentions"].append(item)
                if "review" in str(item).lower()[:200]:
                    signals["structured_mentions"].append(item)
        except Exception:
            pass

    # 5. Data attributes that mention customers
    for el in soup.find_all(attrs={"data-customer": True}):
        signals["meta_customers"].append(el.get("data-customer", ""))
    for el in soup.find_all(attrs={"data-company": True}):
        signals["meta_customers"].append(el.get("data-company", ""))

    soup.decompose()
    return signals


def _trim_page_text(html: str, max_chars: int = 12000) -> str:
    """Extract clean text from HTML, capped to max_chars.
    Preserves alt text and title attributes for logo wall detection."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    # Inject alt text and title attrs as visible text so Claude can see them
    for img in soup.find_all("img", alt=True):
        alt = img.get("alt", "").strip()
        if alt and len(alt) > 2:
            img.replace_with(f" [IMAGE: {alt}] ")
    for a in soup.find_all("a", title=True):
        title = a.get("title", "").strip()
        if title and title not in (a.get_text() or ""):
            a.append(f" [LINK_TITLE: {title}] ")
    text = soup.get_text("\n", strip=True)
    soup.decompose()
    return text[:max_chars]


def crawl_site(start_url: str, max_pages: int = 100) -> tuple[list[str], list[tuple[str, str, dict, str]]]:
    """
    BFS crawl from start_url, same domain only.

    Returns:
        all_urls: list of all crawled URLs
        customer_pages: list of (url, trimmed_text, html_signals, raw_html) for customer-related pages
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

        # Check if this is a customer page (pass raw HTML for structural detection)
        if _is_customer_page(final_url, html):
            trimmed = _trim_page_text(html)
            html_signals = _extract_html_signals(html)
            customer_pages.append((final_url, trimmed, html_signals, html))

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


def crawl_priority_only(start_url: str) -> tuple[list[str], list[tuple[str, str, dict, str]]]:
    """Fast crawl: concurrent fetch of priority paths + discovered case study links.
    Targets 10-25 pages, designed to complete in under 10 seconds."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    parsed = urlparse(start_url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    all_urls: list[str] = []
    customer_pages: list[tuple[str, str, dict, str]] = []
    visited: set[str] = set()

    def _fetch_and_process(url: str) -> tuple[str | None, str | None, list[str]]:
        """Fetch a URL, check if customer page, extract links. Returns (final_url, html, discovered_links)."""
        result = fetch_safe(url, timeout=8.0)
        if result is None:
            return None, None, []
        final_url, html = result
        links = []
        try:
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                abs_url = urljoin(final_url, href)
                if not _same_domain(abs_url, parsed.netloc):
                    continue
                path_lower = urlparse(abs_url).path.lower()
                if any(sig in path_lower for sig in CUSTOMER_PATH_SIGNALS):
                    links.append(abs_url)
            soup.decompose()
        except Exception:
            pass
        return final_url, html, links

    # Phase 1: Fetch all priority paths concurrently
    candidate_urls = [start_url] + [f"{base}{p}" for p in PRIORITY_PATHS]
    deduped = []
    for url in candidate_urls:
        canon = _canonical_url(url)
        if canon not in visited:
            visited.add(canon)
            deduped.append(url)

    discovered_links: list[str] = []

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_fetch_and_process, url): url for url in deduped}
        for future in as_completed(futures):
            final_url, html, links = future.result()
            if final_url is None:
                continue
            all_urls.append(final_url)
            if _is_customer_page(final_url, html):
                trimmed = _trim_page_text(html)
                html_signals = _extract_html_signals(html)
                customer_pages.append((final_url, trimmed, html_signals, html))
            for link in links:
                canon = _canonical_url(link)
                if canon not in visited:
                    visited.add(canon)
                    discovered_links.append(link)

    # Phase 2: Fetch discovered case study links concurrently (cap at 15)
    phase2 = discovered_links[:15]
    if phase2:
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(_fetch_and_process, url): url for url in phase2}
            for future in as_completed(futures):
                final_url, html, _ = future.result()
                if final_url is None:
                    continue
                all_urls.append(final_url)
                if _is_customer_page(final_url, html):
                    trimmed = _trim_page_text(html)
                    html_signals = _extract_html_signals(html)
                    customer_pages.append((final_url, trimmed, html_signals, html))

    logger.info(f"Quick crawl: {len(all_urls)} pages fetched, {len(customer_pages)} customer pages found")
    return all_urls, customer_pages


# ---------------------------------------------------------------------------
# Claude API -- customer story extraction
# ---------------------------------------------------------------------------

def _get_anthropic_client():
    """Lazy import to avoid crash when anthropic is not needed."""
    import anthropic
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


BATCH_EXTRACTION_PROMPT = """Extract ALL customer evidence from these webpages. Return a single JSON array.

Evidence types: quotes, case studies, logos, metrics, videos, blurbs, trust lists.

For each item return:
{{
  "source_url": "...",
  "company_name": "...",
  "vertical": "...",
  "quote": "...",
  "quoted_person": "...",
  "quoted_title": "...",
  "outcome": "...",
  "evidence_type": "quote|case_study|logo|metric|video|blurb|trust_list",
  "belief_shift_from": "what they used to believe, in plain conversational English",
  "belief_shift_to": "what they believe now, in plain conversational English",
  "commitment_signal": "any language showing identity change, not product satisfaction (null if absent)",
  "from_to_from": "the old assumption, written like you are explaining it to a friend over coffee",
  "from_to_to": "the new truth, written the same way",
  "adjacent_products": ["other tools or products mentioned alongside this vendor"]
}}

CRITICAL STYLE RULE for belief_shift and from_to fields:
Write like a human talking, not a consultant presenting. Short. Sharp. No jargon.

BAD examples (too corporate, too abstract):
- "support is a cost center requiring significant engineering and product team overhead"
- "customer support can be strategically leveraged for growth and competitive advantage"
- "manual processes create operational inefficiencies across the organization"

GOOD examples (conversational, specific, sounds like a person):
- "if you want good support, you have to keep hiring people"
- "the best support is when the customer never knows it was a machine"
- "forecasting is an art that lives in the VP's gut"
- "forecasting is a math problem, and the math is better than the gut"
- "the only way to handle more tickets is more agents"
- "most tickets should never become tickets in the first place"
- "compliance and speed are enemies"
- "compliance and speed are the same thing if you design it right"

The test: read your from_to out loud. If it sounds like a slide deck, rewrite it. If it sounds like something a smart founder would say at a bar, keep it.

Rules:
- Extract EVERY company name even from logo alt tags. One entry per company per page. Guess the vertical.
- For belief_shift: look for before/after THINKING. "We used to think..." or "We realized..." If the customer only mentions outcomes (saved hours, reduced costs), infer the belief underneath. Saving 6700 hours means they used to believe "you need people for this" and now believe "this work should not exist at all."
- For commitment_signal: look for EMOTIONAL, IDENTITY language. "I can't go back." "This changed how I see the problem." "We're a different company now." NOT metrics. NOT product praise. NOT "great tool" or "easy to use." The signal is that their identity shifted.
- For from_to: this is the most important field. It must be a BELIEF, not a process. Ask: what would this customer tell their past self was wrong? Write the FROM as that old wrong belief. Write the TO as the new truth they now take for granted. Keep it under 15 words each.
- For adjacent_products: any other tools, platforms, or categories mentioned in the same story.
- Set fields to null if the evidence is genuinely not there. But try hard to infer beliefs from outcomes before giving up.
- Return ONLY a JSON array, no markdown.

PAGES:
{pages}"""


def _format_html_signals(signals: dict) -> str:
    """Format HTML signals dict into readable text for Claude."""
    parts = []
    if signals.get("logo_companies"):
        parts.append(f"Logos: {', '.join(signals['logo_companies'])}")
    if signals.get("alt_companies"):
        parts.append(f"Alts: {', '.join(signals['alt_companies'])}")
    if signals.get("blockquote_texts"):
        for bq in signals["blockquote_texts"][:5]:
            parts.append(f"BQ: {bq[:200]}")
    if signals.get("meta_customers"):
        parts.append(f"Data-customers: {', '.join(signals['meta_customers'])}")
    return " | ".join(parts) if parts else ""


def extract_customer_stories(
    customer_pages: list[tuple[str, str, dict, str]],
    max_pages: int = 50,
) -> list[CustomerStory]:
    """Batch pages into groups and send to Haiku for fast extraction.
    Each batch contains 4-5 pages to minimize API calls."""
    if not ANTHROPIC_API_KEY:
        return []

    client = _get_anthropic_client()
    pages_to_process = customer_pages[:max_pages]
    all_stories: list[CustomerStory] = []
    seen_companies: set[str] = set()

    # Build batches of 4-5 pages each
    BATCH_SIZE = 5
    batches: list[list[tuple[str, str, dict, str]]] = []
    for i in range(0, len(pages_to_process), BATCH_SIZE):
        batches.append(pages_to_process[i:i + BATCH_SIZE])

    def _extract_batch(batch: list[tuple[str, str, dict, str]]) -> list[CustomerStory]:
        try:
            # Build combined page text for the batch
            page_sections = []
            for url, text, html_signals, _raw in batch:
                signals_text = _format_html_signals(html_signals)
                section = f"--- PAGE: {url} ---\n{text[:5000]}"
                if signals_text:
                    section += f"\nHTML SIGNALS: {signals_text}"
                page_sections.append(section)

            combined = "\n\n".join(page_sections)
            prompt = BATCH_EXTRACTION_PROMPT.format(pages=combined)

            logger.info(f"Batch extracting {len(batch)} pages ({len(prompt)} chars)")
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            logger.info(f"Batch response: {raw[:200]}")

            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            import json
            items = json.loads(raw)
            stories = []
            for item in items:
                if isinstance(item, dict) and item.get("company_name"):
                    adj = item.get("adjacent_products") or []
                    if isinstance(adj, str):
                        adj = [adj]
                    stories.append(CustomerStory(
                        source_url=item.get("source_url", batch[0][0]),
                        company_name=item.get("company_name"),
                        vertical=item.get("vertical"),
                        quote=item.get("quote"),
                        quoted_person=item.get("quoted_person"),
                        quoted_title=item.get("quoted_title"),
                        outcome=item.get("outcome"),
                        evidence_type=item.get("evidence_type", "unknown"),
                        belief_shift_from=item.get("belief_shift_from"),
                        belief_shift_to=item.get("belief_shift_to"),
                        commitment_signal=item.get("commitment_signal"),
                        from_to_from=item.get("from_to_from"),
                        from_to_to=item.get("from_to_to"),
                        adjacent_products=[p for p in adj if p],
                    ))
            logger.info(f"Batch extracted {len(stories)} stories")
            return stories
        except Exception as e:
            logger.error(f"Batch extraction failed: {type(e).__name__}: {e}")
            return []

    # Run batches concurrently (typically 3-5 batches)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(_extract_batch, batch) for batch in batches]
        for f in futures:
            for story in f.result():
                key = (story.company_name or "").lower().strip()
                if key and key in seen_companies:
                    continue
                if key:
                    seen_companies.add(key)
                # Score each story on the superconsumer spectrum
                score_superconsumer(story)
                all_stories.append(story)

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

    # Buyer persona aggregation -- who are the champions by role?
    ROLE_PATTERNS = [
        ("C-Suite", ["ceo", "cto", "coo", "cfo", "cmo", "cro", "chief"]),
        ("VP", ["vp ", "vice president"]),
        ("SVP/EVP", ["svp", "evp", "senior vice"]),
        ("Director", ["director"]),
        ("Head of", ["head of"]),
        ("Manager", ["manager"]),
        ("Founder", ["founder", "co-founder"]),
        ("Principal/Lead", ["principal", "lead", "sr.", "senior"]),
    ]
    role_counts: dict[str, list[str]] = {}
    for story in stories:
        title = (story.quoted_title or "").lower()
        if not title:
            continue
        matched = False
        for pattern_name, keywords in ROLE_PATTERNS:
            if any(kw in title for kw in keywords):
                example = f"{story.quoted_person or '?'}, {story.quoted_title} at {story.company_name or '?'}"
                role_counts.setdefault(pattern_name, []).append(example)
                matched = True
                break
        if not matched and story.quoted_title:
            example = f"{story.quoted_person or '?'}, {story.quoted_title} at {story.company_name or '?'}"
            role_counts.setdefault("Other", []).append(example)

    buyer_personas = []
    for role, examples in sorted(role_counts.items(), key=lambda x: -len(x[1])):
        buyer_personas.append(BuyerPersona(
            role_pattern=role,
            count=len(examples),
            examples=examples[:5],
        ))

    strongest_persona = buyer_personas[0].role_pattern if buyer_personas else None

    # Superconsumer scoring -- separate real superconsumers from regular customers
    superconsumers = [s for s in stories if s.is_superconsumer]
    regulars = [s for s in stories if not s.is_superconsumer]
    sc_count = len(superconsumers)
    reg_count = len(regulars)
    sc_names = list({s.company_name for s in superconsumers if s.company_name})[:10]
    sc_ratio = f"{sc_count} of {len(stories)}" if stories else "0 of 0"

    # Verdict -- now grounded in the superconsumer/customer distinction
    if not stories:
        verdict = (
            "No public customer stories detected. Either this company is pre-traction, "
            "or they are hiding their best proof."
        )
    elif sc_count == 0:
        verdict = (
            f"{len(stories)} customer stories found, but none of them show superconsumer signals. "
            "No belief shifts, no identity-level commitment, no worldview change. "
            "These are satisfied customers, not missionaries. The company has users who like the product "
            "but nobody whose identity has shifted because of it. That is the gap."
        )
    elif sc_count <= 2 and len(stories) >= 5:
        verdict = (
            f"Only {sc_ratio} customer stories show real superconsumer signals "
            f"({', '.join(sc_names[:3])}). "
            "The rest are metrics and logos. That means the company has a few believers "
            "buried in a wall of satisfied customers. The superconsumers exist but the company "
            "is not amplifying them. They are treating their best proof the same as their weakest."
        )
    elif sc_count >= 3 and strongest:
        verdict = (
            f"{sc_ratio} stories show superconsumer signals. "
            f"The strongest cluster is {strongest} ({clusters[0].count} stories). "
            f"Superconsumers include {', '.join(sc_names[:3])}. "
            "These are people whose worldview shifted. The question is whether the company "
            "is building its category around what these superconsumers believe, or treating them "
            "like any other case study."
        )
    else:
        verdict = (
            f"{sc_ratio} stories show superconsumer signals. "
            "There are believers, but not enough clustering to call it a movement yet."
        )

    # Append buyer persona insight to verdict if we have role data
    if strongest_persona and buyer_personas[0].count >= 2:
        persona_note = (
            f" The champion pattern is {strongest_persona}-level buyers "
            f"({buyer_personas[0].count} of them)."
        )
        if len(buyer_personas) >= 2:
            persona_note += f" Second most common: {buyer_personas[1].role_pattern}."
        verdict += persona_note

    return SuperconsumerReport(
        total_stories_found=len(stories),
        superconsumer_count=sc_count,
        regular_customer_count=reg_count,
        superconsumer_ratio=sc_ratio,
        superconsumer_names=sc_names,
        verticals=clusters,
        buyer_personas=buyer_personas,
        language_echo_rate=round(echo_rate, 3),
        missionary_signals_in_customers=missionary_in_customers,
        strongest_vertical=strongest,
        strongest_buyer_persona=strongest_persona,
        superconsumer_verdict=verdict,
        raw_stories=stories,
    )


def build_category_analysis(stories: list[CustomerStory]) -> CategoryDesignAnalysis:
    """Aggregate per-story category design fields into synthesized analysis."""

    # --- Belief shifts ---
    shift_map: dict[tuple[str, str], list[str]] = {}
    for s in stories:
        if s.belief_shift_from and s.belief_shift_to:
            key = (s.belief_shift_from.lower().strip(), s.belief_shift_to.lower().strip())
            shift_map.setdefault(key, []).append(s.company_name or "unknown")
    belief_shifts = []
    for (frm, to), companies in sorted(shift_map.items(), key=lambda x: -len(x[1])):
        belief_shifts.append(BeliefShift(
            from_belief=frm,
            to_belief=to,
            frequency=len(companies),
            example_companies=list(set(companies))[:5],
        ))

    # --- Commitment signals ---
    commitment_signals = []
    for s in stories:
        if s.commitment_signal:
            commitment_signals.append({
                "signal": s.commitment_signal,
                "company": s.company_name or "unknown",
                "person": s.quoted_person or "unknown",
            })

    # --- FROM/TO narratives ---
    ft_map: dict[tuple[str, str], list[str]] = {}
    for s in stories:
        if s.from_to_from and s.from_to_to:
            key = (s.from_to_from.lower().strip(), s.from_to_to.lower().strip())
            ft_map.setdefault(key, []).append(s.company_name or "unknown")
    from_to_narratives = []
    for (frm, to), companies in sorted(ft_map.items(), key=lambda x: -len(x[1])):
        from_to_narratives.append(FromToNarrative(
            from_world=frm,
            to_world=to,
            frequency=len(companies),
            example_companies=list(set(companies))[:5],
        ))

    # --- Adjacent categories ---
    adj_map: dict[str, list[str]] = {}
    for s in stories:
        for prod in s.adjacent_products:
            adj_map.setdefault(prod.lower().strip(), []).append(s.company_name or "unknown")
    adjacent_categories = []
    for prod, mentioned_by in sorted(adj_map.items(), key=lambda x: -len(x[1])):
        if len(mentioned_by) >= 1:  # include even single mentions
            adjacent_categories.append({
                "product": prod,
                "mentioned_by": list(set(mentioned_by))[:5],
                "count": len(mentioned_by),
            })

    return CategoryDesignAnalysis(
        belief_shifts=belief_shifts[:10],
        commitment_signals=commitment_signals[:10],
        from_to_narratives=from_to_narratives[:10],
        adjacent_categories=adjacent_categories[:10],
    )


# ---------------------------------------------------------------------------
# Ideal Superconsumer Profile Generator
# ---------------------------------------------------------------------------

IDEAL_SUPERCONSUMER_PROMPT = """You are a category design strategist. Given a company's category and homepage positioning, describe what an IDEAL superconsumer of this category would look like.

Company: {company_name}
Category/vertical: {category}
Homepage H1: {h1}
Homepage subhead: {subhead}
What the company says it does: {description}

Answer in JSON:
{{
  "ideal_belief_before": "what someone in this category used to believe before becoming a superconsumer (conversational, under 20 words)",
  "ideal_belief_after": "what a superconsumer now believes (conversational, under 20 words)",
  "ideal_commitment_language": "what a true superconsumer would say about their identity shift (a single sentence, sounds like a real person talking)",
  "ideal_from_to": "one-sentence FROM/TO: 'From [old truth] to [new truth]' (conversational, not corporate)",
  "what_makes_them_super": "in 1-2 sentences, what separates a superconsumer from a satisfied customer in this category"
}}

CRITICAL: Write like a smart founder talking, not a consultant presenting. No jargon. No "strategically leveraged." If it sounds like a slide deck, rewrite it.

Return ONLY JSON, no markdown."""


def generate_superconsumer_gap(
    stories: list[CustomerStory],
    extracted: Extracted,
    strongest_vertical: str | None = None,
) -> SuperconsumerGap:
    """Generate ideal superconsumer profile and compare against actual evidence."""
    # Infer category from strongest vertical or homepage
    category = strongest_vertical or "B2B SaaS"
    company_name = extracted.og_title or extracted.title or "this company"
    h1 = extracted.h1 or ""
    subhead = extracted.hero_subhead or ""
    description = extracted.meta_description or ""

    ideal = IdealSuperconsumer()

    if ANTHROPIC_API_KEY:
        try:
            client = _get_anthropic_client()
            prompt = IDEAL_SUPERCONSUMER_PROMPT.format(
                company_name=company_name,
                category=category,
                h1=h1,
                subhead=subhead,
                description=description,
            )
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            import json
            data = json.loads(raw)
            ideal = IdealSuperconsumer(**{k: v for k, v in data.items() if k in IdealSuperconsumer.model_fields})
        except Exception as e:
            logger.error(f"Ideal superconsumer generation failed: {e}")

    # Count actual superconsumers
    sc_count = sum(1 for s in stories if s.is_superconsumer)
    total = len(stories)

    # Build gap verdict
    if total == 0:
        gap_verdict = (
            "No public customer stories found. Cannot compare against the ideal. "
            "But the absence itself is a signal: the company either has no superconsumers, "
            "or is not putting them on stage."
        )
    elif sc_count == 0:
        gap_verdict = (
            f"A superconsumer of this category would believe: '{ideal.ideal_belief_after}' "
            f"None of the {total} customer stories on this site show that belief. "
            "Every story is about outcomes, not worldview. The company has customers, not believers."
        )
    elif sc_count <= 2 and total >= 5:
        sc_names = [s.company_name for s in stories if s.is_superconsumer and s.company_name][:3]
        gap_verdict = (
            f"A superconsumer of this category would say: '{ideal.ideal_commitment_language}' "
            f"Only {sc_count} of {total} stories come close ({', '.join(sc_names)}). "
            "The believers exist but the company treats them the same as every other logo on the wall."
        )
    else:
        sc_names = [s.company_name for s in stories if s.is_superconsumer and s.company_name][:3]
        gap_verdict = (
            f"The ideal superconsumer belief: '{ideal.ideal_belief_after}' "
            f"{sc_count} of {total} stories show this shift ({', '.join(sc_names)}). "
            "The company has real superconsumers. The question is whether the homepage and positioning "
            "are built around what these people believe, or still hedging with generic language."
        )

    return SuperconsumerGap(
        ideal_profile=ideal,
        actual_superconsumer_count=sc_count,
        actual_total_stories=total,
        gap_verdict=gap_verdict,
    )


# ---------------------------------------------------------------------------
# Background deep crawl runner
# ---------------------------------------------------------------------------

def _run_deep_crawl(job_id: str, url: str, shallow_result: AnalyzeResponse) -> None:
    """Runs in a background thread."""
    try:
        start = time.time()
        logger.info(f"[job {job_id}] Starting deep crawl for {url}")
        all_urls, customer_pages = crawl_site(url, max_pages=100)
        logger.info(f"[job {job_id}] Crawled {len(all_urls)} pages, {len(customer_pages)} customer pages")
        # Log sample customer page URLs
        for cp_url, cp_text, _, _ in customer_pages[:5]:
            logger.info(f"[job {job_id}]   customer page: {cp_url} ({len(cp_text)} chars)")
        stories = extract_customer_stories(customer_pages)
        logger.info(f"[job {job_id}] Extracted {len(stories)} total stories")
        report = build_superconsumer_report(stories, shallow_result.extracted)
        cat_analysis = build_category_analysis(stories)
        elapsed = time.time() - start

        sc_gap = generate_superconsumer_gap(stories, shallow_result.extracted, report.strongest_vertical)
        deep = DeepAnalysis(
            crawl_meta=CrawlMeta(
                pages_crawled=len(all_urls),
                customer_pages_found=len(customer_pages),
                crawl_time_seconds=round(elapsed, 1),
            ),
            superconsumer_report=report,
            category_analysis=cat_analysis,
            superconsumer_gap=sc_gap,
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
        "version": "2.5",
        "endpoints": {
            "POST /analyze": "Positioning diagnosis. depth=shallow (default), quick (sync superconsumer scan), or deep (async full crawl).",
            "POST /analyze/homepage": "Phase 1: Fast homepage analysis (1-3s). Kicks off background customer crawl.",
            "POST /analyze/customers": "Phase 2: Customer story analysis with superconsumer gap. Uses background crawl from Phase 1 if available.",
            "GET /job/{job_id}": "Poll for deep analysis results.",
            "GET /health": "Health check.",
        },
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/scout", response_class=HTMLResponse)
def scout_landing() -> HTMLResponse:
    """Serve the Scout conversation landing page."""
    import pathlib
    html_path = pathlib.Path(__file__).parent / "scout.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>scout.html not found</h1>", status_code=404)


@app.get("/debug")
def debug(x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    """Debug endpoint to check SDK versions and Anthropic client init."""
    check_auth(x_api_key)
    import httpx as _httpx
    info: dict[str, Any] = {
        "httpx_version": _httpx.__version__,
        "anthropic_key_set": bool(ANTHROPIC_API_KEY),
        "anthropic_key_prefix": ANTHROPIC_API_KEY[:12] + "..." if ANTHROPIC_API_KEY else "not set",
    }
    try:
        import anthropic as _anth
        info["anthropic_version"] = _anth.__version__
        client = _anth.Anthropic(api_key=ANTHROPIC_API_KEY)
        info["client_init"] = "OK"
        # Quick test call
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say OK"}],
        )
        info["api_call"] = "OK"
        info["api_response"] = resp.content[0].text
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def _compact_response(resp: AnalyzeResponse) -> dict:
    """Strip raw_stories and trim category analysis for voice agent consumption.
    Reduces ~30K char responses to ~6-8K chars so the voice LLM can read them."""
    data = resp.model_dump()
    deep = data.get("deep_analysis")
    if deep:
        sr = deep.get("superconsumer_report", {})
        # Remove raw_stories entirely (16K+ chars of noise for the voice agent)
        sr.pop("raw_stories", None)
        # Trim verticals to top 5
        sr["verticals"] = sr.get("verticals", [])[:5]
        # Trim buyer personas to top 3
        sr["buyer_personas"] = sr.get("buyer_personas", [])[:3]
        # Trim category analysis entries
        ca = deep.get("category_analysis")
        if ca:
            ca["belief_shifts"] = ca.get("belief_shifts", [])[:3]
            ca["commitment_signals"] = ca.get("commitment_signals", [])[:3]
            ca["from_to_narratives"] = ca.get("from_to_narratives", [])[:3]
            ca["adjacent_categories"] = ca.get("adjacent_categories", [])[:3]
        # Keep superconsumer_gap as-is (already compact)
    return data


@app.post("/analyze/homepage")
def analyze_homepage(req: AnalyzeRequest, x_api_key: str | None = Header(default=None)):
    """Phase 1: Fast homepage analysis (1-3 seconds).
    Also kicks off background customer crawl so Phase 2 returns faster."""
    check_auth(x_api_key)
    url = normalize_url(req.url)

    # Start background customer crawl immediately (non-blocking)
    _start_bg_crawl(url)

    # Run homepage analysis synchronously (fast)
    final_url, html = fetch(url)
    extracted = extract(html, final_url)
    diagnosis = diagnose(extracted)

    result = AnalyzeResponse(
        ok=True,
        extracted=extracted,
        diagnosis=diagnosis,
        depth="homepage",
    )

    if req.compact:
        return _compact_response(result)
    return result


@app.post("/analyze/customers")
def analyze_customers(req: AnalyzeRequest, x_api_key: str | None = Header(default=None)):
    """Phase 2: Customer story analysis with superconsumer gap.
    Waits for background crawl (started by Phase 1) or runs its own if needed."""
    check_auth(x_api_key)
    url = normalize_url(req.url)

    if not ANTHROPIC_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="Customer analysis requires ANTHROPIC_API_KEY",
        )

    # Try to get background crawl result (may have been started by Phase 1)
    bg = _get_bg_crawl(url, timeout=25.0)

    # Need homepage data for superconsumer report
    final_url, html = fetch(url)
    extracted = extract(html, final_url)

    if bg and bg.status == "complete" and bg.stories is not None:
        # Background crawl finished -- use its results
        stories = bg.stories
        pages_crawled = bg.pages_crawled
        customer_pages_found = bg.customer_pages_found
        crawl_time = bg.crawl_time
    else:
        # No background result -- run crawl synchronously
        start = time.time()
        all_urls, customer_pages = crawl_priority_only(url)
        stories = extract_customer_stories(customer_pages)
        pages_crawled = len(all_urls)
        customer_pages_found = len(customer_pages)
        crawl_time = round(time.time() - start, 1)

    report = build_superconsumer_report(stories, extracted)
    cat_analysis = build_category_analysis(stories)
    sc_gap = generate_superconsumer_gap(stories, extracted, report.strongest_vertical)

    diagnosis = diagnose(extracted)
    deep = DeepAnalysis(
        crawl_meta=CrawlMeta(
            pages_crawled=pages_crawled,
            customer_pages_found=customer_pages_found,
            crawl_time_seconds=crawl_time,
        ),
        superconsumer_report=report,
        category_analysis=cat_analysis,
        superconsumer_gap=sc_gap,
    )

    result = AnalyzeResponse(
        ok=True,
        extracted=extracted,
        diagnosis=diagnosis,
        depth="customers",
        deep_analysis=deep,
    )

    # Cache the full result
    _set_cached_quick(url, result)

    if req.compact:
        return _compact_response(result)
    return result


@app.post("/analyze")
def analyze(req: AnalyzeRequest, x_api_key: str | None = Header(default=None)):
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

    if req.depth in ("quick", "deep"):
        if not ANTHROPIC_API_KEY:
            raise HTTPException(
                status_code=400,
                detail="deep/quick analysis requires ANTHROPIC_API_KEY to be configured on the server",
            )

    if req.depth == "quick":
        # Check cache first
        cached = _get_cached_quick(url)
        if cached:
            # Return cached result with fresh shallow diagnosis
            cached.extracted = extracted
            cached.diagnosis = diagnosis
            if req.compact:
                return _compact_response(cached)
            return cached

        # Synchronous fast crawl: priority pages concurrently, batched Haiku extraction
        start = time.time()
        all_urls, customer_pages = crawl_priority_only(url)
        stories = extract_customer_stories(customer_pages)
        report = build_superconsumer_report(stories, extracted)
        cat_analysis = build_category_analysis(stories)
        sc_gap = generate_superconsumer_gap(stories, extracted, report.strongest_vertical)
        elapsed = time.time() - start

        deep = DeepAnalysis(
            crawl_meta=CrawlMeta(
                pages_crawled=len(all_urls),
                customer_pages_found=len(customer_pages),
                crawl_time_seconds=round(elapsed, 1),
            ),
            superconsumer_report=report,
            category_analysis=cat_analysis,
            superconsumer_gap=sc_gap,
        )
        shallow_result.deep_analysis = deep
        shallow_result.depth = "quick"

        # Cache the full result (compact is applied at response time)
        _set_cached_quick(url, shallow_result)

        if req.compact:
            return _compact_response(shallow_result)
        return shallow_result

    if req.depth == "deep":
        # Async full crawl: 100 pages, returns job_id for polling
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


# ---------------------------------------------------------------------------
# Highlight endpoint -- serves annotated homepage HTML for iframe embedding
# ---------------------------------------------------------------------------

HIGHLIGHT_SCRIPT = """
<style>
  .ds-highlight-hype {
    background: rgba(239, 68, 68, 0.25);
    border-bottom: 2px solid #ef4444;
    padding: 1px 2px;
    border-radius: 2px;
    position: relative;
  }
  .ds-highlight-generic {
    background: rgba(249, 115, 22, 0.25);
    border-bottom: 2px solid #f97316;
    padding: 1px 2px;
    border-radius: 2px;
  }
  .ds-highlight-h1 {
    outline: 3px solid #8b5cf6 !important;
    outline-offset: 4px;
    position: relative;
  }
  .ds-highlight-h1::after {
    content: 'H1 -- this is what buyers see first';
    position: absolute;
    top: -28px;
    left: 0;
    font-size: 11px;
    font-family: monospace;
    background: #8b5cf6;
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    white-space: nowrap;
    z-index: 99999;
  }
  .ds-legend {
    position: fixed;
    top: 10px;
    right: 10px;
    z-index: 999999;
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 12px 16px;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    line-height: 1.8;
  }
  .ds-legend-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 6px;
  }
</style>
<div class="ds-legend">
  <div style="font-weight:600;margin-bottom:4px;">Doubt Scouts Audit</div>
  <div><span class="ds-legend-dot" style="background:#8b5cf6;"></span>H1 headline</div>
  <div><span class="ds-legend-dot" style="background:#ef4444;"></span>Hype words</div>
  <div><span class="ds-legend-dot" style="background:#f97316;"></span>Generic phrases</div>
</div>
<script>
(function() {
  var HYPE = %%HYPE_JSON%%;
  var GENERIC = %%GENERIC_JSON%%;

  // Highlight H1
  var h1 = document.querySelector('h1');
  if (h1) h1.classList.add('ds-highlight-h1');

  // Walk text nodes and wrap matches
  function escapeRegex(s) {
    return s.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&');
  }

  function buildPattern(phrases) {
    var sorted = phrases.slice().sort(function(a,b){ return b.length - a.length; });
    return new RegExp('\\\\b(' + sorted.map(escapeRegex).join('|') + ')\\\\b', 'gi');
  }

  function highlightTextNodes(root, pattern, className) {
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null, false);
    var nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach(function(node) {
      if (!node.nodeValue.trim()) return;
      var parent = node.parentNode;
      if (!parent || parent.tagName === 'SCRIPT' || parent.tagName === 'STYLE') return;
      if (parent.classList && (parent.classList.contains('ds-highlight-hype') || parent.classList.contains('ds-highlight-generic') || parent.classList.contains('ds-legend'))) return;
      var html = node.nodeValue;
      var replaced = html.replace(pattern, '<span class="' + className + '">$1</span>');
      if (replaced !== html) {
        var span = document.createElement('span');
        span.innerHTML = replaced;
        parent.replaceChild(span, node);
      }
    });
  }

  if (HYPE.length) highlightTextNodes(document.body, buildPattern(HYPE), 'ds-highlight-hype');
  if (GENERIC.length) highlightTextNodes(document.body, buildPattern(GENERIC), 'ds-highlight-generic');
})();
</script>
"""




@app.get("/highlight", response_class=HTMLResponse)
def highlight_homepage(
    url: str,
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
    key: str | None = None,
) -> HTMLResponse:
    """Proxy a homepage and inject highlight CSS/JS for hype words, generic phrases, and H1.
    Accepts API key via header (x-api-key) or query param (key) for iframe embedding."""
    check_auth(x_api_key or key)
    target = normalize_url(url)

    try:
        final_url, html = fetch(target)
    except Exception as e:
        return HTMLResponse(
            content=f"<html><body><h1>Could not load {target}</h1><p>{e}</p></body></html>",
            status_code=502,
        )

    import json as _json

    # Strip Content-Security-Policy meta tags so our injected highlight script can run.
    # The proxied page's CSP blocks inline scripts from foreign origins.
    html = re.sub(
        r'<meta[^>]*http-equiv\s*=\s*["\']Content-Security-Policy["\'][^>]*>',
        '', html, flags=re.IGNORECASE,
    )
    # Also strip nonce attributes from script tags (our injected script won't have one)
    html = re.sub(r'\s+nonce\s*=\s*["\'][^"\']*["\']', '', html, flags=re.IGNORECASE)

    # Rewrite relative URLs to absolute so assets load from the original domain
    parsed_target = urlparse(final_url)
    base_tag = f'<base href="{parsed_target.scheme}://{parsed_target.netloc}/">'

    # Inject base tag after <head> if present, otherwise prepend
    if "<head>" in html.lower():
        html = re.sub(r"(<head[^>]*>)", r"\1" + base_tag, html, count=1, flags=re.IGNORECASE)
    elif "<html" in html.lower():
        html = re.sub(r"(<html[^>]*>)", r"\1<head>" + base_tag + "</head>", html, count=1, flags=re.IGNORECASE)
    else:
        html = base_tag + html

    # Build the highlight script with the word lists injected
    script = HIGHLIGHT_SCRIPT.replace("%%HYPE_JSON%%", _json.dumps(HYPE_WORDS))
    script = script.replace("%%GENERIC_JSON%%", _json.dumps(GENERIC_PHRASES))

    # Inject before </body> if present, otherwise append
    if "</body>" in html.lower():
        html = re.sub(r"(</body>)", script + r"\1", html, count=1, flags=re.IGNORECASE)
    else:
        html = html + script

    return HTMLResponse(content=html)


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
