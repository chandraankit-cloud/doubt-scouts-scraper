# Doubt Scouts Positioning Scraper

A one-endpoint FastAPI service that Scout (your ElevenLabs voice agent) calls mid-conversation to scrape and diagnose any website's category positioning.

## What it does

You POST a URL. The service:
1. Fetches the HTML (static only, see JS caveat below).
2. Extracts the signals that matter for category design: title, meta description, OG tags, H1, H2s, hero subhead, CTAs, nav items, first paragraph, total word count.
3. Runs an opinionated rubric against that text: hype word hits, generic phrase hits, point-of-view strength, named problem, named enemy, languaging consistency, missionary signals.
4. Returns a compact JSON with a 0-100 score, a Scout-voice verdict paragraph, and a single-line roast.

The response is short enough for a voice agent to read aloud in under 30 seconds.

## JSON contract

### Request
```json
POST /analyze
Content-Type: application/json
x-api-key: <your-key-if-set>

{ "url": "acme.com" }
```
The URL may or may not include `https://`. The service normalizes.

### Response (abridged)
```json
{
  "ok": true,
  "extracted": {
    "final_url": "https://acme.com",
    "title": "...",
    "meta_description": "...",
    "h1": "The hero headline",
    "h2s": ["...", "..."],
    "hero_subhead": "...",
    "cta_texts": ["Get started", "Book a demo"],
    "nav_items": ["Product", "Pricing", "Customers"],
    "first_paragraph": "...",
    "word_count": 842
  },
  "diagnosis": {
    "named_problem": false,
    "named_enemy": false,
    "pov_strength": 1,
    "hype_word_hits": ["seamless", "unlock"],
    "generic_phrase_hits": ["all-in-one", "built for scale"],
    "languaging_consistency": true,
    "missionary_signals": false,
    "score": 42,
    "verdict": "The hero names a solution, not a problem...",
    "sharpest_roast": "Your homepage reads like a board deck that was asked to be a buyer story and refused."
  }
}
```

## Run locally

```bash
cd positioning_scraper
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Hit it:
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "stripe.com"}'
```

## Deploy to Render (recommended, free tier)

1. Push this `positioning_scraper/` folder to a GitHub repo.
2. Log into [render.com](https://render.com) and click **New +** then **Blueprint**.
3. Point it at the repo. Render will detect `render.yaml` and deploy.
4. Render auto-generates a value for `DOUBT_SCOUTS_API_KEY`. Copy it from the Environment tab. You will need this for the ElevenLabs tool config.
5. After the deploy finishes, grab the public URL. It looks like `https://doubt-scouts-scraper.onrender.com`.
6. Test:
   ```bash
   curl -X POST https://doubt-scouts-scraper.onrender.com/analyze \
     -H "Content-Type: application/json" \
     -H "x-api-key: YOUR-KEY" \
     -d '{"url": "stripe.com"}'
   ```

Fact: Render's free tier sleeps after 15 minutes of inactivity. The first call after a sleep takes 30-60 seconds to wake. For a prospect-facing voice agent, this is a problem. Fix: either upgrade to the $7/month starter tier, or set up a cron ping every 10 minutes to keep the service warm. Easiest cron ping is [cron-job.org](https://cron-job.org) hitting `/health`.

## Alternate hosts

- **Fly.io:** `fly launch` in this folder, it will detect the Dockerfile. Always-on free tier is more generous than Render.
- **Railway:** import from GitHub, detects the Dockerfile automatically. Paid only as of 2025.
- **ngrok (for local testing only):** run `uvicorn` locally, then `ngrok http 8000` to get a public URL you can register with ElevenLabs. Useful while iterating. Do not use for production.

## Register as an ElevenLabs agent tool

1. Go to your Scout agent in the ElevenLabs dashboard.
2. Find the **Tools** tab or section (inference: sometimes labeled "Server Tools," "Webhooks," or "Functions" depending on UI version).
3. Click **Add Tool** or **+ New Tool**.
4. Fill in:
   - **Name:** `analyze_positioning`
   - **Description:** `Scrape a website homepage and diagnose its category positioning. Use whenever the prospect names a URL, mentions their company website, or asks you to look at a specific homepage. Pass the bare domain or full URL in the 'url' field.`
   - **Method:** `POST`
   - **URL:** `https://doubt-scouts-scraper.onrender.com/analyze` (replace with your deployed URL)
   - **Headers:**
     - `Content-Type: application/json`
     - `x-api-key: <paste your DOUBT_SCOUTS_API_KEY>`
   - **Parameters schema** (JSON Schema for the request body):
     ```json
     {
       "type": "object",
       "properties": {
         "url": {
           "type": "string",
           "description": "The website URL to analyze. Bare domain like 'acme.com' is fine, https is optional."
         }
       },
       "required": ["url"]
     }
     ```
5. Save the tool.
6. Open the agent's System Prompt and make sure it references the tool (Scout's system prompt already does, see `voice_agent/scout_system_prompt.md`).
7. Test by saying to Scout: "Look at stripe.com and tell me what you think." Scout should call the tool and narrate the diagnosis.

Inference: ElevenLabs' exact tool config UI varies by version. The concepts above (name, description, URL, headers, parameters schema) are universal across webhook tool implementations. If a field is named slightly differently, the nearest equivalent is almost always correct.

## JavaScript-heavy sites (the caveat)

This scraper only reads static HTML. Roughly 70 percent of B2B marketing sites still serve server-rendered HTML for SEO reasons, so it works on them. The other 30 percent (React SPAs that render the hero client-side) will return an empty or near-empty body. You will see `h1: null` and a very low word count.

Two upgrade paths if you hit this often:

1. **Playwright upgrade (best fidelity):** replace the `fetch()` function with a Playwright `page.goto()` that waits for network idle. Adds 300 MB to the Docker image, 3-5 seconds per request, and requires a bigger host tier. I can write this for you if you need it.
2. **ScrapingBee / ScraperAPI / Browserless (easiest):** pipe the fetch through a commercial rendering API with one line of code. Costs roughly $0.001 to $0.005 per request. Good middle ground.

## Abuse protection

The `DOUBT_SCOUTS_API_KEY` environment variable enables a simple `x-api-key` header check. Any request missing the right key gets a 401. This is enough to keep random internet scanners from burning your free-tier minutes. It is not enough to stop a motivated attacker, so do not check the key into GitHub and do rotate it if you think it leaked.

## What this does not do (on purpose)

- Does not crawl more than one page. Homepage only. Most positioning lives there, and voice agents should not be waiting 30 seconds for a recursive crawl.
- Does not analyze pricing, feature pages, or blog posts. Out of scope for category diagnosis.
- Does not follow redirects across domains without logging. It does follow normal HTTPS redirects.
- Does not cache. Every call is a fresh fetch. If you start hitting the same URLs repeatedly, add a Redis cache layer.
