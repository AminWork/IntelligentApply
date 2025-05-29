# service-fetcher/app/main.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Scrapes recent PhD-position listings from phdfinder.com, extracts structured
metadata with Metis Chat, stores them in MongoDB, and indexes embeddings in
FAISS.

2025-05-29  • Base-64 workaround for /embeddings 500
           • Robust JSON parsing + LLM re-prompt on malformed output
           • 3-attempt exponential-back-off around embeddings
2025-05-29b • FIX: /scrape page-count bug (“<' not supported …”)
"""

import os, re, json, asyncio, logging, base64
from datetime import datetime, timedelta
from typing import List, Dict, Any
import httpx, numpy as np
from bs4 import BeautifulSoup
from dateutil.parser import parse as date_parse
from fastapi import FastAPI, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from openai import OpenAI

# ─── LOGGING ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", force=True)
logger = logging.getLogger("service-fetcher")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
METIS_API_URL = os.getenv("METIS_API_URL", "https://api.metisai.ir/api/v1")
METIS_API_KEY = os.getenv("METIS_API_KEY", "")
if not METIS_API_KEY:
    raise RuntimeError("METIS_API_KEY environment variable is required")

EMBED_MODEL   = "text-embedding-3-small"
HEADERS       = {"Authorization": f"Bearer {METIS_API_KEY}",
                 "Content-Type":  "application/json"}
MONGO_URI      = os.getenv("MONGO_URI",      "mongodb://root:example@mongo:27017")
FAISS_ENDPOINT = os.getenv("FAISS_ENDPOINT", "http://faiss-db:8080")
BASE_URL       = "https://phdfinder.com/positions/"
EMBED_DIM      = 1536

client = OpenAI(api_key=METIS_API_KEY,
                base_url="https://api.metisai.ir/openai/v1")

mongo = AsyncIOMotorClient(MONGO_URI)
db    = mongo["fetcher"]["positions"]
app   = FastAPI()

# ─── REQUEST MODEL ────────────────────────────────────────────────────────────
from pydantic import BaseModel
class PositionRequest(BaseModel):
    id:  str
    url: str

# ─── JSON HELPERS ─────────────────────────────────────────────────────────────
_JSON_RE = re.compile(r"\{.*?\}", re.S)
def extract_json(text: str) -> str:
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError("No JSON object found in LLM response")
    return m.group(0)

def safe_json(raw: str) -> Dict[str, Any]:
    """Attempt strict JSON parse, then sanitise trailing commas / single quotes."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
        cleaned = re.sub(r"'", '"', cleaned)
        return json.loads(cleaned)

async def extract_with_retry(prompt: List[Dict[str, str]],
                             provider: str = "openai_chat_completion",
                             model: str = "gpt-4o-mini",
                             max_attempts: int = 2) -> Dict[str, Any]:
    for attempt in range(1, max_attempts + 1):
        raw = await metis_chat(prompt, provider, model)
        try:
            return safe_json(extract_json(raw))
        except Exception as e:
            logger.warning("JSON parse failed (%d/%d): %s",
                           attempt, max_attempts, e, exc_info=True)
            if attempt == max_attempts:
                raise
            prompt.extend([
                {"role": "assistant", "content": raw},
                {"role": "system",
                 "content": "❗ Output was invalid JSON. "
                            "Respond with *only* valid JSON."},
            ])

# ─── LOGGING HELPERS ─────────────────────────────────────────────────────────
def log_httpx_error(prefix: str, exc: httpx.HTTPStatusError):
    resp = exc.response
    body = resp.text[:500] if resp.text else "<no body>"
    logger.error("%s – HTTP %s %s\n↳ %s",
                 prefix, resp.status_code, resp.request.url, body, exc_info=True)

# ─── METIS CHAT / EMBEDDINGS ─────────────────────────────────────────────────
async def metis_chat(messages: List[Dict[str, Any]],
                     provider: str,
                     model: str) -> str:
    url = f"{METIS_API_URL}/wrapper/{provider}/chat/completions"
    async with httpx.AsyncClient() as h:
        try:
            r = await h.post(url, headers=HEADERS,
                             json={"model": model, "messages": messages},
                             timeout=120)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            log_httpx_error("CHAT ⇦ ERROR", e); raise
    return r.json()["choices"][0]["message"]["content"]

# ─── METIS CHAT / EMBEDDINGS ─────────────────────────────────────────────────

# ─── METIS CHAT / EMBEDDINGS ─────────────────────────────────────────────────

# ─── METIS CHAT / EMBEDDINGS ─────────────────────────────────────────────────

# ─── METIS CHAT / EMBEDDINGS ─────────────────────────────────────────────────

async def embed(texts: List[str], model: str = EMBED_MODEL) -> List[List[float]]:
    """
    Call MetisAI’s embedding service via the OpenAI-compatible client,
    requesting float-encoded embeddings. Retries up to 3 times with
    exponential back-off on failure.
    """
    backoff = 1.0
    for attempt in range(1, 4):
        try:
            # mirror the JS style: encoding_format="float"
            resp = await asyncio.to_thread(
                client.embeddings.create,
                model=model,
                input=texts,
                encoding_format="float"
            )
            break
        except Exception as e:
            logger.warning("EMBED attempt %d/3 failed: %s", attempt, e, exc_info=True)
            if attempt == 3:
                raise
            await asyncio.sleep(backoff)
            backoff *= 2

    # extract the float vectors directly
    embeddings: List[List[float]] = []
    for item in getattr(resp, "data", resp.get("data", [])):
        vec = getattr(item, "embedding", item["embedding"])
        if not isinstance(vec, list) or len(vec) != EMBED_DIM:
            raise ValueError(f"Unexpected embedding format/size: got {type(vec)} len={len(vec) if isinstance(vec, list) else 'N/A'}")
        embeddings.append(vec)

    print("EMBEDDINGS:", embeddings)

    return embeddings




async def faiss_upsert(doc_id: str, vec: List[float]):
    async with httpx.AsyncClient() as h:
        try:
            r = await h.post(f"{FAISS_ENDPOINT}/add",
                             json={"id": doc_id, "vector": vec}, timeout=10)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            log_httpx_error("FAISS ⇦ ERROR", e); raise

# ─── SCRAPER UTILITIES ────────────────────────────────────────────────────────
async def fetch_html(url: str) -> str:
    async with httpx.AsyncClient(timeout=30) as h:
        try:
            r = await h.get(url); r.raise_for_status()
        except httpx.HTTPStatusError as e:
            log_httpx_error("FETCH ⇦ ERROR", e); raise
    return r.text

def get_total_pages(soup: BeautifulSoup) -> int:
    """Return largest page number visible in pagination (≥1)."""
    pages = [int(a.text) for a in soup.select(".page-numbers") if a.text.isdigit()]
    return max(pages) if pages else 1

def listings_from_page(soup: BeautifulSoup) -> List[tuple[str, str]]:
    week_ago = datetime.utcnow() - timedelta(days=7)
    out: list[tuple[str, str]] = []
    for art in soup.select("article.post"):
        dtag = art.select_one("span.published")
        if not dtag:
            continue
        try:
            published = date_parse(dtag.text)
        except Exception:
            continue
        if published < week_ago:
            continue
        link = art.select_one("h2.entry-title a")
        if link:
            out.append((link.text.strip(), link["href"]))
    return out

# ─── CORE PIPELINE ────────────────────────────────────────────────────────────
async def process_position(doc_id: str, url: str):
    logger.info("PROCESS ➜ %s", doc_id)
    try:
        text = BeautifulSoup(await fetch_html(url),
                             "html.parser").get_text(" ", strip=True)
    except Exception as e:
        logger.error("html fetch failed: %s", e, exc_info=True); return

    # 1) structured fields
    prompt = [
        {"role": "system",
         "content": ("Extract as JSON: position_title, university_name, department_faculty, "
                     "location_country, application_deadline, contact_person, "
                     "contact_email, summary.")},
        {"role": "user", "content": text}]
    try:
        fields = await extract_with_retry(prompt)
        logger.info("FIELDS ⇦ %s", fields)
    except Exception:
        return

    # 2) keywords
    kw_prompt = [
        {"role": "system",
         "content": "Give 5-10 keywords as JSON array under key 'keywords'."},
        {"role": "user", "content": fields.get("summary", "")}]
    try:
        keywords = (await extract_with_retry(kw_prompt)).get("keywords", [])
        logger.info("KEYWORDS ⇦ %s", keywords)
    except Exception:
        return

    # 3) embedding
    try:
        vecs   = await embed(keywords)
        avgvec = np.mean(np.array(vecs, dtype=np.float32), axis=0).tolist()
    except Exception:
        return

    # 4) Mongo
    rec = {**fields, "keywords": keywords, "embedding": avgvec,
           "url": url, "_id": doc_id}
    try:
        await db.replace_one({"_id": doc_id}, rec, upsert=True)
    except Exception as e:
        logger.error("mongo upsert failed: %s", e, exc_info=True)

    # 5) FAISS
    try:
        await faiss_upsert(doc_id, avgvec)
    except Exception:
        pass

# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.post("/fetch")
async def fetch_single(req: PositionRequest):
    await process_position(req.id, req.url)
    return {"status": "ok", "id": req.id}

@app.get("/scrape")
async def scrape_all():
    """Scrape up to 10 newest listings (≤2 pages) asynchronously."""
    try:
        soup0        = BeautifulSoup(await fetch_html(BASE_URL), "html.parser")
        total_pages  = min(get_total_pages(soup0), 2)   # ← fixed
    except Exception as e:
        raise HTTPException(502, detail=f"listing page fetch failed: {e}")

    listings: list[tuple[str, str]] = []
    for p in range(1, total_pages + 1):
        page_url = BASE_URL if p == 1 else f"{BASE_URL}page/{p}/"
        listings.extend(
            listings_from_page(
                BeautifulSoup(await fetch_html(page_url), "html.parser")))
        await asyncio.sleep(0.5)

    listings = listings[:10]
    await asyncio.gather(
        *[process_position(str(abs(hash(u)) % (1 << 63)), u) for _, u in listings])
    return {"status": "scraped", "count": len(listings)}

@app.get("/ping")
async def ping():
    return {"ping": "pong"}
