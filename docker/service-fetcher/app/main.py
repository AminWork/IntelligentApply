# service-fetcher/app/main.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Scrapes recent PhD-position listings from phdfinder.com, extracts structured
metadata with Metis Chat, stores them in MongoDB, and indexes embeddings in
FAISS.

2025-05-30  • Add Mongo connection diagnostics & better upsert logging
           • Fail-fast on startup if Mongo unreachable or auth fails
           • Propagate upstream errors instead of silent early returns
           • Fixed MongoDB writing issues
"""

import os, re, json, asyncio, logging, base64
from datetime import datetime, timedelta
from typing import List, Dict, Any

import httpx, numpy as np
from bs4 import BeautifulSoup
from dateutil.parser import parse as date_parse
from fastapi import FastAPI, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from openai import OpenAI
from pydantic import BaseModel

# ─── LOGGING ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger("service-fetcher")

# ─── CONFIG ───────────────────────────────────────────────────────────────────

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

METIS_API_URL = os.getenv("METIS_API_URL", "https://api.metisai.ir/api/v1")
METIS_API_KEY = os.getenv("METIS_API_KEY")
MONGO_URI     = os.getenv("MONGO_URI")
FAISS_ENDPOINT = os.getenv("FAISS_ENDPOINT", "http://faiss-db:8080")
BASE_URL       = "https://phdfinder.com/positions/"
# Check what dimension your FAISS index actually uses
# Common options:
# EMBED_MODEL = "text-embedding-ada-002"     # 1536 dims
# EMBED_MODEL = "text-embedding-3-small"     # 1536 dims  
# EMBED_MODEL = "text-embedding-3-large"     # 3072 dims

EMBED_MODEL    = "text-embedding-3-small"
EMBED_DIM      = 1536  # Make sure this matches your FAISS index dimension!

if not METIS_API_KEY:
    raise RuntimeError("METIS_API_KEY environment variable is required")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI environment variable is required")

HEADERS = {
    "Authorization": f"Bearer {METIS_API_KEY}",
    "Content-Type":  "application/json",
}

# ─── EXTERNAL CLIENTS ─────────────────────────────────────────────────────────
client = OpenAI(api_key=METIS_API_KEY,
                base_url="https://api.metisai.ir/openai/v1")

mongo_client = AsyncIOMotorClient(
    MONGO_URI,
    serverSelectionTimeoutMS=10_000,   # Increased timeout
    connectTimeoutMS=10_000,
    socketTimeoutMS=10_000,
)
database       = mongo_client["fetcher"]
positions_col: AsyncIOMotorCollection = database["positions"]

app = FastAPI()

# ─── STARTUP CHECKS ───────────────────────────────────────────────────────────
@app.on_event("startup")
async def verify_mongo_connection() -> None:
    try:
        # Test both connection and write permissions
        server_info = await mongo_client.server_info()
        logger.info("✅ Mongo connected: %s", server_info.get("version"))
        
        # Test database and collection access
        db_stats = await database.command("dbStats")
        logger.info("   Database stats: %s", db_stats.get("db"))
        
        # Test write operation with a simple document
        test_doc = {"_id": "startup_test", "timestamp": datetime.utcnow()}
        await positions_col.replace_one({"_id": "startup_test"}, test_doc, upsert=True)
        await positions_col.delete_one({"_id": "startup_test"})
        
        logger.info("   Writing to db='%s', collection='%s' ✅",
                    database.name, positions_col.name)
    except Exception as e:
        logger.exception("❌ Mongo connection/write test failed: %s", e)
        # Hard-fail: there is no point starting if we cannot persist data
        raise RuntimeError("MongoDB unavailable or not writable; aborting startup") from e

# ─── REQUEST MODEL ────────────────────────────────────────────────────────────
class PositionRequest(BaseModel):
    id:  str
    url: str

# ─── JSON HELPERS ─────────────────────────────────────────────────────────────
_JSON_RE = re.compile(r"\{.*?\}", re.S)

def extract_json(text: str) -> str:
    try:
        # Try parsing as JSON first
        json.loads(text)
        return text
    except json.JSONDecodeError:
        # Fallback to regex extraction
        m = _JSON_RE.search(text)
        if not m:
            raise ValueError(f"No JSON object found in:\n{text[:500]}")
        return m.group(0)

def safe_json(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("JSON decode error: %s\nRaw content:\n%s", e, raw[:1000])
        # Try fixing common issues
        for pattern, replacement in [
            (r",\s*([}\]])", r"\1"),  # Trailing commas
            (r"'\s*:\s*'", r'":"'),   # Single quotes
            (r"\\'", r"'")             # Escaped quotes
        ]:
            raw = re.sub(pattern, replacement, raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.exception("Final JSON parse failed after sanitization")
            raise

async def extract_with_retry(
    prompt: List[Dict[str, str]],
    provider: str = "openai_chat_completion",
    model: str = "gpt-4o-mini",
    max_attempts: int = 2,
) -> Dict[str, Any]:
    raw = ""  # Initialize to avoid UnboundLocalError
    for attempt in range(1, max_attempts + 1):
        try:
            raw = await metis_chat(prompt, provider, model)
            logger.debug("LLM raw response (attempt %d):\n%s", attempt, raw)
            return safe_json(extract_json(raw))
        except Exception as e:
            logger.error("JSON extraction failed (%d/%d): %s\nRaw content:\n%s", 
                         attempt, max_attempts, e, raw[:2000])
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Failed to extract valid JSON after {max_attempts} attempts"
                ) from e
            # Add correction instruction
            prompt.append({
                "role": "system",
                "content": "ERROR: Output was invalid JSON. Respond with ONLY valid JSON without any additional text."
            })

# ─── LOGGING HELPERS ─────────────────────────────────────────────────────────
def log_httpx_error(prefix: str, exc: httpx.HTTPStatusError) -> None:
    resp = exc.response
    body = resp.text[:500] if resp.text else "<no body>"
    logger.error("%s – HTTP %s %s\n↳ %s",
                 prefix, resp.status_code, resp.request.url, body, exc_info=True)

# ─── METIS CHAT / EMBEDDINGS ─────────────────────────────────────────────────
async def metis_chat(
    messages: List[Dict[str, Any]],
    provider: str,
    model: str,
) -> str:
    url = f"{METIS_API_URL}/wrapper/{provider}/chat/completions"
    async with httpx.AsyncClient() as h:
        try:
            r = await h.post(url, headers=HEADERS,
                             json={"model": model, "messages": messages},
                             timeout=120)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            log_httpx_error("CHAT ⇦ ERROR", e)
            raise
    return r.json()["choices"][0]["message"]["content"]

async def embed(texts: List[str], model: str = EMBED_MODEL) -> List[List[float]]:
    """
    Call MetisAI's embedding service via the OpenAI-compatible client.
    Retries up to 3 times with exponential back-off.
    """
    backoff = 1.0
    for attempt in range(1, 4):
        try:
            resp = await asyncio.to_thread(
                client.embeddings.create,
                model=model,
                input=texts,
                encoding_format="float",
            )
            break
        except Exception as e:
            logger.warning("EMBED attempt %d/3 failed: %s", attempt, e, exc_info=True)
            if attempt == 3:
                raise
            await asyncio.sleep(backoff)
            backoff *= 2

    # --- fixed section ------------------------------------------------------
    if hasattr(resp, "data"):                 # openai-python ≥1.14 response object
        items = resp.data
    elif isinstance(resp, dict):              # raw JSON (older client / custom call)
        items = resp.get("data", [])
    else:
        raise TypeError(f"Unexpected response type from embeddings API: {type(resp)}")
    # -----------------------------------------------------------------------

    embeddings: List[List[float]] = []
    for item in items:
        # Handle both object attributes and dictionary access
        if hasattr(item, "embedding"):
            vec = item.embedding
        elif isinstance(item, dict) and "embedding" in item:
            vec = item["embedding"]
        else:
            raise ValueError(f"Cannot extract embedding from item: {type(item)}")
            
        if not isinstance(vec, list) or len(vec) != EMBED_DIM:
            raise ValueError(
                f"Unexpected embedding format/size: got {type(vec)} "
                f"len={len(vec) if isinstance(vec, list) else 'N/A'}"
            )
        embeddings.append(vec)

    return embeddings

async def faiss_upsert(doc_id: str, vec: List[float], metadata: Dict[str, Any] = None) -> None:
    payload = {
        "id": doc_id, 
        "vector": vec
    }
    if metadata:
        payload["metadata"] = metadata
        
    async with httpx.AsyncClient() as h:
        try:
            r = await h.post(f"{FAISS_ENDPOINT}/add",
                             json=payload, timeout=10)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            log_httpx_error("FAISS ⇦ ERROR", e)
            raise

# ─── SCRAPER UTILITIES ────────────────────────────────────────────────────────
async def fetch_html(url: str) -> str:
    backoff = 1.0
    for attempt in range(1, 4):  # Retry up to 3 times
        async with httpx.AsyncClient(
            headers={"User-Agent": BROWSER_UA},
            timeout=httpx.Timeout(30.0)
        ) as h:
            try:
                r = await h.get(url, follow_redirects=True)
                r.raise_for_status()
                return r.text
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                if attempt == 3:
                    logger.exception("Final fetch attempt failed for %s", url)
                    raise HTTPException(
                        502, 
                        detail=f"Network error after 3 attempts: {e}"
                    ) from e
                logger.warning("Fetch attempt %d/3 failed: %s", attempt, e)
                await asyncio.sleep(backoff)
                backoff *= 2  # Exponential backoff

def get_total_pages(soup: BeautifulSoup) -> int:
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
async def process_position(doc_id: str, url: str) -> None:
    logger.info("PROCESS ➜ %s", doc_id)

    try:
        # 0) Pull raw page
        logger.debug("Fetching HTML from %s", url)
        html_content = await fetch_html(url)
        text = BeautifulSoup(html_content, "html.parser").get_text(" ", strip=True)
        logger.debug("Extracted text length: %d chars", len(text))

        # 1) Extract structured fields
        logger.debug("Extracting structured fields...")
        prompt = [
            {"role": "system",
             "content": ("Extract as JSON: position_title, university_name, "
                         "department_faculty, location_country, application_deadline, "
                         "contact_person, contact_email, summary.")},
            {"role": "user", "content": text},
        ]
        fields = await extract_with_retry(prompt)
        logger.info("FIELDS ⇦ %s", fields)

        # 2) Extract keywords
        logger.debug("Extracting keywords...")
        kw_prompt = [
            {"role": "system",
             "content": "Give 5-10 keywords as JSON array under key 'keywords'."},
            {"role": "user", "content": fields.get("summary", "")},
        ]
        keywords = (await extract_with_retry(kw_prompt)).get("keywords", [])
        logger.info("KEYWORDS ⇦ %s", keywords)

        # 3) Generate embeddings
        logger.debug("Generating embeddings...")
        if keywords:  # Only if we have keywords
            vecs = await embed(keywords)
            avgvec = np.mean(np.array(vecs, dtype=np.float32), axis=0).tolist()
        else:
            # Fallback: create embedding from summary if no keywords
            summary_text = fields.get("summary", "No summary available")
            vecs = await embed([summary_text])
            avgvec = vecs[0]

        # 4) Prepare and insert into MongoDB
        logger.debug("Preparing MongoDB document...")
        rec = {
            **fields,
            "keywords": keywords,
            "embedding": avgvec,
            "url": url,
            "_id": doc_id,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        # Log the document before insertion (without embedding for brevity)
        log_rec = {k: v for k, v in rec.items() if k != "embedding"}
        logger.debug("Document to insert: %s", log_rec)

        # Perform the MongoDB operation with better error handling
        logger.debug("Inserting into MongoDB...")
        try:
            result = await positions_col.replace_one(
                {"_id": doc_id}, 
                rec, 
                upsert=True
            )
            logger.info("✅ Mongo upsert success: matched=%d modified=%d upserted_id=%s",
                        result.matched_count, result.modified_count, result.upserted_id)
            
            # Verify the document was actually written
            verify_doc = await positions_col.find_one({"_id": doc_id})
            if verify_doc:
                logger.info("✅ Document verification successful for %s", doc_id)
            else:
                logger.error("❌ Document verification failed - not found after insert!")
                
        except Exception as mongo_error:
            logger.exception("❌ MongoDB operation failed: %s", mongo_error)
            raise

        # 5) FAISS indexing (non-critical, so continue on failure)
        logger.debug("Adding to FAISS index...")
        try:
            # Verify vector dimension before sending to FAISS
            if len(avgvec) != EMBED_DIM:
                logger.warning("Vector dimension mismatch: got %d, expected %d. Skipping FAISS.", 
                              len(avgvec), EMBED_DIM)
            else:
                # Send meaningful metadata to FAISS
                faiss_metadata = {
                    "title": fields.get("position_title", "")[:100],  # Truncate for storage
                    "university": fields.get("university_name", "")[:100],
                    "location": fields.get("location_country", "")[:50],
                    "keywords": keywords[:5],  # First 5 keywords only
                    "url": url,
                    "added_at": datetime.utcnow().isoformat()
                }
                await faiss_upsert(doc_id, avgvec, faiss_metadata)
                logger.debug("✅ FAISS upsert successful")
        except Exception as faiss_error:
            logger.exception("❌ FAISS upsert failed (continuing): %s", faiss_error)

        logger.info("✅ PROCESS COMPLETE ➜ %s", doc_id)

    except Exception as e:
        logger.exception("❌ PROCESS FAILED ➜ %s: %s", doc_id, e)
        raise

# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.post("/fetch")
async def fetch_single(req: PositionRequest):
    try:
        await process_position(req.id, req.url)
        return {"status": "ok", "id": req.id}
    except Exception as e:
        logger.exception("Fetch single failed for %s: %s", req.id, e)
        raise HTTPException(500, detail=str(e)) from e

@app.get("/scrape")
async def scrape_all():
    """Scrape up to 10 newest listings (≤2 pages) asynchronously."""
    try:
        logger.info("Starting scrape operation...")
        soup0 = BeautifulSoup(await fetch_html(BASE_URL), "html.parser")
        total_pages = min(get_total_pages(soup0), 2)
        logger.info("Will scrape %d pages", total_pages)
    except Exception as e:
        logger.exception("Listing page fetch failed: %s", e)
        raise HTTPException(502, detail=f"listing page fetch failed: {e}")

    listings: list[tuple[str, str]] = []
    for p in range(1, total_pages + 1):
        page_url = BASE_URL if p == 1 else f"{BASE_URL}page/{p}/"
        logger.info("Scraping page %d: %s", p, page_url)
        page_listings = listings_from_page(
            BeautifulSoup(await fetch_html(page_url), "html.parser"))
        listings.extend(page_listings)
        logger.info("Found %d listings on page %d", len(page_listings), p)
        await asyncio.sleep(0.5)

    listings = listings[:10]
    logger.info("Processing %d total listings", len(listings))
    
    # Process listings with better error handling
    successful = 0
    failed = 0
    
    for title, url in listings:
        doc_id = str(abs(hash(url)) % (1 << 63))
        try:
            await process_position(doc_id, url)
            successful += 1
            logger.info("✅ Successfully processed: %s", title[:50])
        except Exception as e:
            failed += 1
            logger.error("❌ Failed to process '%s': %s", title[:50], e)
    
    logger.info("Scrape complete: %d successful, %d failed", successful, failed)
    return {
        "status": "scraped", 
        "total_listings": len(listings),
        "successful": successful,
        "failed": failed
    }

@app.get("/ping")
async def ping():
    return {"ping": "pong"}

@app.get("/health")
async def health_check():
    """Health check endpoint that verifies MongoDB connectivity"""
    try:
        # Test MongoDB connection
        server_info = await mongo_client.server_info()
        mongo_status = "connected"
        
        # Test collection access
        count = await positions_col.count_documents({})
        
        return {
            "status": "healthy",
            "mongodb": {
                "status": mongo_status,
                "version": server_info.get("version"),
                "document_count": count
            },
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.exception("Health check failed: %s", e)
        raise HTTPException(503, detail=f"Service unhealthy: {e}")