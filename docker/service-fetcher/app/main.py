# service-fetcher/app/main.py

import os
import json
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from bs4 import BeautifulSoup
import httpx
import numpy as np
from datetime import datetime, timedelta
from dateutil.parser import parse as date_parse

# ─── LOGGING CONFIG ───────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("service-fetcher")

# ─── CONFIGURATION ───────────────────────────────────────────────────────────

METIS_API_URL = os.getenv("METIS_API_URL", "https://api.metisai.ir/api/v1")
METIS_API_KEY = os.getenv("METIS_API_KEY", "")
if not METIS_API_KEY:
    logger.error("METIS_API_KEY environment variable is required")
    raise RuntimeError("METIS_API_KEY environment variable is required")

HEADERS = {
    "Authorization": f"Bearer {METIS_API_KEY}",
    "Content-Type": "application/json"
}

MONGO_URI      = os.getenv("MONGO_URI",      "mongodb://root:example@mongo:27017")
FAISS_ENDPOINT = os.getenv("FAISS_ENDPOINT", "http://faiss-db:8080")
BASE_URL       = "https://phdfinder.com/positions/"
EMBED_DIM      = int(os.getenv("EMBEDDING_DIM", "1536"))

# ─── INITIALIZE DATABASES & APP ──────────────────────────────────────────────

mongo = AsyncIOMotorClient(MONGO_URI)
db    = mongo["fetcher"]["positions"]

app = FastAPI()
logger.info("Starting service-fetcher with MongoDB at %s and FAISS at %s",
            MONGO_URI, FAISS_ENDPOINT)

# ─── REQUEST/RESPONSE MODELS ─────────────────────────────────────────────────

class PositionRequest(BaseModel):
    id:  str
    url: str

# ─── METIS HELPER FUNCTIONS ──────────────────────────────────────────────────

async def metis_chat_json(messages: list[dict], provider: str, model: str, schema: dict) -> dict:
    """
    Call MetisAI chat/completions in JSON mode to extract structured data.
    """
    logger.debug("Calling Metis chat: provider=%s model=%s schema=%s",
                 provider, model, schema)
    payload = {
        "chatProvider": provider,
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "json_schema": schema
    }
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(f"{METIS_API_URL}/chat/completions",
                                  headers=HEADERS, json=payload, timeout=120)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            logger.debug("Metis chat response: %s", content)
            return content
        except Exception as e:
            logger.exception("Metis chat/completions failed")
            raise

async def metis_embed(inputs: list[str], provider: str, model: str) -> list[list[float]]:
    """
    Call MetisAI embeddings endpoint to get vectors for each input string.
    """
    logger.debug("Calling Metis embeddings: provider=%s model=%s inputs=%s",
                 provider, model, inputs)
    payload = {
        "embeddingProvider": provider,
        "model": model,
        "input": inputs
    }
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(f"{METIS_API_URL}/embeddings",
                                  headers=HEADERS, json=payload, timeout=60)
            r.raise_for_status()
            data = [item["embedding"] for item in r.json()["data"]]
            logger.debug("Metis embeddings response length: %d", len(data))
            return data
        except Exception as e:
            logger.exception("Metis embeddings failed")
            raise

async def faiss_upsert(doc_id: str, vector: list[float]):
    """
    Upsert a single vector into the FAISS microservice.
    """
    logger.debug("Upserting to FAISS id=%s vector_dim=%d", doc_id, len(vector))
    async with httpx.AsyncClient() as client:
        try:
            await client.post(f"{FAISS_ENDPOINT}/add",
                              json={"id": doc_id, "vector": vector},
                              timeout=10)
            logger.info("FAISS upsert successful for id=%s", doc_id)
        except Exception:
            logger.exception("FAISS upsert failed for id=%s", doc_id)
            raise

# ─── SCRAPING HELPERS ────────────────────────────────────────────────────────

async def fetch_html(url: str) -> str:
    logger.info("Fetching URL: %s", url)
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            logger.debug("Fetched %d bytes from %s", len(resp.text), url)
            return resp.text
        except Exception:
            logger.exception("Failed to fetch URL: %s", url)
            raise

def get_total_pages(soup: BeautifulSoup) -> int:
    pages = [int(a.text) for a in soup.select(".page-numbers") if a.text.isdigit()]
    total = max(pages) if pages else 1
    logger.info("Total pages found: %d", total)
    return total

def listings_from_page(soup: BeautifulSoup) -> list[tuple[str,str]]:
    one_week_ago = datetime.utcnow() - timedelta(days=7)
    results = []
    for art in soup.select("article.post"):
        date_tag = art.select_one("span.published")
        if not date_tag:
            continue
        try:
            post_date = date_parse(date_tag.text)
        except Exception:
            continue
        if post_date < one_week_ago:
            continue
        link = art.select_one("h2.entry-title a")
        if link:
            title, href = link.text.strip(), link["href"]
            results.append((title, href))
    logger.info("Filtered %d listings from page", len(results))
    return results

# ─── CORE PROCESSOR ──────────────────────────────────────────────────────────

async def process_position(doc_id: str, url: str):
    logger.info("Processing position id=%s url=%s", doc_id, url)
    try:
        html = await fetch_html(url)
        text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)

        # 1) Extract fields
        field_schema = {
            "type": "object",
            "properties": {
                "university_name":      {"type": "string"},
                "department_faculty":   {"type": "string"},
                "location_country":     {"type": "string"},
                "application_deadline": {"type": "string"},
                "contact_person":       {"type": "string"},
                "contact_email":        {"type": "string"},
                "summary":              {"type": "string"}
            },
            "required": ["summary"]
        }
        fields = await metis_chat_json(
            messages=[
                {"role": "system", "content":
                 "Extract the following from this PhD position text: "
                 "university_name, department_faculty, location_country, "
                 "application_deadline, contact_person, contact_email, summary."},
                {"role": "user", "content": text}
            ],
            provider="openai_chat_completion",
            model="gpt-4o",
            schema=field_schema
        )

        # 2) Generate keywords
        kw_schema = {
            "type": "object",
            "properties": {"keywords": {"type": "array","items": {"type": "string"}}},
            "required": ["keywords"]
        }
        kw_resp = await metis_chat_json(
            messages=[
                {"role": "system",
                 "content": "From this summary, list 5–10 relevant keywords in JSON."},
                {"role": "user", "content": fields["summary"]}
            ],
            provider="openai_chat_completion",
            model="gpt-4o",
            schema=kw_schema
        )
        keywords = kw_resp["keywords"]
        logger.info("Extracted %d keywords for id=%s", len(keywords), doc_id)

        # 3) Embed keywords & average
        embs = await metis_embed(inputs=keywords,
                                 provider="openai",
                                 model="text-embedding-3-small")
        avg_vec = list(np.mean(np.array(embs, dtype="float32"), axis=0))
        logger.info("Computed embedding vector for id=%s", doc_id)

        # 4) Upsert into MongoDB
        record = {**fields, "keywords": keywords,
                  "embedding": avg_vec, "url": url, "_id": doc_id}
        await db.replace_one({"_id": doc_id}, record, upsert=True)
        logger.info("Stored record in MongoDB for id=%s", doc_id)

        # 5) Upsert into FAISS
        await faiss_upsert(doc_id, avg_vec)

    except Exception as e:
        logger.error("Failed processing position id=%s: %s", doc_id, e, exc_info=True)
        # Optionally re-raise or continue
        raise

# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.post("/fetch")
async def fetch_single(req: PositionRequest):
    logger.info("POST /fetch %s", req.json())
    await process_position(req.id, req.url)
    return {"status": "ok", "id": req.id}

@app.get("/scrape")
async def scrape_all():
    logger.info("GET /scrape start")
    html0 = await fetch_html(BASE_URL)
    soup0 = BeautifulSoup(html0, "html.parser")
    total_pages = get_total_pages(soup0)

    listings = []
    for p in range(1, total_pages + 1):
        page_url = BASE_URL if p == 1 else f"{BASE_URL}page/{p}/"
        html_p = await fetch_html(page_url)
        soup_p = BeautifulSoup(html_p, "html.parser")
        listings.extend(listings_from_page(soup_p))
        await asyncio.sleep(0.5)

    if not listings:
        logger.warning("No recent listings found")
        raise HTTPException(status_code=404, detail="No recent listings found")

    logger.info("Scraping %d listings", len(listings))
    tasks = []
    for title, url in listings:
        doc_id = str(abs(hash(url)) % (1 << 63))
        tasks.append(process_position(doc_id, url))

    await asyncio.gather(*tasks)
    logger.info("Completed scrape_all")
    return {"status": "scraped", "count": len(listings)}

@app.get("/ping")
async def ping():
    logger.debug("GET /ping")
    return {"ping": "pong"}
