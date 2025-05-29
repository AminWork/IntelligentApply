import os, numpy as np, httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient

app = FastAPI()

MONGO_URI     = os.getenv("MONGO_URI", "mongodb://root:example@mongo:27017")
FAISS_ENDPOINT = os.getenv("FAISS_ENDPOINT", "http://faiss-db:8080")
DIM            = int(os.getenv("EMBEDDING_DIM", 1536))

mongo = AsyncIOMotorClient(MONGO_URI)
db    = mongo["fetcher"]

class Item(BaseModel):
    id:  str
    url: str

@app.get("/ping")
async def ping():
    return {"ping": "pong"}

@app.post("/fetch")
async def fetch(item: Item):
    # 1. download raw text
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(item.url)
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Cannot download source")
    text = resp.text

    # 2. (placeholder) embed → random vector
    vector = np.random.random(DIM).astype("float32").tolist()

    # 3. save raw doc
    await db.docs.insert_one({"_id": item.id, "url": item.url, "text": text})

    # 4. push vector to faiss
    async with httpx.AsyncClient(timeout=10) as client:
        await client.post(f"{FAISS_ENDPOINT}/add",
                          json={"id": item.id, "vector": vector})

    return {"stored": True}
