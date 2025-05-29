from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss, numpy as np, os, pathlib

app = FastAPI()

DIM         = int(os.getenv("EMBEDDING_DIM", 1536))
INDEX_PATH  = pathlib.Path(os.getenv("FAISS_INDEX_PATH", "/app/data/index.faiss"))
INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

# Initialise or load index ---------------------------------------------------
if INDEX_PATH.exists():
    index = faiss.read_index(str(INDEX_PATH))
else:
    # IVF-Flat with 256 centroids
    quantizer = faiss.IndexFlatIP(DIM)
    index     = faiss.IndexIVFFlat(quantizer, DIM, 256, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = 10
    index.train(np.random.random((1000, DIM)).astype("float32"))  # dummy train
    faiss.write_index(index, str(INDEX_PATH))

# ---------------------------------------------------------------------------

class Vector(BaseModel):
    id: str
    vector: list[float]

class Query(BaseModel):
    vector: list[float]
    k: int = 5

@app.get("/ping")
async def ping():
    return {"ping": "pong"}

@app.post("/add")
async def add(v: Vector):
    vec = np.asarray(v.vector, dtype="float32").reshape(1, -1)
    ids = np.asarray([abs(hash(v.id)) % (1 << 63)], dtype="int64")
    index.add_with_ids(vec, ids)
    faiss.write_index(index, str(INDEX_PATH))
    return {"stored": True, "ntotal": index.ntotal}

@app.post("/search")
async def search(q: Query):
    if index.ntotal == 0:
        raise HTTPException(status_code=400, detail="Index is empty")
    vec = np.asarray(q.vector, dtype="float32").reshape(1, -1)
    D, I = index.search(vec, q.k)
    return {"ids": I.tolist()[0], "distances": D.tolist()[0]}
