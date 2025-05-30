from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss, numpy as np, os, pathlib, json
from typing import List, Dict, Any

app = FastAPI()

DIM         = int(os.getenv("EMBEDDING_DIM", 1536))
INDEX_PATH  = pathlib.Path(os.getenv("FAISS_INDEX_PATH", "/app/data/index.faiss"))
METADATA_PATH = pathlib.Path(os.getenv("FAISS_METADATA_PATH", "/app/data/metadata.json"))
VECTORS_PATH = pathlib.Path(os.getenv("FAISS_VECTORS_PATH", "/app/data/vectors.npy"))
INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

# Store ID mappings, metadata, and vectors separately
id_to_internal_id = {}    # original_id -> internal_sequential_id
internal_id_to_id = {}    # internal_sequential_id -> original_id
metadata_store = {}       # internal_sequential_id -> metadata
vector_store = None       # numpy array of vectors

def load_metadata():
    """Load metadata from disk"""
    global id_to_internal_id, internal_id_to_id, metadata_store
    if METADATA_PATH.exists():
        with open(METADATA_PATH, 'r') as f:
            data = json.load(f)
            id_to_internal_id = data.get('id_to_internal_id', {})
            internal_id_to_id = {int(k): v for k, v in data.get('internal_id_to_id', {}).items()}
            metadata_store = {int(k): v for k, v in data.get('metadata_store', {}).items()}

def save_metadata():
    """Save metadata to disk"""
    data = {
        'id_to_internal_id': id_to_internal_id,
        'internal_id_to_id': {str(k): v for k, v in internal_id_to_id.items()},
        'metadata_store': {str(k): v for k, v in metadata_store.items()}
    }
    with open(METADATA_PATH, 'w') as f:
        json.dump(data, f, indent=2)

def load_vectors():
    """Load vectors from disk"""
    global vector_store
    if VECTORS_PATH.exists():
        vector_store = np.load(VECTORS_PATH)
        print(f"✅ Loaded {len(vector_store)} vectors from {VECTORS_PATH}")
    else:
        vector_store = np.empty((0, DIM), dtype=np.float32)
        print("✅ Initialized empty vector store")

def save_vectors():
    """Save vectors to disk"""
    np.save(VECTORS_PATH, vector_store)

# Initialize index and storage
def init_index():
    global index, vector_store
    
    # Load metadata and vectors first
    load_metadata()
    load_vectors()
    
    if INDEX_PATH.exists():
        try:
            index = faiss.read_index(str(INDEX_PATH))
            print(f"✅ Loaded existing index: {type(index).__name__} with {index.ntotal} vectors")
            
            # Verify consistency
            if index.ntotal != len(vector_store):
                print(f"⚠️  Index/vector mismatch: index={index.ntotal}, vectors={len(vector_store)}")
                if len(vector_store) > 0:
                    print("🔄 Rebuilding index from stored vectors...")
                    rebuild_index_from_vectors()
                    
        except Exception as e:
            print(f"❌ Failed to load index: {e}")
            print("🔄 Creating new index...")
            create_new_index()
    else:
        create_new_index()

def create_new_index():
    global index, vector_store
    # Use simple IndexFlatIP (no IDs needed, we manage them separately)
    index = faiss.IndexFlatIP(DIM)
    vector_store = np.empty((0, DIM), dtype=np.float32)
    faiss.write_index(index, str(INDEX_PATH))
    save_vectors()
    print(f"✅ Created new IndexFlatIP with dimension {DIM}")

def rebuild_index_from_vectors():
    global index
    # Rebuild the FAISS index from stored vectors
    index = faiss.IndexFlatIP(DIM)
    if len(vector_store) > 0:
        index.add(vector_store)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"✅ Rebuilt index with {index.ntotal} vectors")

# Initialize everything
init_index()

class Vector(BaseModel):
    id: str
    vector: list[float]
    metadata: Dict[str, Any] = None

class Query(BaseModel):
    vector: list[float]
    k: int = 5

@app.get("/ping")
async def ping():
    return {"ping": "pong"}

@app.get("/info")
async def get_info():
    """Get index information"""
    return {
        "dimension": DIM,
        "total_vectors": index.ntotal,
        "stored_vectors": len(vector_store),
        "index_type": type(index).__name__,
        "stored_ids": len(internal_id_to_id),
        "supports_reconstruction": True,
        "storage_method": "separate_vector_storage"
    }

@app.get("/stats")
async def get_stats():
    """Get detailed statistics"""
    return {
        "index_ntotal": index.ntotal,
        "stored_vectors": len(vector_store),
        "dimension": DIM,
        "index_path": str(INDEX_PATH),
        "metadata_path": str(METADATA_PATH),
        "vectors_path": str(VECTORS_PATH),
        "id_mappings": len(id_to_internal_id),
        "metadata_entries": len(metadata_store),
        "index_type": type(index).__name__
    }

@app.post("/add")
async def add(v: Vector):
    global vector_store
    
    try:
        print(f"🔍 Adding vector: ID={v.id}, vector_len={len(v.vector)}, expected_dim={DIM}")
        
        # Validate input
        if len(v.vector) != DIM:
            raise HTTPException(
                status_code=400, 
                detail=f"Vector dimension mismatch: got {len(v.vector)}, expected {DIM}"
            )
        
        # Check for invalid values
        if not all(isinstance(x, (int, float)) and not (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) for x in v.vector):
            raise HTTPException(
                status_code=400,
                detail="Vector contains invalid values (NaN, inf, or non-numeric)"
            )
        
        # Convert to numpy array
        vec = np.asarray(v.vector, dtype="float32")
        print(f"✅ Vector converted: shape={vec.shape}, dtype={vec.dtype}")
        
        # Check if ID already exists
        if v.id in id_to_internal_id:
            # Update existing vector
            internal_id = id_to_internal_id[v.id]
            print(f"🔄 Updating existing vector: {v.id} -> internal_id {internal_id}")
            
            # Update vector store
            vector_store[internal_id] = vec
            
            # Update metadata
            if v.metadata:
                metadata_store[internal_id] = v.metadata
            
            # Rebuild index (simple approach for updates)
            rebuild_index_from_vectors()
            
        else:
            # Add new vector
            internal_id = len(vector_store)
            print(f"➕ Adding new vector: {v.id} -> internal_id {internal_id}")
            
            # Add to vector store
            vector_store = np.vstack([vector_store, vec.reshape(1, -1)])
            
            # Add to FAISS index
            index.add(vec.reshape(1, -1))
            
            # Store mappings and metadata
            id_to_internal_id[v.id] = internal_id
            internal_id_to_id[internal_id] = v.id
            
            if v.metadata:
                metadata_store[internal_id] = v.metadata
            else:
                metadata_store[internal_id] = {
                    "original_id": v.id,
                    "added_at": str(np.datetime64('now')),
                    "vector_dim": len(v.vector)
                }
        
        print(f"✅ Vector added successfully, ntotal={index.ntotal}")
        
        # Save everything to disk
        faiss.write_index(index, str(INDEX_PATH))
        save_vectors()
        save_metadata()
        print(f"✅ All data saved to disk")
        
        return {
            "stored": True, 
            "ntotal": index.ntotal,
            "internal_id": internal_id,
            "original_id": v.id,
            "index_type": type(index).__name__,
            "action": "updated" if v.id in id_to_internal_id else "added"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in add(): {type(e).__name__}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to add vector: {type(e).__name__}: {str(e)}"
        )

@app.post("/search")
async def search(q: Query):
    if index.ntotal == 0:
        raise HTTPException(status_code=400, detail="Index is empty")
    
    vec = np.asarray(q.vector, dtype="float32").reshape(1, -1)
    D, I = index.search(vec, q.k)
    
    # Convert internal IDs back to original IDs with metadata
    results = []
    for i, (internal_id, distance) in enumerate(zip(I[0], D[0])):
        if internal_id == -1:  # FAISS returns -1 for missing results
            continue
        
        result = {
            "internal_id": int(internal_id),
            "distance": float(distance),
            "original_id": internal_id_to_id.get(internal_id, f"unknown_{internal_id}"),
        }
        
        if internal_id in metadata_store:
            result["metadata"] = metadata_store[internal_id]
            
        results.append(result)
    
    return {"results": results, "total_found": len(results)}

@app.get("/get/{vector_id}")
async def get_by_id(vector_id: str):
    """Get a specific vector by original ID with full vector data"""
    if vector_id not in id_to_internal_id:
        raise HTTPException(status_code=404, detail=f"Vector ID '{vector_id}' not found")
    
    internal_id = id_to_internal_id[vector_id]
    
    if internal_id >= len(vector_store):
        raise HTTPException(status_code=404, detail=f"Internal ID {internal_id} out of range")
    
    vector = vector_store[internal_id]
    metadata = metadata_store.get(internal_id, {})
    
    return {
        "id": vector_id,
        "internal_id": internal_id,
        "vector": vector.tolist(),
        "metadata": metadata
    }

@app.get("/list")
async def list_all_ids():
    """List all vector IDs in the database"""
    return {
        "total_count": len(internal_id_to_id),
        "vector_ids": list(id_to_internal_id.keys()),
        "internal_ids": list(internal_id_to_id.keys()),
        "index_type": type(index).__name__
    }

@app.get("/all")
async def get_all_vectors():
    """Get all vectors with metadata"""
    if len(vector_store) == 0:
        return {"vectors": [], "total": 0}
    
    all_vectors = []
    for internal_id, original_id in internal_id_to_id.items():
        vector_info = {
            "id": original_id,
            "internal_id": internal_id,
            "metadata": metadata_store.get(internal_id, {}),
            "has_vector_data": True
        }
        all_vectors.append(vector_info)
    
    return {
        "vectors": all_vectors,
        "total": len(all_vectors),
        "index_type": type(index).__name__,
        "storage_method": "separate_vector_storage"
    }

@app.get("/all_with_vectors")
async def get_all_with_vectors():
    """Get all vectors WITH their actual vector data"""
    if len(vector_store) == 0:
        return {"vectors": [], "total": 0}
    
    all_vectors = []
    
    for internal_id, original_id in internal_id_to_id.items():
        if internal_id < len(vector_store):
            vector = vector_store[internal_id]
            vector_info = {
                "id": original_id,
                "internal_id": internal_id,
                "vector": vector.tolist(),
                "metadata": metadata_store.get(internal_id, {})
            }
            all_vectors.append(vector_info)
    
    return {
        "vectors": all_vectors,
        "total": len(all_vectors),
        "index_type": type(index).__name__,
        "storage_method": "separate_vector_storage"
    }

@app.get("/reconstruct/{internal_id}")
async def reconstruct_vector(internal_id: int):
    """Get a vector by internal ID"""
    if internal_id not in internal_id_to_id:
        raise HTTPException(status_code=404, detail=f"Internal ID {internal_id} not found")
    
    if internal_id >= len(vector_store):
        raise HTTPException(status_code=404, detail=f"Internal ID {internal_id} out of range")
    
    vector = vector_store[internal_id]
    original_id = internal_id_to_id[internal_id]
    metadata = metadata_store.get(internal_id, {})
    
    return {
        "id": original_id,
        "internal_id": internal_id,
        "vector": vector.tolist(),
        "metadata": metadata
    }

@app.delete("/clear")
async def clear_all():
    """Clear all vectors from the index"""
    global id_to_internal_id, internal_id_to_id, metadata_store, vector_store, index
    
    # Recreate everything
    create_new_index()
    
    # Clear all storage
    id_to_internal_id = {}
    internal_id_to_id = {}
    metadata_store = {}
    
    # Save to disk
    save_metadata()
    
    return {
        "cleared": True, 
        "ntotal": index.ntotal,
        "index_type": type(index).__name__
    }

@app.delete("/delete/{vector_id}")
async def delete_vector(vector_id: str):
    """Delete a specific vector"""
    global vector_store
    
    if vector_id not in id_to_internal_id:
        raise HTTPException(status_code=404, detail=f"Vector ID '{vector_id}' not found")
    
    internal_id = id_to_internal_id[vector_id]
    
    # Remove from mappings and metadata
    del id_to_internal_id[vector_id]
    del internal_id_to_id[internal_id]
    if internal_id in metadata_store:
        del metadata_store[internal_id]
    
    # Remove from vector store (set to zeros, but keep position for consistency)
    if internal_id < len(vector_store):
        vector_store[internal_id] = np.zeros(DIM, dtype=np.float32)
    
    # Rebuild index (simple approach)
    rebuild_index_from_vectors()
    
    # Save everything
    save_vectors()
    save_metadata()
    
    return {
        "deleted": True,
        "vector_id": vector_id,
        "internal_id": internal_id
    }

@app.post("/test_add")
async def test_add():
    """Test endpoint to add a simple vector"""
    try:
        test_vector = [0.1] * DIM
        test_id = "test_vector_001"
        
        test_data = Vector(
            id=test_id,
            vector=test_vector,
            metadata={"test": True, "created": "debug"}
        )
        
        result = await add(test_data)
        return {
            "test_passed": True,
            "result": result
        }
    except Exception as e:
        return {
            "test_passed": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@app.get("/debug")
async def debug_info():
    """Get debug information about the current state"""
    try:
        return {
            "index_type": type(index).__name__,
            "index_ntotal": index.ntotal,
            "stored_vectors": len(vector_store),
            "expected_dimension": DIM,
            "vector_store_shape": vector_store.shape if vector_store is not None else None,
            "metadata_entries": len(metadata_store),
            "id_mappings": len(id_to_internal_id),
            "index_path_exists": INDEX_PATH.exists(),
            "metadata_path_exists": METADATA_PATH.exists(),
            "vectors_path_exists": VECTORS_PATH.exists(),
        }
    except Exception as e:
        return {
            "error": str(e),
            "error_type": type(e).__name__
        }
@app.post("/force_reset")
async def force_reset():
    """Force reset to simple IndexFlatIP"""
    global index, id_to_internal_id, internal_id_to_id, metadata_store, vector_store
    
    try:
        print("🚨 Force reset: clearing everything...")
        
        # Clear memory variables
        id_to_internal_id = {}
        internal_id_to_id = {}
        metadata_store = {}
        
        # Create fresh IndexFlatIP (NOT IndexIDMap)
        print("Creating IndexFlatIP...")
        index = faiss.IndexFlatIP(DIM)
        vector_store = np.empty((0, DIM), dtype=np.float32)
        
        # Delete old files
        for path in [INDEX_PATH, METADATA_PATH, VECTORS_PATH]:
            if path.exists():
                path.unlink()
                print(f"Deleted {path}")
        
        # Save fresh state
        faiss.write_index(index, str(INDEX_PATH))
        save_vectors()
        save_metadata()
        
        print(f"✅ Force reset complete: {type(index).__name__}")
        
        return {
            "reset": True,
            "index_type": type(index).__name__,
            "message": "Reset to IndexFlatIP complete"
        }
        
    except Exception as e:
        print(f"❌ Force reset failed: {e}")
        return {"reset": False, "error": str(e)}