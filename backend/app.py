import fastapi
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
import cv2
import numpy as np
from pinecone import Pinecone
import asyncio
import functools
import time
from concurrent.futures import ThreadPoolExecutor
import io
from typing import Union, List, Dict, Any

app = FastAPI()

# Enable CORS to allow React frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple keyword-based query enhancer
class QueryEnhancer:
    def __init__(self):
        self.product_attributes = {
            "color": ["red", "blue", "green", "black", "white", "yellow", "purple", "pink",
                      "gray", "brown", "orange", "silver", "gold", "dark", "light"],
            "size": ["small", "medium", "large", "xl", "xxl", "extra large", "tiny", "huge", "compact"],
            "material": ["leather", "cotton", "wool", "silk", "plastic", "metal", "wood",
                        "glass", "ceramic", "steel", "aluminum", "polyester"],
            "style": ["modern", "classic", "vintage", "casual", "formal", "sporty", "elegant",
                     "minimalist", "traditional", "contemporary", "bohemian", "luxury"],
            "price": ["cheap", "affordable", "expensive", "premium", "budget", "low-cost",
                     "high-end", "discount", "quality", "value"],
            "brand": ["brand", "designer", "manufacturer", "company", "maker"],
            "condition": ["new", "used", "refurbished", "pre-owned", "mint condition", "like new"],
            "rating": ["top rated", "best seller", "popular", "highly rated", "well reviewed"]
        }
        self.synonyms = {
            "shirt": ["tee", "top", "tshirt", "t-shirt", "blouse"],
            "pants": ["trousers", "jeans", "slacks", "leggings"],
            "shoes": ["sneakers", "boots", "footwear", "sandals"],
            "laptop": ["computer", "notebook", "pc"],
            "phone": ["smartphone", "mobile", "cell phone"],
            "watch": ["timepiece", "wristwatch", "smartwatch"],
            "tv": ["television", "monitor", "screen", "display"],
            "camera": ["digital camera", "dslr", "point and shoot"],
            "furniture": ["chair", "table", "sofa", "desk"],
            "makeup": ["cosmetics", "beauty products"],
            "book": ["novel", "textbook", "ebook", "paperback"],
            "jewelry": ["necklace", "bracelet", "ring", "earrings"],
            "bag": ["purse", "handbag", "backpack", "tote"],
            "kitchen": ["cookware", "appliance", "utensil"],
            "striped": ["stripes", "stripe pattern", "lined"],
            "women": ["womens", "ladies", "female"],
            "men": ["mens", "gentlemen", "male"]
        }
        print("Simple query enhancer initialized")

    def enhance_query(self, text_query):
        query_lower = text_query.lower()
        words = query_lower.split()
        enhanced_terms = set(words)
        for word in words:
            if word in self.synonyms:
                enhanced_terms.update(self.synonyms[word][:2])
        detected_attrs = {}
        for attr_type, attr_values in self.product_attributes.items():
            for value in attr_values:
                if value in query_lower:
                    detected_attrs[attr_type] = value
                    break
        if "product" not in enhanced_terms and "item" not in enhanced_terms:
            enhanced_terms.add("product")
        enhanced_query = " ".join(enhanced_terms)
        print(f"Original query: '{text_query}'")
        print(f"Enhanced query: '{enhanced_query}'")
        return enhanced_query

# Embedding Cache
class EmbeddingCache:
    def __init__(self, max_size=100):
        self.image_cache = {}
        self.text_cache = {}
        self.max_size = max_size

    def get_image_embedding(self, image_hash):
        return self.image_cache.get(image_hash)

    def set_image_embedding(self, image_hash, embedding):
        if len(self.image_cache) >= self.max_size:
            self.image_cache.pop(next(iter(self.image_cache)))
        self.image_cache[image_hash] = embedding

    def get_text_embedding(self, text):
        return self.text_cache.get(text)

    def set_text_embedding(self, text, embedding):
        if len(self.text_cache) >= self.max_size:
            self.text_cache.pop(next(iter(self.text_cache)))
        self.text_cache[text] = embedding

# Merged Search Engine
class ProductSearchEngine:
    def __init__(self, api_key, index_name, device="cpu", clip_model_name="openai/clip-vit-base-patch32", yolo_model_path="yolov5s.pt"):
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        print(f"Using device: {self.device}")
        start = time.time()
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        try:
            self.yolo_model = YOLO(yolo_model_path)
        except Exception as e:
            print(f"Error loading YOLO model: {e}. Please ensure yolov5s.pt is available.")
            self.yolo_model = None
        print(f"Models loaded in {time.time() - start:.2f} seconds")
        try:
            self.pc = Pinecone(api_key=api_key)
            if index_name not in self.pc.list_indexes().names():
                print(f"Index '{index_name}' does not exist. Please create it in Pinecone.")
            self.index = self.pc.Index(index_name)
        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            self.index = None
        self.cache = EmbeddingCache()
        self.query_enhancer = QueryEnhancer()
        self.executor = ThreadPoolExecutor(max_workers=4)

    def get_clip_embedding(self, image: Image.Image):
        image_hash = hash(image.tobytes())
        cached = self.cache.get_image_embedding(image_hash)
        if cached is not None:
            return cached
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embedding = self.clip_model.get_image_features(**inputs).cpu().squeeze().tolist()
        self.cache.set_image_embedding(image_hash, embedding)
        return embedding

    def get_text_embeddings(self, texts: List[str]):
        results = []
        uncached_texts = []
        uncached_indices = []
        for i, text in enumerate(texts):
            cached = self.cache.get_text_embedding(text)
            if cached is not None:
                results.append((i, cached))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        if uncached_texts:
            inputs = self.clip_processor(text=uncached_texts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                embeddings = self.clip_model.get_text_features(**inputs).cpu()
            for idx, text_idx in enumerate(uncached_indices):
                embedding = embeddings[idx].squeeze().tolist()
                self.cache.set_text_embedding(texts[text_idx], embedding)
                results.append((text_idx, embedding))
        return [emb for _, emb in sorted(results, key=lambda x: x[0])]

    def get_text_embedding(self, text: str):
        return self.get_text_embeddings([text])[0]

    async def async_query(self, vector, filter=None, top_k=5, include_metadata=True, namespace=""):
        if self.index is None:
            raise Exception("Pinecone index not initialized.")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            functools.partial(
                self.index.query,
                vector=vector,
                filter=filter,
                top_k=top_k,
                include_metadata=include_metadata,
                namespace=namespace
            )
        )

    async def metadata_search(self, query_terms, top_k=10):
        filter_terms = [term for term in query_terms.lower().split() if len(term) >= 3]
        if not filter_terms:
            return {"matches": [], "message": "No valid search terms provided for metadata search."}
        metadata_filter = {"$or": []}
        for field in ["title", "description", "category"]:
            metadata_filter["$or"].append({field: {"$in": filter_terms}})
        try:
            results = await self.async_query(vector=[0] * 512, filter=metadata_filter, top_k=top_k)
            return results
        except Exception as e:
            return {"matches": [], "message": f"Error in metadata search: {str(e)}"}

    async def enhanced_text_search(self, query, top_k=5, use_metadata=True, use_vector=True):
        try:
            enhanced_query = self.query_enhancer.enhance_query(query)
            tasks = []
            if use_vector:
                try:
                    text_vector = self.get_text_embedding(enhanced_query)
                    tasks.append(self.async_query(text_vector, top_k=top_k))
                except Exception as e:
                    print(f"Error during vector search: {e}")
            if use_metadata:
                tasks.append(self.metadata_search(enhanced_query, top_k=top_k))
            if not tasks:
                return {"matches": [], "message": "No search tasks could be executed."}
            results = await asyncio.gather(*tasks)
            if len(results) == 1:
                return results[0]
            combined_scores = {}
            for i, result in enumerate(results):
                if "message" in result:
                    continue
                weight = 0.6 if i == 0 else 0.4
                for match in result.matches:
                    if match.id in combined_scores:
                        combined_scores[match.id]['score'] += match.score * weight
                        combined_scores[match.id]['count'] += 1
                    else:
                        combined_scores[match.id] = {
                            'score': match.score * weight,
                            'count': 1,
                            'metadata': match.metadata
                        }
            for item_id in combined_scores:
                combined_scores[item_id]['score'] /= combined_scores[item_id]['count']
            sorted_items = sorted(combined_scores.items(), key=lambda x: x[1]['score'], reverse=True)[:top_k]
            class EnhancedResult:
                def __init__(self, matches):
                    self.matches = matches
            class Match:
                def __init__(self, id, score, metadata):
                    self.id = id
                    self.score = score
                    self.metadata = metadata
            matches = [
                Match(id=item_id, score=item_data['score'], metadata=item_data['metadata'])
                for item_id, item_data in sorted_items
            ]
            return EnhancedResult(matches)
        except Exception as e:
            return {"matches": [], "message": f"Error in enhanced text search: {str(e)}"}

    async def image_search(self, image: Image.Image, top_k=5):
        if self.yolo_model is None:
            return {"matches": [], "message": "YOLO model not loaded. Image search unavailable."}
        try:
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            results = self.yolo_model(cv_image)
            cropped_images = []
            for result in results:
                for box in result.boxes.data:
                    x1, y1, x2, y2, conf, cls = map(int, box[:6])
                    cropped_img = image.crop((x1, y1, x2, y2))
                    cropped_images.append(cropped_img)
            if not cropped_images:
                return {"matches": [], "message": "No objects detected in the image."}
            query_vectors = [self.get_clip_embedding(img) for img in cropped_images]
            all_results = []
            for i, query_vector in enumerate(query_vectors):
                result = await self.async_query(vector=query_vector, top_k=top_k)
                all_results.append({"object": f"Object {i+1}", "results": result})
            return all_results
        except Exception as e:
            return {"matches": [], "message": f"Error in image search: {str(e)}"}

# Initialize Search Engine
API_KEY = "pcsk_2vRc2j_PzX8dPALzZ7oWgtxiFURfFUBFGdCBxW458xutz598age2jTBdKQBk3stGNT6uaZ"
INDEX_NAME = "product-search"
YOLO_MODEL_PATH = "yolov5s.pt"  # Update with correct path
search_engine = ProductSearchEngine(API_KEY, INDEX_NAME, yolo_model_path=YOLO_MODEL_PATH)

# API Endpoints
@app.get("/text-search")
async def text_search(query: str, top_k: int = 10):
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required.")
    results = await search_engine.enhanced_text_search(query, top_k=top_k)
    # Check if results is an EnhancedResult object (successful search)
    if hasattr(results, 'matches'):
        results_data = [
            {
                "Object": "Text Query",
                "Rank": i + 1,
                "Score": f"{match.score:.4f}",
                "Title": match.metadata.get("title", "N/A"),
                "Price": match.metadata.get("price", "N/A"),
                "Website": match.metadata.get("product_url", "N/A"),
                "Description": (match.metadata.get("description", "N/A")[:100] + "..." if len(str(match.metadata.get("description", ""))) > 100 else match.metadata.get("description", "N/A"))
            }
            for i, match in enumerate(results.matches)
        ]
        return {"results": results_data, "message": f"Text Search Results: '{query}' - {len(results_data)} items found"}
    # Handle error case where results is a dictionary with a message
    return {"results": [], "message": results.get("message", "Unknown error in text search.")}

@app.post("/image-search")
async def image_search(file: UploadFile = File(...), top_k: int = 5):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        results = await search_engine.image_search(image, top_k=top_k)
        if "message" in results:
            return {"results": [], "message": results["message"]}
        results_data = []
        for obj_result in results:
            obj_name = obj_result["object"]
            for i, match in enumerate(obj_result["results"].matches):
                metadata = match.metadata
                results_data.append({
                    "Object": obj_name,
                    "Rank": i + 1,
                    "Score": f"{match.score:.4f}",
                    "Title": metadata.get("title", "N/A"),
                    "Price": metadata.get("price", "N/A"),
                    "Website": metadata.get("product_url", "N/A"),
                    "Description": (metadata.get("description", "N/A")[:100] + "..." if len(str(metadata.get("description", ""))) > 100 else metadata.get("description", "N/A"))
                })
        return {"results": results_data, "message": f"Image Search Results - {len(results_data)} items found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in image search: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
