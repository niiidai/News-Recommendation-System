from fastapi import FastAPI, Depends, HTTPException, Query
from typing import List, Dict, Any, Optional
import logging
import redis
import json
import time
from pantic import BaseModel

# Import the model and feature processing modules
from models.collaborative.matrix_factorization import MatrixFactorization
from models.deep_learning.neural_cf import NeuralCollaborativeFiltering
from feature_engineering.user_features import UserFeatureExtractor
from feature_engineering.item_features import ItemFeatureExtractor
from data.storage.mongodb_client import MongoDBClient

app = FastAPI(title="Recommendation System API", description="A REST API for providing personalized recommendation services")

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recommender-api")

# Redis configuration
redis_client = redis.Redis(host='localhost', port=6379, db=0)
CACHE_TTL = 3600  # Cache TTL of 1 hour

# MongoDB configuration
mongo_client = MongoDBClient(connection_string="mongodb://localhost:27017/", db_name="recommender")

# Load models (in a real environment, this should be loaded via configuration)
matrix_factorization_model = None
neural_cf_model = None

# Response model
class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Dict[str, Any]]
    model_used: str
    timestamp: float

@app.on_event("startup")
async def startup_event():
    """Load models on service startup"""
    global matrix_factorization_model, neural_cf_model
    # In a real application, pre-trained models would be loaded from persistent storage here
    logger.info("Loading recommendation models...")
    # Load the actual models here

@app.get("/recommendations/", response_model=RecommendationResponse)
async def get_recommendations(
    user_id: str,
    count: int = Query(10, gt=0, lt=100),
    model_type: str = Query("collaborative", regex="^(collaborative|neural|hybrid)$"),
    context: Optional[Dict[str, Any]] = None
):
    """Get personalized recommendations"""
    start_time = time.time()
    
    # Try to get recommendations from the cache
    cache_key = f"recommendations:{user_id}:{model_type}:{count}"
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        logger.info(f"Returning recommendations for user {user_id} from cache")
        return json.loads(cached_result)
    
    # Get user features
    user_events = mongo_client.get_user_events(user_id)
    if not user_events:
        raise HTTPException(status_code=404, detail=f"Data for user {user_id} not found")
    
    # Generate recommendations based on the model type
    recommendations = []
    model_used = model_type
    
    if model_type == "collaborative":
        # Collaborative filtering recommendations
        if matrix_factorization_model is not None:
            item_scores = matrix_factorization_model.recommend_for_user(user_id, top_n=count)
            for item_id, score in item_scores:
                item_data = mongo_client.get_item_events(item_id)
                if item_data:
                    recommendations.append({
                        "item_id": item_id,
                        "score": score,
                        "title": item_data[0].get("title", ""),
                        "category": item_data[0].get("category", "")
                    })
    
    elif model_type == "neural":
        # Neural network recommendations
        if neural_cf_model is not None:
            # Get candidate items
            # In a real application, an appropriate candidate set would be selected based on business logic
            all_items = [doc["item_id"] for doc in mongo_client.db["items"].find({}, {"item_id": 1})]
            item_scores = neural_cf_model.recommend_for_user(user_id, all_items, top_n=count)
            for item_id, score in item_scores:
                item_data = mongo_client.get_item_events(item_id)
                if item_data:
                    recommendations.append({
                        "item_id": item_id,
                        "score": float(score),
                        "title": item_data[0].get("title", ""),
                        "category": item_data[0].get("category", "")
                    })
    
    elif model_type == "hybrid":
        # Hybrid recommendations (combining results from multiple models)
        # Simplified here to only use collaborative filtering
        if matrix_factorization_model is not None:
            item_scores = matrix_factorization_model.recommend_for_user(user_id, top_n=count)
            for item_id, score in item_scores:
                item_data = mongo_client.get_item_events(item_id)
                if item_data:
                    recommendations.append({
                        "item_id": item_id,
                        "score": score,
                        "title": item_data[0].get("title", ""),
                        "category": item_data[0].get("category", "")
                    })
    
    # Consider context for personalized adjustments (if needed)
    if context:
        # Adjust recommendation results based on context
        # For example, consider contextual factors like time, location, etc.
        pass
    
    # Build the response
    response = {
        "user_id": user_id,
        "recommendations": recommendations,
        "model_used": model_used,
        "timestamp": time.time()
    }
    
    # Cache the result
    redis_client.setex(cache_key, CACHE_TTL, json.dumps(response))
    
    logger.info(f"Generated recommendations for user {user_id} in {time.time() - start_time:.3f} seconds")
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}