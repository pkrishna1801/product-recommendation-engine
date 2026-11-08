import numpy as np
import openai
import json
import os
import hashlib
from config import config

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Falling back to NumPy for similarity search.")


def _normalize(vec):
    """Return a unit-length copy of vec (or vec unchanged if it is all zeros)."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class EmbeddingService:
    """
    Service to handle embeddings for products and user data
    Supports semantic similarity search for more accurate recommendations
    """

    # Decay applied per step back in browsing history (newest item has weight 1.0)
    RECENCY_DECAY = 0.8

    def __init__(self):
        """Initialize the embedding service with configuration"""
        openai.api_key = config.get('OPENAI_API_KEY')
        self.embedding_model = config.get('EMBEDDING_MODEL', 'text-embedding-ada-002')
        self.embedding_dim = config.get('EMBEDDING_DIM', 1536)  # Dimension of embeddings
        self.cache_dir = config.get('CACHE_DIR', 'cache')

        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # Index over the FULL catalog; filtering happens at query time
        self.product_embeddings = {}
        self.product_index = None
        self.product_ids = []

    def get_embedding(self, text):
        """
        Get embedding for a text string using OpenAI's API.
        Uses a disk cache keyed by a stable hash so it survives restarts.
        Returns None if the embedding could not be obtained.
        """
        # Normalize text for consistent embedding
        text = text.strip().lower()

        # Stable cache key (Python's built-in hash() is randomized per process)
        cache_key = hashlib.sha256(f"{self.embedding_model}:{text}".encode('utf-8')).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"emb_{cache_key}.json")

        # Check if we have this embedding cached
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return np.array(json.load(f))
            except (json.JSONDecodeError, IOError):
                # If there's an issue with the cache file, proceed to generate a new embedding
                pass

        try:
            # Get embedding from OpenAI API
            response = openai.Embedding.create(
                model=self.embedding_model,
                input=text
            )
            embedding = response["data"][0]["embedding"]

            # Cache the embedding
            with open(cache_path, 'w') as f:
                json.dump(embedding, f)

            return np.array(embedding)

        except Exception as e:
            # Don't return a zero vector: it would silently pollute averages and the index
            print(f"Error getting embedding: {str(e)}")
            return None

    def get_product_embedding(self, product):
        """
        Get embedding for a product by combining its relevant attributes.
        Reuses the in-memory index embedding when available.
        """
        cached = self.product_embeddings.get(product.get('id'))
        if cached is not None:
            return cached

        # Extract relevant product information
        product_info = [
            f"Name: {product.get('name', '')}",
            f"Category: {product.get('category', '')}, Subcategory: {product.get('subcategory', '')}",
            f"Brand: {product.get('brand', '')}, Price: {product.get('price', 0)}",
            f"Description: {product.get('description', '')}"
        ]

        # Add features if available
        if 'features' in product and product['features']:
            product_info.append(f"Features: {', '.join(product['features'])}")

        # Add tags if available
        if 'tags' in product and product['tags']:
            product_info.append(f"Tags: {', '.join(product['tags'])}")

        # Combine all product information into a single text
        product_text = " ".join(product_info)

        # Get embedding for the product text
        return self.get_embedding(product_text)

    def embed_all_products(self, products):
        """
        Create embeddings for all products and build a cosine-similarity search index.
        Products whose embedding fails are left out of the index.
        """
        # Reset storage
        self.product_embeddings = {}
        self.product_ids = []
        self.product_index = None

        if not products:
            return

        # Generate embeddings for all products
        embeddings = []
        for product in products:
            embedding = self.get_product_embedding(product)
            if embedding is None:
                continue

            embedding = _normalize(embedding)
            self.product_ids.append(product['id'])
            self.product_embeddings[product['id']] = embedding
            embeddings.append(embedding)

        if not embeddings:
            return

        embeddings_array = np.array(embeddings).astype('float32')

        # Build search index (inner product on unit vectors == cosine similarity)
        if FAISS_AVAILABLE:
            self.product_index = faiss.IndexFlatIP(self.embedding_dim)
            self.product_index.add(embeddings_array)
        else:
            self.product_index = embeddings_array

    def ensure_index(self, all_products):
        """
        Build the index over the full catalog if it hasn't been built yet,
        or rebuild it if the catalog has changed.
        """
        catalog_ids = {p['id'] for p in all_products}
        if self.product_index is None or catalog_ids != set(self.product_ids):
            self.embed_all_products(all_products)

    def get_user_interests_embedding(self, browsed_products, user_preferences):
        """
        Generate an embedding representing the user's interests
        based on browsing history and explicit preferences.
        Returns None if no usable signal is available.
        """
        history_embedding = self._get_history_embedding(browsed_products)
        preferences_embedding = self._get_preferences_embedding(user_preferences)

        if history_embedding is None:
            return preferences_embedding
        if preferences_embedding is None:
            return history_embedding

        # Combine with 70% weight on browsing history, 30% on preferences
        return _normalize(0.7 * history_embedding + 0.3 * preferences_embedding)

    def _get_history_embedding(self, browsed_products):
        """
        Create an embedding representing the user's browsing history.
        browsed_products is ordered newest-first (as stored by UserService and the frontend),
        so weights decay with position.
        """
        if not browsed_products:
            return None

        weighted_embedding = np.zeros(self.embedding_dim)
        found = False
        for position, product in enumerate(browsed_products):
            embedding = self.get_product_embedding(product)
            if embedding is None:
                continue
            weighted_embedding += (self.RECENCY_DECAY ** position) * _normalize(embedding)
            found = True

        return _normalize(weighted_embedding) if found else None

    def _get_preferences_embedding(self, user_preferences):
        """
        Create an embedding representing the user's explicit preferences.
        Returns None if there are no preferences or the embedding fails.
        """
        if not user_preferences:
            return None

        # Extract preference info
        preference_texts = []

        for key, value in user_preferences.items():
            if value:
                # Handle different types of preference values
                if isinstance(value, dict):
                    # Handle nested dictionaries like price range
                    for sub_key, sub_value in value.items():
                        if sub_value is not None:
                            preference_texts.append(f"{key.replace('_', ' ')} {sub_key}: {sub_value}")
                elif isinstance(value, list):
                    # Handle lists like category preferences
                    preference_texts.append(f"{key.replace('_', ' ')}: {', '.join(map(str, value))}")
                else:
                    # Handle simple values
                    preference_texts.append(f"{key.replace('_', ' ')}: {value}")

        if not preference_texts:
            return None

        # Combine preferences into a single text
        preferences_text = "User prefers " + ". ".join(preference_texts)

        embedding = self.get_embedding(preferences_text)
        return _normalize(embedding) if embedding is not None else None

    def find_similar_products(self, query_embedding, products, top_n=5, exclude_ids=None):
        """
        Find the most similar products to a query embedding.

        The index must already cover the full catalog (see ensure_index); `products`
        restricts which products are eligible to be returned.

        Returns:
        - List of tuples (product, cosine_similarity)
        """
        exclude_ids = set(exclude_ids or [])

        if query_embedding is None or self.product_index is None or not self.product_ids:
            return []

        eligible = {p['id']: p for p in products if p['id'] not in exclude_ids}
        if not eligible:
            return []

        query = np.array([_normalize(query_embedding)]).astype('float32')

        if FAISS_AVAILABLE and isinstance(self.product_index, faiss.Index):
            scores, indices = self.product_index.search(query, len(self.product_ids))
            ranked = zip(indices[0], scores[0])
        else:
            scores = self.product_index @ query[0]
            order = np.argsort(-scores)
            ranked = ((idx, scores[idx]) for idx in order)

        results = []
        for idx, score in ranked:
            if idx < 0 or idx >= len(self.product_ids):
                continue
            product = eligible.get(self.product_ids[idx])
            if product is not None:
                results.append((product, float(score)))
                if len(results) >= top_n:
                    break

        return results
