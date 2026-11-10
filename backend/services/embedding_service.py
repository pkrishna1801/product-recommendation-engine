import numpy as np
import openai
import json
import os
import hashlib
import threading
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
        self._index_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Raw embeddings + disk cache
    # ------------------------------------------------------------------

    def _cache_path(self, normalized_text):
        """Disk cache path. Stable sha256 key (Python's hash() is randomized per process)."""
        key = hashlib.sha256(f"{self.embedding_model}:{normalized_text}".encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"emb_{key}.json")

    @staticmethod
    def _read_cache(cache_path):
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return np.array(json.load(f))
            except (json.JSONDecodeError, IOError):
                pass
        return None

    @staticmethod
    def _write_cache(cache_path, embedding):
        try:
            with open(cache_path, 'w') as f:
                json.dump(embedding, f)
        except IOError:
            pass

    def get_embedding(self, text):
        """
        Get embedding for a text string using OpenAI's API.
        Uses a disk cache keyed by a stable hash so it survives restarts.
        Returns None if the embedding could not be obtained.
        """
        text = text.strip().lower()
        cache_path = self._cache_path(text)

        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached

        try:
            response = openai.Embedding.create(
                model=self.embedding_model,
                input=text
            )
            embedding = response["data"][0]["embedding"]
            self._write_cache(cache_path, embedding)
            return np.array(embedding)

        except Exception as e:
            # Don't return a zero vector: it would silently pollute averages and the index
            print(f"Error getting embedding: {str(e)}")
            return None

    def get_embeddings_batch(self, texts, batch_size=100):
        """
        Embed many texts with as few API calls as possible (one call per
        batch_size uncached texts). Returns a list aligned with `texts`;
        items that failed are None.
        """
        normalized = [t.strip().lower() for t in texts]
        results = [None] * len(texts)
        missing = []  # indices not found in the disk cache

        for i, text in enumerate(normalized):
            cached = self._read_cache(self._cache_path(text))
            if cached is not None:
                results[i] = cached
            else:
                missing.append(i)

        for start in range(0, len(missing), batch_size):
            chunk = missing[start:start + batch_size]
            try:
                response = openai.Embedding.create(
                    model=self.embedding_model,
                    input=[normalized[i] for i in chunk]
                )
                for item in response["data"]:
                    i = chunk[item["index"]]
                    embedding = item["embedding"]
                    self._write_cache(self._cache_path(normalized[i]), embedding)
                    results[i] = np.array(embedding)
            except Exception as e:
                print(f"Error getting batch embeddings: {str(e)}")

        return results

    # ------------------------------------------------------------------
    # Products + index
    # ------------------------------------------------------------------

    @staticmethod
    def _product_text(product):
        """Text used to embed a product"""
        product_info = [
            f"Name: {product.get('name', '')}",
            f"Category: {product.get('category', '')}, Subcategory: {product.get('subcategory', '')}",
            f"Brand: {product.get('brand', '')}, Price: {product.get('price', 0)}",
            f"Description: {product.get('description', '')}"
        ]

        if product.get('features'):
            product_info.append(f"Features: {', '.join(product['features'])}")

        if product.get('tags'):
            product_info.append(f"Tags: {', '.join(product['tags'])}")

        return " ".join(product_info)

    def get_product_embedding(self, product):
        """
        Get embedding for a product. Reuses the in-memory index embedding when available.
        """
        cached = self.product_embeddings.get(product.get('id'))
        if cached is not None:
            return cached
        return self.get_embedding(self._product_text(product))

    def embed_all_products(self, products):
        """
        Create embeddings for all products and build a cosine-similarity search index.
        Products whose embedding fails are left out of the index.
        """
        if not products:
            self.product_embeddings, self.product_ids, self.product_index = {}, [], None
            return

        # One batched API call for the whole catalog. Keeps cold start well under
        # Heroku's 30s request timeout (the dyno's disk cache is wiped on restart).
        raw = self.get_embeddings_batch([self._product_text(p) for p in products])

        product_ids, product_embeddings, embeddings = [], {}, []
        for product, embedding in zip(products, raw):
            if embedding is None:
                continue
            embedding = _normalize(embedding)
            product_ids.append(product['id'])
            product_embeddings[product['id']] = embedding
            embeddings.append(embedding)

        index = None
        if embeddings:
            embeddings_array = np.array(embeddings).astype('float32')
            # Inner product on unit vectors == cosine similarity
            if FAISS_AVAILABLE:
                index = faiss.IndexFlatIP(self.embedding_dim)
                index.add(embeddings_array)
            else:
                index = embeddings_array

        # Swap in together so concurrent readers never see a half-built index
        self.product_embeddings, self.product_ids, self.product_index = product_embeddings, product_ids, index

    def ensure_index(self, all_products):
        """
        Build the index over the full catalog if it hasn't been built yet,
        or rebuild it if the catalog has changed. Safe to call from multiple threads.
        """
        catalog_ids = {p['id'] for p in all_products}
        if self.product_index is not None and catalog_ids == set(self.product_ids):
            return
        with self._index_lock:
            if self.product_index is None or catalog_ids != set(self.product_ids):
                self.embed_all_products(all_products)

    # ------------------------------------------------------------------
    # User profile
    # ------------------------------------------------------------------

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

        preference_texts = []
        for key, value in user_preferences.items():
            if value:
                if isinstance(value, dict):
                    # Nested dictionaries like price range
                    for sub_key, sub_value in value.items():
                        if sub_value is not None:
                            preference_texts.append(f"{key.replace('_', ' ')} {sub_key}: {sub_value}")
                elif isinstance(value, list):
                    # Lists like category preferences
                    preference_texts.append(f"{key.replace('_', ' ')}: {', '.join(map(str, value))}")
                else:
                    preference_texts.append(f"{key.replace('_', ' ')}: {value}")

        if not preference_texts:
            return None

        preferences_text = "User prefers " + ". ".join(preference_texts)
        embedding = self.get_embedding(preferences_text)
        return _normalize(embedding) if embedding is not None else None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def find_similar_products(self, query_embedding, products, top_n=5, exclude_ids=None):
        """
        Find the most similar products to a query embedding.

        The index must already cover the full catalog (see ensure_index); `products`
        restricts which products are eligible to be returned.

        Returns:
        - List of tuples (product, cosine_similarity)
        """
        exclude_ids = set(exclude_ids or [])

        # Snapshot so a concurrent rebuild can't mix old ids with a new index
        index, ids = self.product_index, self.product_ids
        if query_embedding is None or index is None or not ids:
            return []

        eligible = {p['id']: p for p in products if p['id'] not in exclude_ids}
        if not eligible:
            return []

        query = np.array([_normalize(query_embedding)]).astype('float32')

        if FAISS_AVAILABLE and isinstance(index, faiss.Index):
            scores, indices = index.search(query, len(ids))
            ranked = zip(indices[0], scores[0])
        else:
            scores = index @ query[0]
            order = np.argsort(-scores)
            ranked = ((idx, scores[idx]) for idx in order)

        results = []
        for idx, score in ranked:
            if idx < 0 or idx >= len(ids):
                continue
            product = eligible.get(ids[idx])
            if product is not None:
                results.append((product, float(score)))
                if len(results) >= top_n:
                    break

        return results
