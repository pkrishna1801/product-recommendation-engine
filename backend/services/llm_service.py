import openai
import json
import re
import time
from functools import lru_cache
from config import config
from services.embedding_service import EmbeddingService

class LLMService:
    """
    Service to handle interactions with the LLM API for product recommendations
    with enhanced prompt engineering and RAG capabilities
    """
    
    def __init__(self):
        """Initialize the LLM service with configuration"""
        openai.api_key = config.get('OPENAI_API_KEY')
        self.model_name = config.get('MODEL_NAME', 'gpt-3.5-turbo')
        self.max_tokens = config.get('MAX_TOKENS', 1000)
        self.temperature = config.get('TEMPERATURE', 0.7)

        self.embedding_service = EmbeddingService()
        self.use_embeddings = True if self.embedding_service else False
        self.cache = {}

    def generate_recommendations(self, user_preferences, browsing_history, all_products, num_recommendations=5):
        """
        Generate personalized product recommendations based on user preferences and browsing history
        
        Parameters:
        - user_preferences (dict): User's stated preferences
        - browsing_history (list): List of product IDs the user has viewed
        - all_products (list): Full product catalog
        - num_recommendations (int): Number of recommendations to generate
        
        Returns:
        - dict: Recommended products with explanations
        """
        start_time = time.time()
        
        # Validate inputs
        if not isinstance(user_preferences, dict):
            user_preferences = {}
        
        if not isinstance(browsing_history, list):
            browsing_history = []
        
        # Get browsed products details
        browsed_products = self._get_browsed_products(browsing_history, all_products)
        
        # Determine the approach based on available data and services.
        # Embeddings work from browsing history, explicit preferences, or both (cold start).
        relevant_products = []
        approach = 'standard'
        if self.use_embeddings and (browsed_products or self._has_preferences(user_preferences)):
            relevant_products = self._find_relevant_products_with_embeddings(
                user_preferences, browsed_products, all_products
            )

        if relevant_products:
            approach = 'rag'
            prompt = self._create_rag_enhanced_prompt(
                user_preferences, browsed_products, relevant_products, num_recommendations
            )
        else:
            # Use basic filtering approach
            relevant_products = self._prefilter_products(
                user_preferences, browsed_products, all_products
            )
            prompt = self._create_enhanced_prompt(
                user_preferences, browsed_products, relevant_products, num_recommendations
            )
        
        # Call the LLM API with retry logic
        try:
            # Only accept IDs we actually offered, and never items the user already browsed
            browsed_ids = {p['id'] for p in browsed_products}
            allowed_ids = {p['id'] for p in relevant_products} - browsed_ids
            recommendations = self._call_llm_with_retry(prompt, all_products, allowed_ids)

            # Add timing information for performance analysis
            process_time = time.time() - start_time
            recommendations['metadata'] = {
                'process_time': process_time,
                'approach': approach,
                'prompt_tokens': len(prompt) // 4,  # Approximate token count
                'relevant_products_count': len(relevant_products)
            }
            
            return recommendations
            
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return {
                "recommendations": [],
                "count": 0,
                "error": f"Failed to generate recommendations: {str(e)}"
            }
    
    def _get_browsed_products(self, browsing_history, all_products):
        """
        Get full details of products in the browsing history
        """
        browsed_products = []
        product_lookup = {product["id"]: product for product in all_products}
        
        for product_id in browsing_history:
            if product_id in product_lookup:
                browsed_products.append(product_lookup[product_id])
        
        return browsed_products
    
    def _find_relevant_products_with_embeddings(self, user_preferences, browsed_products, all_products, max_products=20):
        """
        Use embeddings to find the most relevant products for the user
        """
        if not self.embedding_service:
            return self._prefilter_products(user_preferences, browsed_products, all_products, max_products)

        # Index covers the full catalog; filters are applied at query time
        self.embedding_service.ensure_index(all_products)

        # Apply basic filtering first to reduce the candidate pool
        filtered_products = self._apply_basic_filters(user_preferences, all_products)

        # Get IDs of products already browsed to exclude them
        browsed_ids = [p['id'] for p in browsed_products]

        # Get user interest embedding (combining browsing history and preferences)
        user_embedding = self.embedding_service.get_user_interests_embedding(
            browsed_products, user_preferences
        )

        # No usable embedding (e.g. API failure) -> let caller fall back to rule-based filtering
        if user_embedding is None:
            return []

        # Find similar products using embeddings
        similar_products = self.embedding_service.find_similar_products(
            user_embedding, 
            filtered_products,
            top_n=max_products,
            exclude_ids=browsed_ids
        )
        
        # Extract just the products (without scores)
        relevant_products = [p[0] for p in similar_products]
        
        # Ensure we have enough products by adding some diversity picks if needed
        if len(relevant_products) < max_products // 2:
            # Add some products from popular categories not yet represented
            diversity_products = self._get_diversity_products(
                relevant_products, filtered_products, browsed_ids, max_products - len(relevant_products)
            )
            relevant_products.extend(diversity_products)
        
        return relevant_products[:max_products]
    
    @staticmethod
    def _get_pref_list(user_preferences, *keys):
        """
        Read a list-valued preference, accepting several key names.
        The frontend stores 'category_preferences' / 'brand_preferences';
        older callers may send 'category' / 'brand'.
        """
        for key in keys:
            value = user_preferences.get(key)
            if value:
                return [value] if isinstance(value, str) else list(value)
        return []

    def _has_preferences(self, user_preferences):
        """True if the user has any explicit preference set"""
        return bool(
            self._get_pref_list(user_preferences, 'category_preferences', 'category') or
            self._get_pref_list(user_preferences, 'brand_preferences', 'brand') or
            any(v is not None for v in (user_preferences.get('price_range') or {}).values())
        )

    def _apply_basic_filters(self, user_preferences, products):
        """
        Apply basic filtering based on user preferences
        """
        filtered_products = products.copy()
        price_range = user_preferences.get('price_range') or {}

        # Filter by category if specified
        categories = self._get_pref_list(user_preferences, 'category_preferences', 'category')
        if categories:
            filtered_products = [p for p in filtered_products if p.get('category') in categories]

        # Filter by price range if specified
        if price_range.get('min') is not None:
            filtered_products = [p for p in filtered_products if p.get('price', 0) >= price_range['min']]
        if price_range.get('max') is not None:
            filtered_products = [p for p in filtered_products if p.get('price', 0) <= price_range['max']]

        # Filter by brand if specified
        brands = self._get_pref_list(user_preferences, 'brand_preferences', 'brand')
        if brands:
            filtered_products = [p for p in filtered_products if p.get('brand') in brands]

        # If we've filtered too aggressively, restore some products
        if len(filtered_products) < 10:
            # Just apply the price filter as a fallback
            filtered_products = products.copy()
            if price_range:
                if 'min' in price_range and price_range['min'] is not None:
                    filtered_products = [p for p in filtered_products if p.get('price', 0) >= price_range['min']]
                if 'max' in price_range and price_range['max'] is not None:
                    filtered_products = [p for p in filtered_products if p.get('price', 0) <= price_range['max']]
        
        return filtered_products
    
    def _get_diversity_products(self, current_products, all_filtered_products, exclude_ids, max_count):
        """
        Get diverse products from categories not already represented
        """
        # Get categories already represented
        current_categories = set(p.get('category') for p in current_products if p.get('category'))
        
        # Find products from other categories
        diversity_products = []
        for product in all_filtered_products:
            if (product.get('id') not in exclude_ids and 
                product not in current_products and
                product.get('category') not in current_categories):
                diversity_products.append(product)
                current_categories.add(product.get('category'))
                
                if len(diversity_products) >= max_count:
                    break
        
        # If we still don't have enough, add some random products
        if len(diversity_products) < max_count:
            for product in all_filtered_products:
                if (product.get('id') not in exclude_ids and 
                    product not in current_products and
                    product not in diversity_products):
                    diversity_products.append(product)
                    
                    if len(diversity_products) >= max_count:
                        break
        
        return diversity_products
    
    def _prefilter_products(self, user_preferences, browsed_products, all_products, max_products=20):
        """
        Pre-filter products based on user preferences to reduce token usage
        """
        # Start with basic filtering
        filtered_products = self._apply_basic_filters(user_preferences, all_products)
        
        # Calculate similarity to browsed products for remaining products
        if browsed_products:
            # Extract categories, subcategories, brands, and tags from browsed products
            browsed_categories = set(p.get('category') for p in browsed_products if p.get('category'))
            browsed_subcategories = set(p.get('subcategory') for p in browsed_products if p.get('subcategory'))
            browsed_brands = set(p.get('brand') for p in browsed_products if p.get('brand'))
            browsed_tags = set()
            for p in browsed_products:
                if p.get('tags'):
                    browsed_tags.update(p.get('tags'))
            
            # Score products based on similarity to browsed items
            scored_products = []
            for product in filtered_products:
                # Don't recommend products already browsed
                if product in browsed_products:
                    continue
                    
                score = 0
                
                # Category match
                if product.get('category') in browsed_categories:
                    score += 3
                
                # Subcategory match
                if product.get('subcategory') in browsed_subcategories:
                    score += 2
                
                # Brand match
                if product.get('brand') in browsed_brands:
                    score += 2
                
                # Tag matches
                if product.get('tags'):
                    tag_matches = sum(1 for tag in product.get('tags') if tag in browsed_tags)
                    score += tag_matches
                
                # Add the product with its score
                scored_products.append((product, score))
            
            # Sort by score (highest first)
            scored_products.sort(key=lambda x: x[1], reverse=True)
            
            # Take top scoring products, plus some non-matching ones for diversity
            top_products = [p[0] for p in scored_products[:int(max_products * 0.7)]]
            
            # Add some products not browsed for diversity (up to max_products)
            diversity_products = [p[0] for p in scored_products[int(max_products * 0.7):]]
            
            # Ensure we're not exceeding max_products
            filtered_products = top_products + diversity_products[:max_products - len(top_products)]
        
        # If too few products remain after filtering, loosen criteria
        if len(filtered_products) < 10:
            filtered_products = self._apply_basic_filters(
                {'price_range': user_preferences.get('price_range', {})},
                all_products
            )[:max_products]
        
        # Limit to max_products
        return filtered_products[:max_products]
    
    def _create_rag_enhanced_prompt(self, user_preferences, browsed_products, relevant_products, num_recommendations=5):
        """
        Create an optimized prompt for RAG-based recommendation
        """
        # Create a structured prompt with clear sections
        prompt = """
        You are an expert e-commerce recommendation specialist who understands user preferences deeply.
        Your task is to recommend the most relevant products for a user based on their browsing history and preferences.
        
        # USER BROWSING PATTERNS
        """
        
        # Add browsing history with analysis of patterns
        if browsed_products:
            # Extract and analyze patterns
            categories = {}
            subcategories = {}
            brands = {}
            price_points = []
            tags = {}
            
            for product in browsed_products:
                # Collect stats
                cat = product.get('category')
                if cat:
                    categories[cat] = categories.get(cat, 0) + 1
                
                subcat = product.get('subcategory')
                if subcat:
                    subcategories[subcat] = subcategories.get(subcat, 0) + 1
                
                brand = product.get('brand')
                if brand:
                    brands[brand] = brands.get(brand, 0) + 1
                
                price = product.get('price')
                if price:
                    price_points.append(price)
                
                if 'tags' in product and product['tags']:
                    for tag in product['tags']:
                        tags[tag] = tags.get(tag, 0) + 1
            
            # Add summary of browsing patterns
            prompt += "## BROWSING SUMMARY\n"
            
            # Add categories
            if categories:
                top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
                prompt += f"- User has shown interest in categories: {', '.join([cat for cat, _ in top_categories])}\n"
            
            # Add subcategories
            if subcategories:
                top_subcategories = sorted(subcategories.items(), key=lambda x: x[1], reverse=True)[:3]
                prompt += f"- Specific interest in subcategories: {', '.join([subcat for subcat, _ in top_subcategories])}\n"
            
            # Add brands
            if brands:
                top_brands = sorted(brands.items(), key=lambda x: x[1], reverse=True)[:3]
                prompt += f"- Brand affinity for: {', '.join([brand for brand, _ in top_brands])}\n"
            
            # Add price behavior
            if price_points:
                avg_price = sum(price_points) / len(price_points)
                min_price = min(price_points)
                max_price = max(price_points)
                prompt += f"- Price range: ${min_price:.2f}-${max_price:.2f} (avg: ${avg_price:.2f})\n"
            
            # Add tags/interests
            if tags:
                top_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)[:5]
                prompt += f"- Interest in features/aspects: {', '.join([tag for tag, _ in top_tags])}\n"
            
            # Add individual browsed products
            prompt += "\n## BROWSED PRODUCTS\n"
            for i, product in enumerate(browsed_products):
                prompt += f"{i+1}. {product['name']} (ID: {product['id']})\n"
                prompt += f"   {product.get('category', 'N/A')}/{product.get('subcategory', 'N/A')} | "
                prompt += f"${product.get('price', 0)} | {product.get('brand', 'N/A')}\n"
        else:
            prompt += "No browsing history available.\n"
        
        # Add user preferences section
        prompt += "\n# USER PREFERENCES\n"
        if user_preferences:
            for key, value in user_preferences.items():
                if value:
                    prompt += f"- {key.replace('_', ' ').title()}: {value}\n"
        else:
            prompt += "No specific preferences provided.\n"
        
        # Add available products section in a compact format
        prompt += "\n# CANDIDATE PRODUCTS FOR RECOMMENDATION\n"
        
        for i, product in enumerate(relevant_products):
            # Use a more compact format for products
            prompt += f"Product {i+1}: {product['name']} (ID: {product['id']})\n"
            prompt += f"- Category: {product.get('category', 'N/A')}, Subcategory: {product.get('subcategory', 'N/A')}\n"
            prompt += f"- Brand: {product.get('brand', 'N/A')}, Price: ${product.get('price', 0)}, Rating: {product.get('rating', 'N/A')}\n"
            
            # Add tags which are very informative for matching
            if 'tags' in product and product['tags']:
                prompt += f"- Tags: {', '.join(product['tags'])}\n"
            
            # Add a compact description
            description = product.get('description', '')
            if len(description) > 100:
                description = description[:97] + "..."
            prompt += f"- Description: {description}\n"
            
            # Add a few features for context
            if 'features' in product and product['features']:
                features = product['features'][:3]  # Limit to 3 features
                prompt += f"- Key Features: {', '.join(features)}\n"
            
            prompt += "\n"
        
        # Add detailed instructions for the recommendation task
        prompt += f"""
        # YOUR TASK
        Based on the user's browsing patterns and preferences, recommend {num_recommendations} products from the candidate products list.
        
        ## SELECTION CRITERIA
        1. RELEVANCE: Select products that align with the user's demonstrated interests
        2. DIVERSITY: Include some variety in your recommendations (different categories or styles)
        3. PERSONALIZATION: Consider both explicit preferences and implicit interests from browsing
        4. COHERENCE: Ensure recommendations make sense together as a collection
        
        ## OUTPUT INSTRUCTIONS
        Return your recommendations as valid JSON with this exact structure:
        [
          {{
            "product_id": "product123",
            "relevance_score": 95,
            "explanation": "Specific explanation of why this product matches this user's preferences"
          }},
          ...
        ]
        
        ## QUALITY GUIDELINES FOR EXPLANATIONS
        - GOOD EXPLANATION: "This running shoe matches your interest in athletic footwear with responsive cushioning, similar to the Ultra-Comfort model you browsed, while offering better water resistance for trail running."
        - BAD EXPLANATION: "This is a good product that you might like based on your browsing."
        
        Your explanations should be specific, personalized, and reference actual features of both the product and the user's interests.
        """
        
        return prompt
    
    def _create_enhanced_prompt(self, user_preferences, browsed_products, relevant_products, num_recommendations=5):
        """
        Create a structured prompt for the standard recommendation approach
        """
        # Create a structured prompt with clear sections
        prompt = """
        You are an expert e-commerce recommendation specialist who deeply understands customer preferences.
        Your task is to recommend the most relevant products for a user based on their browsing history and preferences.
        
        # USER INFORMATION
        """
        
        # Add browsing history with detailed information
        prompt += "## BROWSING HISTORY\n"
        if browsed_products:
            for i, product in enumerate(browsed_products):
                prompt += f"{i+1}. {product['name']} (ID: {product['id']})\n"
                prompt += f"   Category: {product.get('category', 'N/A')}, Subcategory: {product.get('subcategory', 'N/A')}\n"
                prompt += f"   Brand: {product.get('brand', 'N/A')}, Price: ${product.get('price', 0)}\n"
                
                if 'tags' in product and product['tags']:
                    prompt += f"   Tags: {', '.join(product['tags'])}\n"
                
                if 'features' in product and product['features']:
                    prompt += f"   Features: {', '.join(product['features'][:3])}\n"
                
                prompt += "\n"
        else:
            prompt += "No browsing history available.\n\n"
        
        # Add user preferences
        prompt += "## USER PREFERENCES\n"
        if user_preferences:
            for key, value in user_preferences.items():
                if value:
                    prompt += f"- {key.replace('_', ' ').title()}: {value}\n"
        else:
            prompt += "No specific preferences provided.\n"
        
        # Extract implicit preferences if browsing history exists
        if browsed_products:
            prompt += "\n## IMPLICIT PREFERENCES (Based on browsing history)\n"
            
            # Extract and count categories
            categories = {}
            subcategories = {}
            brands = {}
            price_points = []
            tags = {}
            
            for product in browsed_products:
                # Count categories
                cat = product.get('category')
                if cat:
                    categories[cat] = categories.get(cat, 0) + 1
                
                # Count subcategories
                subcat = product.get('subcategory')
                if subcat:
                    subcategories[subcat] = subcategories.get(subcat, 0) + 1
                
                # Count brands
                brand = product.get('brand')
                if brand:
                    brands[brand] = brands.get(brand, 0) + 1
                
                # Track price points
                price = product.get('price')
                if price:
                    price_points.append(price)
                
                # Count tags
                if 'tags' in product and product['tags']:
                    for tag in product['tags']:
                        tags[tag] = tags.get(tag, 0) + 1
            
            # Add observed categories
            if categories:
                top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
                prompt += f"- Frequently browsed categories: {', '.join([f'{cat} ({count})' for cat, count in top_categories])}\n"
            
            # Add observed subcategories
            if subcategories:
                top_subcategories = sorted(subcategories.items(), key=lambda x: x[1], reverse=True)[:3]
                prompt += f"- Frequently browsed subcategories: {', '.join([f'{subcat} ({count})' for subcat, count in top_subcategories])}\n"
            
            # Add observed brands
            if brands:
                top_brands = sorted(brands.items(), key=lambda x: x[1], reverse=True)[:3]
                prompt += f"- Preferred brands: {', '.join([f'{brand} ({count})' for brand, count in top_brands])}\n"
            
            # Add price range
            if price_points:
                avg_price = sum(price_points) / len(price_points)
                min_price = min(price_points)
                max_price = max(price_points)
                prompt += f"- Price behavior: Avg=${avg_price:.2f}, Range=${min_price:.2f}-${max_price:.2f}\n"
            
            # Add observed tags/interests
            if tags:
                top_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)[:5]
                prompt += f"- Interest areas: {', '.join([f'{tag} ({count})' for tag, count in top_tags])}\n"
        
        # Add available products section
        prompt += "\n# CANDIDATE PRODUCTS FOR RECOMMENDATION\n"
        
        for i, product in enumerate(relevant_products):
            prompt += f"Product {i+1}: {product['name']} (ID: {product['id']})\n"
            prompt += f"- Category: {product.get('category', 'N/A')}, Subcategory: {product.get('subcategory', 'N/A')}\n"
            prompt += f"- Brand: {product.get('brand', 'N/A')}, Price: ${product.get('price', 0)}\n"
            
            # Add rating if available
            if 'rating' in product:
                prompt += f"- Rating: {product['rating']}\n"
            
            # Add description (truncated if too long)
            if 'description' in product:
                description = product['description']
                if len(description) > 150:
                    description = description[:147] + "..."
                prompt += f"- Description: {description}\n"
            
            # Add tags
            if 'tags' in product and product['tags']:
                prompt += f"- Tags: {', '.join(product['tags'])}\n"
            
            # Add features (limited to 3 for brevity)
            if 'features' in product and product['features']:
                prompt += f"- Features: {', '.join(product['features'][:3])}\n"
            
            prompt += "\n"
        
        # Add detailed instructions for the recommendations
        prompt += f"""
        # YOUR TASK
        Based on the user's browsing history and preferences, recommend {num_recommendations} products from the candidate products list.
        
        ## RECOMMENDATION CRITERIA
        1. RELEVANCE: Select products that closely match the user's interests
        2. DIVERSITY: Include some variety in your recommendations
        3. PERSONALIZATION: Consider both explicit preferences and implicit interests
        4. COHERENCE: Ensure your selections make sense as a collection
        
        ## REQUIRED OUTPUT FORMAT
        Return your recommendations as a valid JSON array with this exact structure:
        [
          {{
            "product_id": "product123",
            "relevance_score": 95,
            "explanation": "Detailed explanation of why this product matches the user's preferences"
          }},
          ...
        ]
        
        ## QUALITY GUIDELINES FOR EXPLANATIONS
        
        GOOD EXPLANATION: "This premium wireless headphone aligns with your interest in audio equipment as shown by your browsing of the SoundWave portable speaker. The noise cancellation feature addresses the preference for high-quality sound you've demonstrated, while the 30-hour battery life offers extended use for your travel needs."
        
        BAD EXPLANATION: "This product seems like something you would like based on your browsing history."
        
        Make sure explanations are specific to both the product and the user's demonstrated interests.
        """
        
        return prompt
    
    def _call_llm_with_retry(self, prompt, all_products, allowed_ids=None, max_retries=2):
        """
        Call the LLM API with retry logic
        """
        retries = 0
        
        while retries <= max_retries:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert eCommerce product recommendation specialist."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
                # Parse the LLM response to extract recommendations
                recommendations = self._parse_recommendation_response(
                    response.choices[0].message.content,
                    all_products,
                    allowed_ids
                )
                
                # If we successfully parsed recommendations, return them
                if recommendations.get('recommendations'):
                    return recommendations
                
                # If parsing failed, increment retries and try again with a more explicit prompt
                retries += 1
                if retries <= max_retries:
                    # Add more explicit instructions about the JSON format
                    prompt += """
                    IMPORTANT: Make sure your response contains ONLY a valid JSON array with the exact structure shown above.
                    Don't include any explanation text outside of the JSON array. The response should start with '[' and end with ']'.
                    """
                else:
                    # Return whatever we got on the last attempt
                    return recommendations
                
            except Exception as e:
                print(f"Error on attempt {retries+1}: {str(e)}")
                retries += 1
                if retries > max_retries:
                    raise
                time.sleep(1)  # Short delay before retry
    
    def _parse_recommendation_response(self, llm_response, all_products, allowed_ids=None):
        """
        Parse the LLM response to extract product recommendations.
        If allowed_ids is given, IDs outside it (hallucinated, not offered as
        candidates, or already browsed) are dropped, as are duplicates.
        """
        json_str = ''
        try:
            # Find JSON content in the response using regex
            json_pattern = r'\[\s*{.*}\s*\]'
            match = re.search(json_pattern, llm_response, re.DOTALL)
            
            if match:
                json_str = match.group(0)
            else:
                # Try another approach - look for array brackets
                start_idx = llm_response.find('[')
                end_idx = llm_response.rfind(']') + 1
                
                if start_idx == -1 or end_idx <= 0:
                    return {
                        "recommendations": [],
                        "error": "Could not parse recommendations from LLM response"
                    }
                
                json_str = llm_response[start_idx:end_idx]
            
            # Clean up the JSON string
            json_str = re.sub(r'```json|```', '', json_str)
            
            # Parse the JSON
            rec_data = json.loads(json_str)
            
            # Create a lookup dictionary for faster product matching
            product_lookup = {product['id']: product for product in all_products}
            
            # Enrich recommendations with full product details
            recommendations = []
            seen_ids = set()
            for rec in rec_data:
                product_id = rec.get('product_id')

                if product_id in seen_ids:
                    continue
                if allowed_ids is not None and product_id not in allowed_ids:
                    continue
                seen_ids.add(product_id)

                # Find the product in the lookup dictionary
                product_details = product_lookup.get(product_id)
                
                if product_details:
                    # Normalize the relevance score to a 0-1 range
                    relevance_score = rec.get('relevance_score', 50)
                    
                    # Handle percentage or float representation
                    if isinstance(relevance_score, str):
                        relevance_score = relevance_score.strip('%')
                        try:
                            relevance_score = float(relevance_score)
                        except ValueError:
                            relevance_score = 50
                    
                    # Convert from 0-100 scale to 0-1 scale if needed
                    if relevance_score > 1:
                        relevance_score = relevance_score / 100.0
                    
                    # Ensure score is between 0 and 1
                    relevance_score = max(0, min(1, relevance_score))
                    
                    recommendations.append({
                        "product": product_details,
                        "explanation": rec.get('explanation', ''),
                        "relevance_score": relevance_score
                    })
            
            # Sort recommendations by relevance score (highest first)
            recommendations.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return {
                "recommendations": recommendations,
                "count": len(recommendations)
            }
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Problematic JSON string: {json_str}")
            return {
                "recommendations": [],
                "error": f"Failed to parse JSON from LLM response: {str(e)}"
            }
        except Exception as e:
            print(f"Error parsing LLM response: {str(e)}")
            return {
                "recommendations": [],
                "error": f"Failed to parse recommendations: {str(e)}"
            }