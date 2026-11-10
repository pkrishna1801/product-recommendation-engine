from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import threading
from functools import wraps
from services.llm_service import LLMService
from services.product_service import ProductService
from services.user_service import UserService
from services.email_service import EmailService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Initialize services
product_service = ProductService()
llm_service = LLMService()
user_service = UserService()
email_service = EmailService()

# Warm the embedding index in the background at boot so the first
# recommendation request after a dyno restart doesn't hit Heroku's 30s timeout
def _warm_embedding_index():
    try:
        llm_service.embedding_service.ensure_index(product_service.get_all_products())
        logger.info("Embedding index warmed")
    except Exception as e:
        logger.error(f"Failed to warm embedding index: {str(e)}")

threading.Thread(target=_warm_embedding_index, daemon=True).start()

# Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Check if token is in headers
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            token_parts = auth_header.split()
            
            if len(token_parts) == 2 and token_parts[0].lower() == 'bearer':
                token = token_parts[1]
        
        if not token:
            return jsonify({'success': False, 'message': 'Token is missing'}), 401
        
        # Verify token
        result = user_service.verify_token(token)
        if not result['valid']:
            return jsonify({'success': False, 'message': result['error']}), 401
        
        # Add user to request
        request.user = result['user']
        request.user_id = result['user_id']
        
        return f(*args, **kwargs)
    
    return decorated

# Authentication routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Check required fields
        required_fields = ['username', 'email', 'password']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400
        
        result = user_service.register_user(
            username=data['username'],
            email=data['email'],
            password=data['password']
        )
        
        if result['success']:
            # Fire-and-forget: a mail failure never fails registration
            email_service.send_welcome_email(result['user']['email'], result['user']['username'])
            return jsonify(result), 201
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Error in register: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Authenticate a user"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Check required fields
        required_fields = ['username', 'password']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400
        
        result = user_service.login_user(
            username=data['username'],
            password=data['password']
        )
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 401
            
    except Exception as e:
        logger.error(f"Error in login: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/auth/verify', methods=['GET'])
@token_required
def verify_token():
    """Verify an authentication token"""
    return jsonify({
        'success': True,
        'user': request.user
    })

# Profile and preferences routes
@app.route('/api/user/profile', methods=['GET'])
@token_required
def get_user_profile():
    """Get the current user's profile"""
    try:
        user = user_service.get_user_by_id(request.user_id)
        if user:
            return jsonify({
                'success': True,
                'user': user
            })
        else:
            return jsonify({
                'success': False,
                'error': 'User not found'
            }), 404
            
    except Exception as e:
        logger.error(f"Error in get_user_profile: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/user/preferences', methods=['GET'])
@token_required
def get_user_preferences():
    """Get the current user's preferences"""
    try:
        user = user_service.get_user_by_id(request.user_id)
        if user:
            return jsonify({
                'success': True,
                'preferences': user.get('preferences', {})
            })
        else:
            return jsonify({
                'success': False,
                'error': 'User not found'
            }), 404
            
    except Exception as e:
        logger.error(f"Error in get_user_preferences: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/user/preferences', methods=['POST'])
@token_required
def save_user_preferences():
    """
    Save user preferences
    """
    try:
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'error': 'No preference data provided'
            }), 400
        
        result = user_service.update_user_preferences(request.user_id, data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in save_user_preferences: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Browsing history routes
@app.route('/api/user/browsing-history', methods=['GET'])
@token_required
def get_user_browsing_history():
    """Get the current user's browsing history"""
    try:
        result = user_service.get_browsing_history(request.user_id)
        return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error in get_user_browsing_history: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/user/browsing-history', methods=['POST'])
@token_required
def add_to_browsing_history():
    """Add a product to the user's browsing history"""
    try:
        data = request.json
        if not data or 'product_id' not in data:
            return jsonify({
                'success': False,
                'error': 'Product ID is required'
            }), 400
        
        result = user_service.save_browsing_history(request.user_id, data['product_id'])
        return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error in add_to_browsing_history: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/user/browsing-history', methods=['DELETE'])
@token_required
def clear_browsing_history():
    """Clear the user's browsing history"""
    try:
        result = user_service.clear_browsing_history(request.user_id)
        return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error in clear_browsing_history: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Product catalog routes
@app.route('/api/products', methods=['GET'])
def get_products():
    """
    Endpoint to retrieve available products.
    Can be filtered by category, price range, etc.
    """
    try:
        # Get query parameters for filtering
        category = request.args.get('category')
        subcategory = request.args.get('subcategory')
        min_price = request.args.get('min_price')
        max_price = request.args.get('max_price')
        brand = request.args.get('brand')
        tags = request.args.getlist('tags')
        min_rating = request.args.get('min_rating')
        search_query = request.args.get('q')
        limit = request.args.get('limit')
        sort_by = request.args.get('sort_by', 'name')
        sort_order = request.args.get('sort_order', 'asc')
        
        # Convert string parameters to correct types
        min_price = float(min_price) if min_price else None
        max_price = float(max_price) if max_price else None
        min_rating = float(min_rating) if min_rating else None
        limit = int(limit) if limit else None
        
        # Support comma-separated categories and brands for multi-select filters
        if category and ',' in category:
            category = category.split(',')
        
        if brand and ',' in brand:
            brand = brand.split(',')
        
        # Get filtered products
        products = product_service.get_products(
            category=category,
            subcategory=subcategory,
            min_price=min_price,
            max_price=max_price,
            brand=brand,
            tags=tags if tags else None,
            min_rating=min_rating,
            search_query=search_query,
            limit=limit,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        return jsonify({
            'success': True,
            'products': products,
            'count': len(products)
        })
        
    except Exception as e:
        logger.error(f"Error in get_products: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/products/<product_id>', methods=['GET'])
def get_product(product_id):
    """Endpoint to get details for a specific product"""
    try:
        product = product_service.get_product_by_id(product_id)
        
        if not product:
            return jsonify({
                'success': False,
                'error': f'Product with ID {product_id} not found'
            }), 404
            
        return jsonify({
            'success': True,
            'product': product
        })
    except Exception as e:
        logger.error(f"Error in get_product: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/products/categories', methods=['GET'])
def get_categories():
    """Endpoint to get all available product categories"""
    try:
        categories = product_service.get_categories()
        return jsonify({
            'success': True,
            'categories': categories
        })
    except Exception as e:
        logger.error(f"Error in get_categories: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/products/brands', methods=['GET'])
def get_brands():
    """Endpoint to get all available product brands"""
    try:
        brands = product_service.get_brands()
        return jsonify({
            'success': True,
            'brands': brands
        })
    except Exception as e:
        logger.error(f"Error in get_brands: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/products/price-range', methods=['GET'])
def get_price_range():
    """Endpoint to get the min and max product prices"""
    try:
        price_range = product_service.get_price_range()
        return jsonify({
            'success': True,
            'price_range': price_range
        })
    except Exception as e:
        logger.error(f"Error in get_price_range: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Recommendations endpoint
@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """
    Endpoint to generate personalized product recommendations
    based on user preferences and browsing history.
    This endpoint works both with and without authentication.
    """
    try:
        # Get request data
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'error': 'No request data provided'
            }), 400
        
        # Check if user is authenticated
        user_id = None
        token = None
        
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            token_parts = auth_header.split()
            
            if len(token_parts) == 2 and token_parts[0].lower() == 'bearer':
                token = token_parts[1]
        
        if token:
            result = user_service.verify_token(token)
            if result['valid']:
                user_id = result['user_id']
        
        # Extract user preferences and browsing history
        user_preferences = data.get('preferences', {})
        browsing_history = data.get('browsing_history', [])
        
        # If authenticated, supplement with stored data
        if user_id:
            user = user_service.get_user_by_id(user_id)
            if user and 'preferences' in user:
                # Merge preferences (explicit preferences take precedence)
                stored_prefs = user.get('preferences', {})
                for key, value in stored_prefs.items():
                    if key not in user_preferences:
                        user_preferences[key] = value
            
            # Get stored browsing history if current history is empty
            if not browsing_history:
                history_result = user_service.get_browsing_history(user_id)
                if history_result['success']:
                    browsing_history = history_result['browsing_history']
        
        # Validate user inputs
        if not isinstance(browsing_history, list):
            return jsonify({
                'success': False,
                'error': 'Browsing history must be a list of product IDs'
            }), 400
        
        # Get all products
        all_products = product_service.get_all_products()
        
        # Generate recommendations
        recommendations = llm_service.generate_recommendations(
            user_preferences=user_preferences,
            browsing_history=browsing_history,
            all_products=all_products
        )
        
        # Return the recommendations
        return jsonify({
            'success': True,
            'recommendations': recommendations.get('recommendations', []),
            'count': recommendations.get('count', 0),
            'metadata': recommendations.get('metadata', {})
        })
        
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Start the Flask application
    app.run(host='0.0.0.0', port=5000, debug=True)