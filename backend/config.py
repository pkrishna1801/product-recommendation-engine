import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration settings
config = {
    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
    'MODEL_NAME': os.getenv('MODEL_NAME', 'gpt-3.5-turbo'),
    'MAX_TOKENS': int(os.getenv('MAX_TOKENS', 1000)),
    'TEMPERATURE': float(os.getenv('TEMPERATURE', 0.3)),
    'DATA_PATH': os.getenv('DATA_PATH', 'data/products.json'),

    # Email (welcome mail on registration). Leave SMTP_HOST unset to disable.
    'SMTP_HOST': os.getenv('SMTP_HOST'),
    'SMTP_PORT': int(os.getenv('SMTP_PORT', 587)),
    'SMTP_USERNAME': os.getenv('SMTP_USERNAME'),
    'SMTP_PASSWORD': os.getenv('SMTP_PASSWORD'),
    'MAIL_FROM': os.getenv('MAIL_FROM'),
    'APP_URL': os.getenv('APP_URL'),  # frontend URL linked in emails
}