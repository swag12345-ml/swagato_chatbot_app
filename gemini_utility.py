import os
import json
import base64
from PIL import Image
import google.generativeai as genai
from io import BytesIO

# Load configuration
working_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(working_dir, "config.json")

with open(config_file_path) as f:
    config_data = json.load(f)

GOOGLE_API_KEY = config_data["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

def load_swag_ai_model():
    return genai.GenerativeModel("gemini-pro")

def encode_image(image):
    """Convert PIL image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # Convert image to PNG format
    return base64.b64encode(buffered.getvalue()).decode()

def swag_ai_vision_response(prompt, image):
    """Send prompt and image to Gemini Vision API"""
    swag_ai_vision_model = genai.GenerativeModel("gemini-1.5-flash")
    encoded_image = encode_image(image)  # Convert image to base64
    response = swag_ai_vision_model.generate_content([prompt, encoded_image])
    
    return response.text if response.text else "No response received."

def swag_ai_embeddings_response(input_text):
    """Get text embeddings from Gemini API"""
    embedding_model = "models/embedding-001"
    embedding = genai.embed_content(model=embedding_model, content=input_text, task_type="retrieval_document")
    return embedding["embedding"]

def swag_ai_response(user_prompt):
    """Get text response from Gemini AI"""
    swag_ai_model = genai.GenerativeModel("gemini-pro")
    response = swag_ai_model.generate_content(user_prompt)
    return response.text
