import os
import json
from PIL import Image
import google.generativeai as genai

# Working directory path
working_dir = os.path.dirname(os.path.abspath(__file__))

# Path of config_data file
config_file_path = os.path.join(working_dir, "config.json")
with open(config_file_path) as f:
    config_data = json.load(f)

# Load the Google API Key
GOOGLE_API_KEY = config_data["GOOGLE_API_KEY"]

# Configure google.generativeai with API key
genai.configure(api_key=GOOGLE_API_KEY)

# Load Swag AI model
def load_swag_ai_model():
    swag_ai_model = genai.GenerativeModel("gemini-pro")
    return swag_ai_model

# Vision response for image captioning
def swag_ai_vision_response(prompt, image):
    swag_ai_vision_model = genai.GenerativeModel("gemini-1.5-flash")
    response = swag_ai_vision_model.generate_content([prompt, image])
    return response.text

# Text embeddings response
def swag_ai_embeddings_response(input_text):
    embedding_model = "models/embedding-001"
    embedding = genai.embed_content(
        model=embedding_model,
        content=input_text,
        task_type="retrieval_document"
    )
    return embedding["embedding"]

# General text-to-text response
def swag_ai_response(user_prompt):
    swag_ai_model = genai.GenerativeModel("gemini-pro")
    response = swag_ai_model.generate_content(user_prompt)
    return response.text

