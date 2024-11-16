import os
import json
from PIL import Image
import google.generativeai as genai

working_dir = os.path.dirname(os.path.abspath(_file_))

# Load API key from config
config_file_path = f"{working_dir}/config.json"
with open(config_file_path) as f:
    config_data = json.load(f)

GOOGLE_API_KEY = config_data["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

def load_swag_ai_model():
    return genai.GenerativeModel("gemini-pro")

# Vision model for image captioning
def swag_ai_vision_response(prompt, image_data):
    swag_ai_vision_model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        # Passing preprocessed image data directly
        response = swag_ai_vision_model.generate_content([prompt, image_data])
        return response.text
    except Exception as e:
        raise RuntimeError(f"Failed to generate image caption: {e}")

# Embedding model for text
def swag_ai_embeddings_response(input_text):
    embedding_model = "models/embedding-001"
    embedding = genai.embed_content(
        model=embedding_model,
        content=input_text,
        task_type="retrieval_document",
    )
    return embedding["embedding"]

# Text-to-text model
def swag_ai_response(user_prompt):
    swag_ai_model = genai.GenerativeModel("gemini-pro")
    try:
        response = swag_ai_model.generate_content(user_prompt)
        return response.text
    except Exception as e:
        raise RuntimeError(f"Failed to generate response: {e}")
