import os
import json
from PIL import Image
import google.generativeai as genai

working_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = f"{working_dir}/config.json"
with open(config_file_path) as f:
    config_data = json.load(f)

GOOGLE_API_KEY = config_data["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

def load_swag_ai_model():
    return genai.GenerativeModel("gemini-pro")

def swag_ai_vision_response(prompt, image):
    swag_ai_vision_model = genai.GenerativeModel("gemini-1.5-flash")
    if isinstance(image, Image.Image):
        image_bytes = image.tobytes()
    else:
        raise ValueError("Invalid image format")
    response = swag_ai_vision_model.generate_content([prompt, image_bytes])
    return response.text

def swag_ai_embeddings_response(input_text):
    embedding_model = "models/embedding-001"
    embedding = genai.embed_content(model=embedding_model, content=input_text, task_type="retrieval_document")
    return embedding["embedding"]

def swag_ai_response(user_prompt):
    swag_ai_model = genai.GenerativeModel("gemini-pro")
    response = swag_ai_model.generate_content(user_prompt)
    return response.text
