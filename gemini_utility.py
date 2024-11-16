import os
import json
from PIL import Image
from google.cloud import vision
import google.generativeai as genai

# Load configuration
working_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(working_dir, "config.json")
with open(config_file_path) as f:
    config_data = json.load(f)

# Load the API key
GOOGLE_API_KEY = config_data["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

# Load Swag AI model
def load_swag_ai_model():
    return genai.GenerativeModel("gemini-pro")

# Faster captioning using Google Cloud Vision
def generate_caption_with_vision_api(image):
    client = vision.ImageAnnotatorClient()

    # Convert PIL Image to bytes
    image_byte_array = io.BytesIO()
    image.save(image_byte_array, format="PNG")
    content = image_byte_array.getvalue()

    # Prepare vision API request
    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    
    # Extract labels and form a caption
    labels = [label.description for label in response.label_annotations]
    return " ".join(labels[:5])  # Return top 5 labels as a caption

# Embeddings response
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


