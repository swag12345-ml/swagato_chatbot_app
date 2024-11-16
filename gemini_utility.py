import os
import json
from PIL import Image
import google.generativeai as genai
from io import BytesIO

# Working directory path
working_dir = os.path.dirname(os.path.abspath(__file__))

# Path of config_data file
config_file_path = f"{working_dir}/config.json"
with open(config_file_path) as f:
    config_data = json.load(f)

# Loading the GOOGLE_API_KEY
GOOGLE_API_KEY = config_data["GOOGLE_API_KEY"]

# Configuring google.generativeai with API key
genai.configure(api_key=GOOGLE_API_KEY)

# Cache model to avoid loading it multiple times
swag_ai_model = genai.GenerativeModel("gemini-pro")
swag_ai_vision_model = genai.GenerativeModel("gemini-1.5-flash")

# Load Swag AI model once
def load_swag_ai_model():
    return swag_ai_model

# Optimized response for image captioning - image/text to text
def swag_ai_vision_response(prompt, image):
    try:
        # Resize and compress image before sending to the model
        image = preprocess_image(image)
        response = swag_ai_vision_model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Error in image captioning: {str(e)}"

# Preprocess the image to ensure it's in a proper format and size
def preprocess_image(image):
    # Convert RGBA to RGB (remove alpha channel)
    if image.mode == "RGBA":
        image = image.convert("RGB")
    
    # Resize the image to a reasonable size (e.g., 800x500)
    image = image.resize((800, 500))
    
    # Save the image to a byte stream and return it
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    
    return img_byte_arr

# Get response from embeddings model - text to embeddings
def swag_ai_embeddings_response(input_text):
    try:
        embedding_model = "models/embedding-001"
        embedding = genai.embed_content(model=embedding_model,
                                        content=input_text,
                                        task_type="retrieval_document")
        return embedding["embedding"]
    except Exception as e:
        return f"Error in embeddings generation: {str(e)}"

# Get response from Swag AI model - text to text
def swag_ai_response(user_prompt):
    try:
        response = swag_ai_model.generate_content(user_prompt)
        return response.text
    except Exception as e:
        return f"Error in generating response: {str(e)}"
