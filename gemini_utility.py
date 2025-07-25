import os
import json
import numpy as np
from PIL import Image
import google.generativeai as genai

# ✅ Load configuration from config.json
working_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(working_dir, "config.json")

try:
    with open(config_file_path, "r") as f:
        config_data = json.load(f)
    GOOGLE_API_KEY = config_data.get("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is missing in config.json")
except FileNotFoundError:
    raise FileNotFoundError(f"config.json not found at: {config_file_path}")
except json.JSONDecodeError:
    raise ValueError("config.json is not a valid JSON file")

# ✅ Configure the Gemini client
genai.configure(api_key=GOOGLE_API_KEY)

# ✅ Load the chat model (Gemini Pro)
def load_swag_ai_model():
    return genai.GenerativeModel("gemini-pro")

# ✅ Vision-based response using Gemini 1.5 Flash
def swag_ai_vision_response(prompt, image):
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Ensure image is converted to a format Gemini understands (RGB NumPy array)
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    try:
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"❌ Error during image response generation: {str(e)}"

# ✅ Generate embeddings from text using Gemini Embeddings API
def swag_ai_embeddings_response(input_text):
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=input_text,
            task_type="retrieval_document"
        )
        return result["embedding"]
    except Exception as e:
        return f"❌ Error during embedding generation: {str(e)}"

# ✅ Generate a text response using Gemini Pro
def swag_ai_response(user_prompt):
    model = genai.GenerativeModel("gemini-pro")

    try:
        response = model.generate_content(user_prompt)
        return response.text
    except Exception as e:
        return f"❌ Error during text generation: {str(e)}"

