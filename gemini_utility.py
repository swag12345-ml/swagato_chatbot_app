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

def load_swag_ai_model():
    swag_ai_model = genai.GenerativeModel("gemini-pro")
    return swag_ai_model

# Get response from Swag AI Vision model - image/text to text
def swag_ai_vision_response(prompt, image):
    try:
        # Convert the image to a compatible format for the API
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save the image to a BytesIO object as JPEG format
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)
        
        # Prepare the request payload with the prompt and image
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content([prompt, img_byte_arr])
        result = response.text
        return result

    except Exception as e:
        return f"Error generating caption: {e}"

# Get response from embeddings model - text to embeddings
def swag_ai_embeddings_response(input_text):
    embedding_model = "models/embedding-001"
    embedding = genai.embed_content(model=embedding_model,
                                    content=input_text,
                                    task_type="retrieval_document")
    embedding_list = embedding["embedding"]
    return embedding_list

# Get response from Swag AI model - text to text
def swag_ai_response(user_prompt):
    swag_ai_model = genai.GenerativeModel("gemini-pro")
    response = swag_ai_model.generate_content(user_prompt)
    result = response.text
    return result
