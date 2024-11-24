import os
import google.generativeai as genai
import json
from io import BytesIO

# Get the working directory
working_directory = os.path.dirname(os.path.abspath(__file__))

# Load API key from config.json
config_file_path = f"{working_directory}/config.json"
config_data = json.load(open(config_file_path))
GOOGLE_API_KEY = config_data["GOOGLE_API_KEY"]

# Configure google.generativeai with API key
genai.configure(api_key=GOOGLE_API_KEY)

# Function to load Pritam AI model
def load_Pritam_ai_model():
    Pritam_ai_model = genai.GenerativeModel("gemini-pro")
    return Pritam_ai_model

# Function for image captioning
def Pritam_Ai_vision_response(prompt, image):
    try:
        # Create the model instance for image captioning
        Pritam_Ai_vision_model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Check if the image is a file-like object (for example, a BytesIO object)
        if isinstance(image, BytesIO):
            # Pass the prompt and image to the model for captioning
            response = Pritam_Ai_vision_model.generate_content([prompt, image])
            return response.text
        else:
            raise ValueError("Image format is not supported. Please upload a valid image.")
    except Exception as e:
        return f"Error generating caption: {str(e)}"
