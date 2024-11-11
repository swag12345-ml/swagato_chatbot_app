import os
import json
import numpy as np
import google.generativeai as genai
import fitz  # PyMuPDF for PDF processing

working_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = f"{working_dir}/config.json"
with open(config_file_path) as f:
    config_data = json.load(f)

GOOGLE_API_KEY = config_data["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

def load_swag_ai_model():
    return genai.GenerativeModel("gemini-pro")

# Image captioning
def swag_ai_vision_response(prompt, image):
    swag_ai_vision_model = genai.GenerativeModel("gemini-1.5-flash")
    response = swag_ai_vision_model.generate_content([prompt, image])
    return response.text

# Text embeddings
def swag_ai_embeddings_response(input_text):
    embedding = genai.embed_content(
        model="models/embedding-001",
        content=input_text,
        task_type="retrieval_document"
    )
    return embedding["embedding"]

# Text response generation
def swag_ai_response(user_prompt):
    swag_ai_model = load_swag_ai_model()
    response = swag_ai_model.generate_content(user_prompt)
    return response.text

# PDF text extraction
def extract_text_from_pdf(uploaded_pdf):
    with fitz.open(stream=uploaded_pdf.read(), filetype="pdf") as pdf:
        text = ""
        for page_num in range(pdf.page_count):
            text += pdf[page_num].get_text("text")
    return text

# Response based on PDF content embeddings
def swag_ai_pdf_response(user_question, pdf_embeddings):
    question_embedding = swag_ai_embeddings_response(user_question)
    similarities = [np.dot(question_embedding, section_embedding) for section_embedding in pdf_embeddings]
    best_match_index = np.argmax(similarities)
    best_match_text = pdf_embeddings[best_match_index]
    response = swag_ai_response(f"Based on the PDF content: {best_match_text}. {user_question}")
    return response
