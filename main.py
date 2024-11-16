import os
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
from gemini_utility import (
    load_swag_ai_model,
    swag_ai_response,
    swag_ai_vision_response,
    swag_ai_embeddings_response,
)
import asyncio
import time
import hashlib

working_dir = os.path.dirname(os.path.abspath(_file_))

st.set_page_config(
    page_title="Swag AI",
    page_icon="🧠",
    layout="centered",
)

with st.sidebar:
    selected = option_menu(
        'Swag AI',
        ['ChatBot', 'Image Captioning', 'Embed Text', 'Ask Me Anything'],
        menu_icon='robot',
        icons=['chat-dots-fill', 'image-fill', 'textarea-t', 'patch-question-fill'],
        default_index=0,
    )

def translate_role_for_streamlit(user_role):
    return "assistant" if user_role == "model" else user_role

# Function to cache image captions
@st.cache_data
def get_cached_caption(image_data, prompt):
    image_hash = hashlib.md5(image_data).hexdigest()
    return swag_ai_vision_response(prompt, image_data), image_hash

# Chatbot page
if selected == 'ChatBot':
    model = load_swag_ai_model()

    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    st.title("🤖 ChatBot")

    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    user_prompt = st.chat_input("Ask Swag AI...")
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        swag_ai_response = st.session_state.chat_session.send_message(user_prompt)
        with st.chat_message("assistant"):
            st.markdown(swag_ai_response.text)

# Image captioning page
if selected == "Image Captioning":
    st.title("📷 Snap Narrate")
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if st.button("Generate Caption") and uploaded_image:
        try:
            image = Image.open(uploaded_image).convert("RGB")
            resized_img = image.resize((256, 256))  # Resize for faster processing

            default_prompt = "Write a short and meaningful caption for this image."
            image_data = resized_img.tobytes()

            # Time profiling
            start_time = time.time()
            caption, _ = get_cached_caption(image_data, default_prompt)
            end_time = time.time()

            col1, col2 = st.columns(2)
            with col1:
                st.image(image.resize((800, 500)))  # Show original size for clarity
            with col2:
                st.info(caption)
                st.write(f"Caption generated in {end_time - start_time:.2f} seconds.")
        except Exception as e:
            st.error(f"Error processing the image: {e}")

# Text embedding page
if selected == "Embed Text":
    st.title("🔡 Embed Text")
    user_prompt = st.text_area(label='', placeholder="Enter the text to get embeddings")
    if st.button("Get Response"):
        response = swag_ai_embeddings_response(user_prompt)
        st.markdown(response)

# Ask me anything page
if selected == "Ask Me Anything":
    st.title("❓ Ask Me a Question")
    user_prompt = st.text_area(label='', placeholder="Ask me anything...")
    if st.button("Get Response"):
        response = swag_ai_response(user_prompt)
        st.markdown(response)
