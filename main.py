import os
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
from gemini_utility import (
    load_swag_ai_model,
    swag_ai_response,
    swag_ai_embeddings_response,
    generate_caption_with_vision_api  # Use a specialized captioning function
)

# Configure Streamlit page
st.set_page_config(
    page_title="Swag AI",
    page_icon="🧠",
    layout="centered",
)

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        'Swag AI',
        ['ChatBot', 'Image Captioning', 'Embed Text', 'Ask Me Anything'],
        menu_icon='robot',
        icons=['chat-dots-fill', 'image-fill', 'textarea-t', 'patch-question-fill'],
        default_index=0
    )

# Function to translate roles between Swag AI and Streamlit terminology
def translate_role_for_streamlit(user_role):
    return "assistant" if user_role == "model" else user_role

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

# Image Captioning page
if selected == "Image Captioning":
    st.title("📷 Snap Narrate")

    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if st.button("Generate Caption") and uploaded_image is not None:
        try:
            image = Image.open(uploaded_image).resize((512, 512))
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Generate caption using a faster Google Cloud Vision-based utility
            caption = generate_caption_with_vision_api(image)
            st.info(f"Caption: {caption}")
        except Exception as e:
            st.error(f"Error processing the image: {e}")

# Embed Text page
if selected == "Embed Text":
    st.title("🔡 Embed Text")

    user_prompt = st.text_area(label='', placeholder="Enter the text to get embeddings")

    if st.button("Get Response"):
        try:
            response = swag_ai_embeddings_response(user_prompt)
            st.json(response)
        except Exception as e:
            st.error(f"Error: {e}")

# Ask Me Anything page
if selected == "Ask Me Anything":
    st.title("❓ Ask Me a Question")

    user_prompt = st.text_area(label='', placeholder="Ask me anything...")

    if st.button("Get Response"):
        try:
            response = swag_ai_response(user_prompt)
            st.markdown(response)
        except Exception as e:
            st.error(f"Error: {e}")
