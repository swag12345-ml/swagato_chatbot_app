import os
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
from gemini_utility import (load_swag_ai_model,
                             swag_ai_response,
                             swag_ai_vision_response,
                             swag_ai_embeddings_response)
import threading
from io import BytesIO

working_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Swag AI",
    page_icon="🧠",
    layout="centered",
)

with st.sidebar:
    selected = option_menu('Swag AI',
                           ['ChatBot',
                            'Image Captioning',
                            'Embed Text',
                            'Ask Me Anything'],
                           menu_icon='robot', icons=['chat-dots-fill', 'image-fill', 'textarea-t', 'patch-question-fill'],
                           default_index=0
                           )

# Function to translate roles between Swag AI and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

# Function to resize and compress image
def resize_image(image, target_size=(800, 500)):
    return image.resize(target_size)

def compress_image(image, quality=85):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="JPEG", quality=quality)
    img_byte_arr.seek(0)
    return img_byte_arr

# Chatbot page
if selected == 'ChatBot':
    model = load_swag_ai_model()

    # Initialize chat session in Streamlit if not already present
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    # Display the chatbot's title on the page
    st.title("🤖 ChatBot")

    # Display the chat history
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    # Input field for user's message
    user_prompt = st.chat_input("Ask Swag AI...")
    if user_prompt:
        # Add user's message to chat and display it
        st.chat_message("user").markdown(user_prompt)

        # Send user's message to Swag AI and get the response
        swag_ai_response = st.session_state.chat_session.send_message(user_prompt)

        # Display Swag AI's response
        with st.chat_message("assistant"):
            st.markdown(swag_ai_response.text)

# Image captioning page
if selected == "Image Captioning":
    st.title("📷 Snap Narrate")

    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    def generate_caption_async(image, prompt, result_callback):
        # Perform caption generation in a separate thread
        caption = swag_ai_vision_response(prompt, image)
        result_callback(caption)

    def display_caption(caption):
        st.info(caption)

    if st.button("Generate Caption") and uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
            compressed_image = compress_image(image)
            resized_image = resize_image(image)

            col1, col2 = st.columns(2)

            with col1:
                st.image(resized_image)

            default_prompt = "Write a short caption for this image"

            with st.spinner("Generating caption..."):
                # Run the caption generation asynchronously
                threading.Thread(target=generate_caption_async, args=(resized_image, default_prompt, display_caption)).start()

        except Exception as e:
            st.error(f"Error processing the image: {e}")

# Text embedding model
if selected == "Embed Text":
    st.title("🔡 Embed Text")

    # Text box to enter prompt
    user_prompt = st.text_area(label='', placeholder="Enter the text to get embeddings")

    if st.button("Get Response"):
        response = swag_ai_embeddings_response(user_prompt)
        st.markdown(response)

# Ask me anything model
if selected == "Ask Me Anything":
    st.title("❓ Ask Me a Question")

    # Text box to enter prompt
    user_prompt = st.text_area(label='', placeholder="Ask me anything...")

    if st.button("Get Response"):
        response = swag_ai_response(user_prompt)
        st.markdown(response)
