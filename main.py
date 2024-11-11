import os
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
from gemini_utility import (
    load_swag_ai_model,
    swag_ai_response,
    swag_ai_vision_response,
    swag_ai_embeddings_response,
    extract_text_from_pdf,
    swag_ai_pdf_response
)

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
                            'Embed Text from PDF',
                            'Ask Me Anything'],
                           menu_icon='robot', icons=['chat-dots-fill', 'image-fill', 'textarea-t', 'file-earmark-pdf-fill', 'patch-question-fill'],
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

# Image captioning page
if selected == "Image Captioning":
    st.title("📷 Snap Narrate")
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if st.button("Generate Caption") and uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
            col1, col2 = st.columns(2)
            with col1:
                resized_img = image.resize((800, 500))
                st.image(resized_img)
            default_prompt = "Write a short caption for this image"
            caption = swag_ai_vision_response(default_prompt, image)
            with col2:
                st.info(caption)
        except Exception as e:
            st.error(f"Error processing the image: {e}")

# Text embedding model
if selected == "Embed Text":
    st.title("🔡 Embed Text")
    user_prompt = st.text_area(label='', placeholder="Enter the text to get embeddings")
    if st.button("Get Response"):
        response = swag_ai_embeddings_response(user_prompt)
        st.markdown(response)

# Embed Text from PDF
if selected == "Embed Text from PDF":
    st.title("📄 Embed Text from PDF")
    uploaded_pdf = st.file_uploader("Upload a PDF file...", type="pdf")
    if uploaded_pdf:
        pdf_text = extract_text_from_pdf(uploaded_pdf)
        st.session_state.pdf_embeddings = swag_ai_embeddings_response(pdf_text)
        st.success("PDF content embedded successfully. You can now ask questions.")

    user_prompt = st.text_area("Ask a question based on the PDF content...")
    if st.button("Get Response") and "pdf_embeddings" in st.session_state:
        response = swag_ai_pdf_response(user_prompt, st.session_state.pdf_embeddings)
        st.markdown(response)
    elif st.button("Get Response"):
        st.warning("Please upload a PDF and embed it first.")

# Ask me anything model
if selected == "Ask Me Anything":
    st.title("❓ Ask Me a Question")
    user_prompt = st.text_area(label='', placeholder="Ask me anything...")
    if st.button("Get Response"):
        response = swag_ai_response(user_prompt)
        st.markdown(response)
