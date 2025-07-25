import os
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
from gemini_utility import (
    load_swag_ai_model,
    swag_ai_response,
    swag_ai_vision_response,
    swag_ai_embeddings_response
)

# ✅ Set Streamlit page config
st.set_page_config(
    page_title="Swag AI",
    page_icon="🧠",
    layout="centered",
)

# ✅ Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        "Swag AI",
        ["ChatBot", "Image Captioning", "Embed Text", "Ask Me Anything"],
        menu_icon="robot",
        icons=["chat-dots-fill", "image-fill", "textarea-t", "patch-question-fill"],
        default_index=0
    )

# ✅ Role translation for Streamlit's chat display
def translate_role_for_streamlit(user_role):
    return "assistant" if user_role == "model" else user_role

# ✅ ChatBot Section
if selected == "ChatBot":
    model = load_swag_ai_model()

    # Initialize chat session
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    st.title("🤖 ChatBot")

    # Display chat history
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    # Chat input
    user_prompt = st.chat_input("Ask Swag AI...")
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        try:
            response = st.session_state.chat_session.send_message(user_prompt)
            with st.chat_message("assistant"):
                st.markdown(response.text)
        except Exception as e:
            st.error(f"❌ Failed to get response: {e}")

# ✅ Image Captioning Section
elif selected == "Image Captioning":
    st.title("📷 Snap Narrate")
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if st.button("Generate Caption") and uploaded_image:
        try:
            image = Image.open(uploaded_image)

            col1, col2 = st.columns(2)
            with col1:
                resized_img = image.resize((800, 500))
                st.image(resized_img, caption="Uploaded Image", use_column_width=True)

            default_prompt = "Write a short caption for this image"
            caption = swag_ai_vision_response(default_prompt, image)

            with col2:
                st.info(caption)
        except Exception as e:
            st.error(f"❌ Error processing the image: {e}")

# ✅ Embed Text Section
elif selected == "Embed Text":
    st.title("🔡 Embed Text")
    user_prompt = st.text_area("Enter text to embed:")

    if st.button("Get Response") and user_prompt.strip():
        try:
            embedding = swag_ai_embeddings_response(user_prompt)
            st.success("✅ Embedding generated successfully!")
            st.json(embedding)
        except Exception as e:
            st.error(f"❌ Failed to generate embedding: {e}")

# ✅ Ask Me Anything Section
elif selected == "Ask Me Anything":
    st.title("❓ Ask Me a Question")
    user_prompt = st.text_area("Ask me anything...")

    if st.button("Get Response") and user_prompt.strip():
        try:
            response = swag_ai_response(user_prompt)
            st.markdown(response)
        except Exception as e:
            st.error(f"❌ Failed to get response: {e}")
