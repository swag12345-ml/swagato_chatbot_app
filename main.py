import streamlit as st
import os
from PIL import Image
from streamlit_option_menu import option_menu
from Gemini_Utility import Pritam_ai_model, Pritam_Ai_vision_response
from io import BytesIO

working_directory = os.path.dirname(os.path.abspath(__file__))  # Get current working directory

# Set page configuration
st.set_page_config(
    page_title="Pritam_Ai",
    page_icon="💀💀",
    layout="centered"
)

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Pritam_Ai",
        options=["Chatbot", "Image Captioning", "Embed Text", "Ask me anything"],
        menu_icon="robot", icons=["chat-left-dots", "card-image", "justify", "patch-question-fill"],
        default_index=0
    )

# Function to translate role between Gemini Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    return "assistant" if user_role == 'model' else user_role

# Chatbot page
if selected == "Chatbot":
    model = Pritam_ai_model()

    # Initialize chat session in Streamlit if not already present
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    st.title("🤖Chatbot")

    # Display chat history
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    # Input field for user's message
    user_prompt = st.chat_input("Ask Pritam_Ai anything .....")

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        gemini_response = st.session_state.chat_session.send_message(user_prompt)

        # Display Gemini Pro response
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)

# Image Captioning page
if selected == "Image Captioning":
    st.title("Pritam_Ai Lens")

    # File uploader for images
    uploaded_image = st.file_uploader("Upload an image here", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        resized_image = image.resize((800, 500))  # Resize for display

        st.image(resized_image, caption="Uploaded Image", use_column_width=True)

        if st.button("Generate Caption"):
            default_prompt = "Write a caption for this image."
            try:
                # Convert image to bytes for API compatibility
                image_bytes = BytesIO()
                image.save(image_bytes, format="PNG")  # Save the image as PNG or JPEG
                image_bytes.seek(0)  # Rewind the BytesIO object to the beginning

                # Get caption from Gemini API
                caption = Pritam_Ai_vision_response(default_prompt, image_bytes)

                # Display generated caption
                st.info(f"Generated Caption: {caption}")
            except Exception as e:
                # Handle errors gracefully
                st.error(f"Error generating caption: {e}")
    else:
        st.warning("Please upload an image to generate a caption.")
