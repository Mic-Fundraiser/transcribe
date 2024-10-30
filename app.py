import streamlit as st
import whisper
import os
import time
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the Whisper model at the start to avoid reloading
@st.cache_resource
def load_model(model_size):
    logging.info(f"Loading Whisper model: {model_size}")
    return whisper.load_model(model_size)

# Function to transcribe audio
def transcribe_with_whisper(audio_file, model, language_option):
    temp_audio_path = Path("temp_audio_file") / audio_file.name

    # Ensure the temp directory exists
    temp_audio_path.parent.mkdir(exist_ok=True)

    # Save the uploaded audio file to a temporary file
    with open(temp_audio_path, "wb") as f:
        f.write(audio_file.getvalue())

    # Display progress and start timing
    with st.spinner("Transcribing audio..."):
        start_time = time.time()
        # Set options for the transcription
        options = {}
        if language_option != "Auto-detect":
            options["language"] = language_option

        result = model.transcribe(str(temp_audio_path), **options)
        end_time = time.time()

    os.remove(temp_audio_path)  # Clean up

    # Display processing time
    st.write(f"Transcription completed in {end_time - start_time:.2f} seconds")
    return result['text']

# Streamlit UI
st.title("Advanced Whisper Audio Transcription")

# Sidebar for options
st.sidebar.title("Settings")

# Model selection
model_size = st.sidebar.selectbox(
    "Select Whisper Model Size",
    ("tiny", "base", "small", "medium", "large"),
    index=1  # default to 'base'
)

# Language selection
language_option = st.sidebar.selectbox(
    "Transcription Language",
    ("Auto-detect", "English", "Spanish", "French", "German", "Italian", "Other")
)

if language_option == "Other":
    language_option = st.sidebar.text_input("Enter Language Code (e.g., 'en', 'es', 'fr')")

# Load the model
model = load_model(model_size)

# Upload the audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg"])

if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file)

    # Transcribe button
    if st.button("Transcribe"):
        transcription = transcribe_with_whisper(uploaded_file, model, language_option)
        st.subheader("Transcription:")
        st.write(transcription)

        # Option to download the transcription
        st.download_button(
            label="Download Transcription",
            data=transcription,
            file_name="transcription.txt",
            mime="text/plain"
        )
