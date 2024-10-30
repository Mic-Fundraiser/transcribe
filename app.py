import streamlit as st
import whisper
import os
import time
from pathlib import Path
from pydub import AudioSegment
from tempfile import NamedTemporaryFile

# Load the Whisper model
@st.cache_resource
def load_model():
    return whisper.load_model("base")

# Function to transcribe audio in chunks
def transcribe_audio_in_chunks(audio_path, model, chunk_duration=10):
    audio = AudioSegment.from_file(audio_path)
    total_duration = len(audio) / 1000  # Duration in seconds
    transcription_text = ""

    # Title styling
    st.markdown("<h3 style='text-align: center; color: #ff4b4b;'>Real-Time Transcription</h3>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #ff4b4b;'>", unsafe_allow_html=True)

    # Process each chunk and update transcription
    for i in range(0, int(total_duration), chunk_duration):
        start_time = i * 1000  # Start time in milliseconds
        end_time = min((i + chunk_duration) * 1000, len(audio))  # End time
        chunk = audio[start_time:end_time]

        # Save the chunk to a temporary file
        with NamedTemporaryFile(suffix=".wav") as temp_chunk_file:
            chunk.export(temp_chunk_file.name, format="wav")
            result = model.transcribe(temp_chunk_file.name)
            transcription_text += result["text"] + " "

            # Display the transcription so far in a continuous manner with styled markdown
            st.markdown("<div style='font-size: 1.1em; color: #333333; margin-top: 20px; padding: 10px; border-radius: 5px; background-color: #f9f9f9;'>" + transcription_text + "</div>", unsafe_allow_html=True)

            # Simulate real-time delay
            time.sleep(chunk_duration / 2)  # Adjust for faster updates

    return transcription_text

# Streamlit UI - App layout and styling
st.markdown("""
    <style>
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            font-size: 1.1em;
            padding: 10px 20px;
            border-radius: 10px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #ff3333;
        }
        .uploaded-audio {
            color: #555555;
            font-size: 1em;
            text-align: center;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>Whisper Audio Transcription</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666666; font-size: 1.1em;'>Upload your audio file to get a real-time transcription.</p>", unsafe_allow_html=True)

# Load the Whisper model (only "base" model)
model = load_model()

# File uploader for audio
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    st.markdown("<div class='uploaded-audio'>Audio file uploaded successfully. Click below to start transcribing.</div>", unsafe_allow_html=True)

    # Save the uploaded audio file to a temporary location
    with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(uploaded_file.getvalue())
        temp_audio_path = temp_audio_file.name

    # Start transcription in real-time chunks
    if st.button("Start Real-Time Transcription"):
        transcription = transcribe_audio_in_chunks(temp_audio_path, model)
        st.subheader("Final Transcription:")
        st.markdown("<div style='font-size: 1.2em; color: #333333; padding: 15px; border-radius: 8px; background-color: #f0f0f0;'>" + transcription + "</div>", unsafe_allow_html=True)
    
    # Clean up the temporary file after use
    os.remove(temp_audio_path)
