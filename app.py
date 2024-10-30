import streamlit as st
import whisper
import os
import time
from pathlib import Path
from pydub import AudioSegment
from tempfile import NamedTemporaryFile

# Load the Whisper model
@st.cache_resource
def load_model(model_size="base"):
    return whisper.load_model(model_size)

# Function to transcribe audio in chunks
def transcribe_audio_in_chunks(audio_path, model, chunk_duration=10):
    audio = AudioSegment.from_file(audio_path)
    total_duration = len(audio) / 1000  # Duration in seconds
    transcription_text = ""
    
    st.write("Starting real-time transcription...")

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

            # Display the transcription so far
            st.write("Transcription (in progress):")
            st.write(transcription_text)
            
            # Simulate real-time delay
            time.sleep(chunk_duration / 2)  # Adjust for faster updates

    return transcription_text

# Streamlit UI
st.title("Near Real-Time Whisper Transcription")

# Sidebar for model settings
model_size = st.sidebar.selectbox(
    "Select Whisper Model Size",
    ("tiny", "base", "small", "medium", "large"),
    index=1
)

# Load the Whisper model
model = load_model(model_size)

# File uploader for audio
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    # Save the uploaded audio file to a temporary location
    with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(uploaded_file.getvalue())
        temp_audio_path = temp_audio_file.name

    # Start transcription in real-time chunks
    if st.button("Start Real-Time Transcription"):
        transcription = transcribe_audio_in_chunks(temp_audio_path, model)
        st.subheader("Final Transcription:")
        st.write(transcription)
    
    # Clean up the temporary file after use
    os.remove(temp_audio_path)
