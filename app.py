import streamlit as st
import whisper
import os

def transcribe_with_whisper(audio_file):
    # Load the Whisper model
    model = whisper.load_model("base")

    # Save the uploaded audio file to a temporary file
    temp_audio_path = "temp_audio_file.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(audio_file.read())

    # Transcribe the audio
    result = model.transcribe(temp_audio_path)

    # Remove the temporary file after transcribing
    os.remove(temp_audio_path)

    # Return the transcription
    return result['text']

# Streamlit UI
st.title("Whisper Audio Transcription")

# Upload the audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

# If a file is uploaded, transcribe it
if uploaded_file is not None:
    st.text("Transcribing audio, please wait...")
    transcription = transcribe_with_whisper(uploaded_file)
    st.subheader("Transcription:")
    st.write(transcription)
