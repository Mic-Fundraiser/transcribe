import streamlit as st
import whisper
import os
from pydub import AudioSegment
from tempfile import NamedTemporaryFile
from yt_dlp import YoutubeDL

# Carica il modello di Whisper
@st.cache_resource
def load_model():
    return whisper.load_model("base")

# Funzione per scaricare l'audio da YouTube usando yt-dlp
def download_youtube_audio(youtube_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': 'youtube_audio.%(ext)s',
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            output_file = ydl.prepare_filename(info_dict)
            return output_file
    except Exception as e:
        st.error("Errore nel download dell'audio da YouTube. Verifica l'URL o prova con un altro video.")
        st.error(f"Dettagli dell'errore: {e}")
        return None

# Funzione per trascrivere l'audio
def transcribe_audio(audio_path, model):
    with st.spinner("Trascrizione in corso..."):
        result = model.transcribe(audio_path)
    return result["text"]

# UI di Streamlit - layout e styling
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
st.markdown("<p style='text-align: center; color: #666666; font-size: 1.1em;'>Carica un file audio o inserisci un link a YouTube per ottenere la trascrizione.</p>", unsafe_allow_html=True)

# Carica il modello Whisper (solo modello "base")
model = load_model()

# Opzione per caricare un file audio
uploaded_file = st.file_uploader("Carica un file audio", type=["wav", "mp3", "m4a", "ogg"])

# Opzione per inserire un URL di YouTube
youtube_url = st.text_input("Oppure inserisci un URL di YouTube")

# Variabili per salvare il percorso dell'audio da trascrivere
audio_path = None

# Gestisci il caricamento dell'audio
if uploaded_file is not None:
    st.audio(uploaded_file)
    st.markdown("<div class='uploaded-audio'>File audio caricato correttamente. Clicca sotto per avviare la trascrizione.</div>", unsafe_allow_html=True)

    # Salva il file audio caricato in una posizione temporanea
    with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(uploaded_file.getvalue())
        audio_path = temp_audio_file.name

elif youtube_url:
    if st.button("Scarica e trascrivi audio da YouTube"):
        with st.spinner("Download dell'audio in corso da YouTube..."):
            audio_path = download_youtube_audio(youtube_url)
        
        if audio_path:
            # Converti in formato .wav se necessario
            audio_segment = AudioSegment.from_file(audio_path)
            audio_path_wav = "youtube_audio.wav"
            audio_segment.export(audio_path_wav, format="wav")
            audio_path = audio_path_wav

# Se c'è un file audio valido, avvia la trascrizione
if audio_path and st.button("Avvia Trascrizione"):
    with st.spinner("Trascrizione in corso..."):
        transcription = transcribe_audio(audio_path, model)

    # Mostra la trascrizione finale
    st.subheader("Trascrizione Finale:")
    st.markdown(
        f"<div style='font-size: 1.2em; color: #333333; padding: 15px; border-radius: 8px; background-color: #f0f0f0;'>{transcription}</div>",
        unsafe_allow_html=True
    )

    # Pulisce i file temporanei dopo l'uso
    os.remove(audio_path)
    if 'audio_path_wav' in locals():
        os.remove(audio_path_wav)
