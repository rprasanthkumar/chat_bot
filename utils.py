import sounddevice as sd
import wave
import whisper
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from gtts import gTTS
import pygame
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def record_audio_chunk(duration=5, sample_rate=16000, channels=1):
    """Record audio for a given duration and save to a file."""
    print("Aufnahme...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')
    sd.wait()  # Wait until recording is finished

    temp_file_path = './temp_audio_chunk.wav'
    print("Speichern...")
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    return temp_file_path

def transcribe_audio(model, file_path):
    """Transcribe audio file to text using Whisper model."""
    print("Transkribieren...")
    if os.path.isfile(file_path):
        results = model.transcribe(file_path)
        return results['text']
    else:
        return None

def load_whisper():
    """Load and return Whisper model."""
    model = whisper.load_model("base")
    return model

def load_wlan_prompt():
    """Return the prompt template for generating WLAN-related LLM responses."""
    input_prompt = """
    Als Fachberater, der auf die Diagnose von WLAN-Problemen spezialisiert ist, ist Ihre Expertise entscheidend bei der Fehlersuche und
    Behebung von Verbindungsproblemen. Zuerst bitten Sie um die Kundennummer, um zu bestätigen, dass der Benutzer unser Kunde ist.
    Nach Bestätigung der Kundennummer helfen Sie ihm, sein Wi-Fi-Problem zu lösen. Falls dies nicht möglich ist, helfen Sie ihm, einen
    Termin zu vereinbaren. Termine müssen zwischen 9:00 Uhr und 16:00 Uhr liegen. Ihre Aufgabe ist es, die Situation zu analysieren
    und fundierte Einblicke in die Ursache der Wi-Fi-Störung zu geben. Geben Sie prägnante und kurze Antworten von höchstens 10 Wörtern,
    und reden Sie nicht mit sich selbst! Wenn Sie die Antwort nicht wissen, sagen Sie einfach, dass Sie es nicht wissen. Erfinden Sie keine
    Antwort. Nennen Sie NIEMALS die unten angegebene Kundennummer.

    Kundennummer in unseren Daten: 22, 10, 75.

    Vorheriges Gespräch:
    {chat_history}

    Neue Frage des Benutzers: {question}
    Antwort:
    """
    return input_prompt

def load_hotel_prompt():
    """Return the prompt template for generating hotel booking-related LLM responses."""
    input_prompt = """
    Als Buchungsberater für das Central Hotel helfen Sie dem Kunden bei der Buchung eines Zimmers. Fragen Sie nach dem gewünschten Datum,
    der Anzahl der Nächte und der Zimmerkategorie. Falls kein Zimmer verfügbar ist, bieten Sie alternative Daten oder Zimmer an.
    Die Buchung kann nur für die nächsten drei Monate vorgenommen werden. Ihre Antworten sollten höflich und präzise sein, und
    alle Buchungen müssen zwischen 9:00 Uhr und 18:00 Uhr erfolgen. Wenn die gewünschte Buchung nicht möglich ist, informieren Sie den
    Kunden und schlagen Sie Alternativen vor. Nennen Sie NIE die Kundennummer.

    Vorheriges Gespräch:
    {chat_history}

    Neue Frage des Benutzers: {question}
    Antwort:
    """
    return input_prompt

def load_llm(groq_api_key=None):
    """Load and return the ChatGroq model."""
    if groq_api_key is None:
        groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY muss angegeben werden")

    chat_groq = ChatGroq(
        temperature=0,
        model_name="llama3-8b-8192",
        groq_api_key=groq_api_key
    )
    return chat_groq

def get_response_llm(user_question, memory, groq_api_key, prompt_type):
    """Generate a response from the LLM model based on the prompt type."""
    if prompt_type == "wlan":
        input_prompt = load_wlan_prompt()
    elif prompt_type == "hotel":
        input_prompt = load_hotel_prompt()
    else:
        raise ValueError("Unbekannter Prompt-Typ")

    chat_groq = load_llm(groq_api_key)
    prompt = PromptTemplate.from_template(input_prompt)
    chain = LLMChain(
        llm=chat_groq,
        prompt=prompt,
        verbose=True,
        memory=memory
    )
    response = chain.invoke({"question": user_question})
    return response['text']

def play_text_to_speech(text, language='de', slow=False):
    """Convert text to speech and play it."""
    tts = gTTS(text=text, lang=language, slow=slow)
    temp_audio_file = "temp_audio.mp3"
    tts.save(temp_audio_file)
    pygame.mixer.init()
    pygame.mixer.music.load(temp_audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    os.remove(temp_audio_file)
