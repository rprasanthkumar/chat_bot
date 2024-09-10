import os
import streamlit as st
from langchain.memory import ConversationBufferMemory
from utils import record_audio_chunk, transcribe_audio, get_response_llm, play_text_to_speech, load_whisper
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Whisper model
model = load_whisper()

def main():
    st.markdown('<h1 style="color: darkblue;">AI Voice Assistant️</h1>', unsafe_allow_html=True)

    # Select scenario
    st.sidebar.header("Select Scenario")
    scenario = st.sidebar.radio("Choose the scenario:", ("WLAN Troubleshooting", "Hotel Booking"))

    memory = ConversationBufferMemory(memory_key="chat_history")
    groq_api_key = os.getenv("GROQ_API_KEY")

    if groq_api_key is None:
        st.error("GROQ_API_KEY environment variable is not set.")
        return

    if st.button("Start Recording"):
        while True:
            # Record and save audio chunk
            temp_file_path = record_audio_chunk(duration=5)

            # Transcribe the recorded audio
            text = transcribe_audio(model, temp_file_path)

            if text is not None:
                st.markdown(
                    f'<div style="background-color: #FF6347; padding: 10px; border-radius: 5px;">Customer 👤: {text}</div>',
                    unsafe_allow_html=True)

                os.remove(temp_file_path)

                # Determine prompt type based on scenario
                prompt_type = "wlan" if scenario == "WLAN Troubleshooting" else "hotel"

                # Generate a response using the AI model
                response_llm = get_response_llm(user_question=text, memory=memory, groq_api_key=groq_api_key, prompt_type=prompt_type)
                st.markdown(
                    f'<div style="background-color: #FF6347; padding: 10px; border-radius: 5px;">AI Assistant 🤖: {response_llm}</div>',
                    unsafe_allow_html=True)

                # Play the response as speech
                play_text_to_speech(text=response_llm)
            else:
                break  # Exit the while loop

        st.markdown('<h2 style="color: darkgreen;">End of Conversation</h2>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
