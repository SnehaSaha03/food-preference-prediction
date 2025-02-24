import streamlit as st
import requests
import io
import os
import base64
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()
LLM="grok-2-1212"
# Get the API key from environment variables
subscription_key = os.getenv("SARVAMAPIKEY")
xai_api_key = os.getenv("XAIAPIKEY")
sarvamurl = "https://api.sarvam.ai/text-to-speech"
sarvamheaders = {
    "accept": "application/json",
    "content-type": "application/json",
    "api-subscription-key": subscription_key
}

# Initialize OpenAI client
LLMclient=OpenAI(
  api_key=os.getenv('XAIAPIKEY'),
  base_url="https://api.x.ai/v1",
)

# Load CSS from file
with open("style.css", "r") as f:
    css = f.read()

# Inject CSS into Streamlit
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Function to send audio to Sarvam API for speech recognition
def transcribe_audio(audio_file, subscription_key, language_code='hi-IN'):
    url = "https://api.sarvam.ai/speech-to-text"
    
    # Prepare the payload with model and language code
    payload = {
        'model': 'saarika:v1',
        'language_code': language_code,
        'with_timesteps': 'false'
    }
    
    # Prepare files for the request
    files = [
        ('file', ('audio.wav', audio_file, 'audio/wav'))
    ]
    
    # Set headers including your API subscription key
    headers = {
        'api-subscription-key': subscription_key
    }
    
    # Make the POST request to the API
    response = requests.post(url, headers=headers, data=payload, files=files)

    if response.status_code == 200:
        return response.json().get('transcript', 'No transcript available.')
    else:
        return f"Error: {response.status_code}, {response.text}"

# Function to process text using OpenAI API
def process_with_llm(text):
    try:
        response = LLMclient.chat.completions.create(
            model=LLM,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Language code mapping
language_mapping = {
    "Hindi": "hi-IN",
    "Bengali": "bn-IN",
    "Kannada": "kn-IN",
    "Malayalam": "ml-IN",
    "Marathi": "mr-IN",
    "Odia": "od-IN",
    "Punjabi": "pa-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "English (Indian)": "en-IN",
    "Gujarati": "gu-IN"
}

# Streamlit UI
st.title("Cuisine Companion")

# Move language selection to sidebar
with st.sidebar:
    selected_language = st.selectbox("Select Language:", 
                                  list(language_mapping.keys()), 
                                  index=0)
    language_code = language_mapping[selected_language]

audio_value = st.audio_input("Record a voice message")

if audio_value:
    if subscription_key:
        with st.spinner("Transcribing..."):
            transcript = transcribe_audio(audio_value, subscription_key, language_code)
            # Display user message without markdown
            st.write(f"User 🙋 : {transcript}")

            # Process the transcript with OpenAI
            if xai_api_key:
                with st.spinner("Processing with Model..."):
                    model_response = process_with_llm(transcript)
                    # Display model response without markdown
                    st.write(f"Model 🤖 : {model_response}")

                    payload = {
                        "inputs": [model_response[:490]],
                        "target_language_code": language_code,
                        "speaker": "meera",
                        "pitch": 0.2,
                        "pace": 1.1,
                        "loudness": 0.8,
                        "enable_preprocessing": True,
                        "model": "bulbul:v1",
                        "speech_sample_rate": 8000
                    }
                    response = requests.request("POST", sarvamurl, json=payload, headers=sarvamheaders)
                    #print(response.json())
                    audio_data = response.json()
                    if "audios" in audio_data and audio_data["audios"]:
                        # Decode base64 audio
                        audio_bytes = base64.b64decode(audio_data["audios"][0])
                        # Play the audio in Streamlit with custom CSS class
                        st.markdown('<div class="st-ae">', unsafe_allow_html=True)
                        st.audio(audio_bytes, format="audio/wav")
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error(" API Key not found in environment variables.")
    else:
        st.error("API Subscription Key not found in environment variables.")
