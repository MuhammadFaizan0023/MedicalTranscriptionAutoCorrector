# import libraries
import streamlit as st
import requests

# Set main page title
st.set_page_config(page_title="Medical Transcription Autocorrect", layout="centered")
# Backend running url
BACKEND_URL = "http://localhost:5000"

# Set page title
st.title("🩺 Medical Transcription Autocorrect")
#st.caption("Edit-distance candidate generation + BERT contextual re-ranking")

# Check backend health before showing the main UI
try:
    health = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
    #st.caption(f"Backend connected — model: `{health.get('model', 'unknown')}`")
except requests.exceptions.ConnectionError:
    #st.error("Backend not reachable. Make sure backend.py is running on port 5000.")
    st.stop()

user_input = st.text_area(
    "Paste a transcription document or sentence with typos:",
    height=250,
    placeholder=(
        "Enter your transcription"
    )
)

if st.button("Process Text"):
    if user_input.strip():
        with st.spinner("Processing... this may take a moment."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/correct",
                    json={"text": user_input},
                    timeout=200  # longer documents take longer; adjust if needed
                )
                response.raise_for_status()
                result = response.json()

                st.subheader("Your results")
                st.success(result["corrected"])

                st.caption(f"Total sentence(s) processed: {result.get('sentence_count', '?')} ")

            except requests.exceptions.Timeout:
                st.error("Request timed out. Try a shorter document, or increase the timeout.")
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
    else:
        st.warning("Please enter some text.")