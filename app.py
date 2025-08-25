import streamlit as st
import openai
import tempfile
import whisper
import os

# Streamlit UI setup
st.set_page_config(page_title="MedNote AI - Doctor Visit Summarizer", layout="centered")
st.title("🩺 MedNote AI")
st.subheader("Summarize your doctor visit from an audio recording")

# API key input
openai_api_key = st.secrets["OPENAI_API_KEY"]
openai.api_base = "https://api.groq.com/openai/v1"

if not openai_api_key:
    st.warning("Please enter your Groq API key to proceed.")
    st.stop()

openai.api_key = openai_api_key

# Upload audio
st.markdown("### Step 1: Upload a doctor-patient conversation (audio)")
audio_file = st.file_uploader("Upload an audio file (mp3, wav, m4a):", type=["mp3", "wav", "m4a"])

if st.button("🧠 Generate Summary"):
    if not audio_file:
        st.warning("Please upload an audio file first.")
        st.stop()

    with st.spinner("🔍 Transcribing audio..."):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name

            # Load Whisper model
            model = whisper.load_model("base")
            transcription_result = model.transcribe(tmp_path)
            transcript = transcription_result["text"]

            # Clean up temp file
            os.remove(tmp_path)

        except Exception as e:
            st.error(f"Transcription error: {e}")
            st.stop()

    with st.spinner("📝 Summarizing visit with GPT-4..."):
        prompt = f"""
You are a medical assistant AI. A patient has just visited a doctor. Below is the conversation transcription.

Summarize the visit with the following structure:
- Chief Complaint / Reason for Visit
- Diagnosis (if any)
- Recommended Tests or Treatments
- Prescriptions or Medications
- Lifestyle or Follow-up Recommendations
- Suggested Questions for the Patient to Ask

Transcript:
\"\"\"
{transcript}
\"\"\"
"""
        try:
            response = openai.ChatCompletion.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )
            summary = response["choices"][0]["message"]["content"]

            st.markdown("### 📝 Medical Visit Summary")
            st.markdown(summary)

            st.markdown("---")
            st.markdown("### 📃 Transcription")
            st.text_area("Full Transcript", transcript, height=200)

        except Exception as e:
            st.error(f"Error generating summary: {e}")

st.markdown("---")
st.caption("⚠️ This is a demo. Not for clinical use or medical advice.")
