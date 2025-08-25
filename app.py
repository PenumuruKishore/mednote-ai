import streamlit as st
import openai
import assemblyai as aai

st.set_page_config(page_title="MedNote AI - Doctor Visit Summarizer", layout="centered")
st.title("🩺 MedNote AI")
st.subheader("Summarize your doctor visit from an audio recording")

# API keys
openai.api_key = st.secrets["OPENAI_API_KEY"]
openai.api_base = "https://api.groq.com/openai/v1"
aai.settings.api_key = st.secrets["ASSEMBLYAI_API_KEY"]

# Upload audio
audio_file = st.file_uploader("Upload audio (mp3/wav):", type=["mp3", "wav", "m4a"])

if st.button("Generate Summary"):
    if not audio_file:
        st.warning("Please upload a file.")
        st.stop()

    with st.spinner("⏳ Transcribing audio..."):
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_file)
        transcription_text = transcript.text

    prompt = f"""
    You are a helpful assistant that summarizes doctor-patient conversations into structured medical notes.

    Transcript:
    {transcription_text}

    Please generate a clear, concise summary with:
    - Symptoms mentioned
    - Diagnoses discussed
    - Medications prescribed
    - Follow-up instructions
    """

    with st.spinner("🤖 Generating summary with LLaMA 3..."):
        try:
            response = openai.ChatCompletion.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You summarize medical transcripts into patient-friendly notes."},
                    {"role": "user", "content": prompt}
                ]
            )
            summary = response["choices"][0]["message"]["content"]
        except Exception as e:
            st.error(f"Error generating summary: {e}")
            st.stop()

    st.markdown("### ✅ Summary")
    st.markdown(summary)

    st.markdown("---")
    st.markdown("### 📝 Full Transcript")
    st.text_area("Transcript", transcription_text, height=300)

