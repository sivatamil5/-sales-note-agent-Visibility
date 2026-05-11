import streamlit as st
from groq import Groq
import json
import base64
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Sales Note Agent", page_icon="📝", layout="centered")

st.markdown("""
    <style>
    .output-box {
        background: #f0faf5;
        border-left: 4px solid #1D9E75;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        white-space: pre-wrap;
        font-size: 14px;
        line-height: 1.8;
    }
    .upload-hint {
        font-size: 12px;
        color: #888;
        margin-top: 4px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📝 Sales Meeting Note Agent")
st.caption("Upload your meeting video, image, or type notes → get instant summary, action items, email & next steps.")
st.divider()

# ── API key from Streamlit Secrets ───────────────────────────────
if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY not found in Streamlit Secrets. Please add it in app settings.")
    st.stop()

api_key = st.secrets["GROQ_API_KEY"]
gc = Groq(api_key=api_key)

# ── Step 1: Meeting Details ──────────────────────────────────────
st.subheader("Step 1 — Meeting Details")
col1, col2 = st.columns(2)
with col1:
    client = st.text_input("Client name", placeholder="e.g. Acme Corp")
with col2:
    date = st.text_input("Meeting date", placeholder="e.g. 11 May 2026")

st.divider()

# ── Step 2: Choose input type ────────────────────────────────────
st.subheader("Step 2 — Choose Your Input")

input_type = st.radio(
    "What do you want to upload or use?",
    ["🎥 Meeting video", "🖼️ Notes image (photo)", "⌨️ Type notes manually"],
    horizontal=True,
    label_visibility="collapsed"
)

uploaded_video = None
uploaded_img   = None
notes          = ""
transcript     = ""

# ── Video upload ─────────────────────────────────────────────────
if input_type == "🎥 Meeting video":
    st.markdown("**Upload your meeting recording**")
    uploaded_video = st.file_uploader(
        "Upload meeting video",
        type=["mp4", "mov", "avi", "mkv", "webm", "m4a", "mp3", "wav"],
        label_visibility="collapsed"
    )
    st.markdown('<p class="upload-hint">Supported: MP4, MOV, AVI, MKV, WEBM, MP3, WAV, M4A · Max 25MB (Groq limit)</p>', unsafe_allow_html=True)

    if uploaded_video:
        file_size_mb = uploaded_video.size / (1024 * 1024)
        if file_size_mb > 25:
            st.error(f"File is {file_size_mb:.1f}MB — Groq Whisper limit is 25MB. Please trim the video and re-upload.")
            uploaded_video = None
        else:
            st.success(f"✅ File uploaded: {uploaded_video.name} ({file_size_mb:.1f}MB)")
            st.info("Audio will be extracted and transcribed automatically when you click Analyze.")

# ── Image upload ─────────────────────────────────────────────────
elif input_type == "🖼️ Notes image (photo)":
    st.markdown("**Upload a photo of your handwritten or printed notes**")
    uploaded_img = st.file_uploader(
        "Upload notes image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    if uploaded_img:
        img = Image.open(uploaded_img)
        st.image(img, caption="Uploaded notes", use_container_width=True)

# ── Manual notes ─────────────────────────────────────────────────
elif input_type == "⌨️ Type notes manually":
    st.markdown("**Type or paste your meeting notes below**")
    notes = st.text_area(
        "Notes",
        height=180,
        label_visibility="collapsed",
        placeholder=(
            "- Client needs 20 new hires\n"
            "- Budget 5L, CFO approval needed\n"
            "- Pain: slow recruitment process\n"
            "- Send pricing deck by Friday\n"
            "- Demo requested for next week"
        )
    )

st.divider()

# ── Step 3: Analyze ──────────────────────────────────────────────
st.subheader("Step 3 — Analyze")

if st.button("✨ Analyze Now", use_container_width=True, type="primary"):

    client_name  = client.strip() or "the client"
    meeting_date = date.strip()   or "today"

    nothing_uploaded = (
        uploaded_video is None and
        uploaded_img   is None and
        notes.strip()  == ""
    )

    if nothing_uploaded:
        st.error("Please upload a video, image, or type your notes before analyzing.")
    else:
        try:
            # ── STEP A: Transcribe video if uploaded ─────────────
            if uploaded_video is not None:
                with st.spinner("🎙️ Transcribing meeting audio with Groq Whisper..."):
                    suffix = "." + uploaded_video.name.split(".")[-1].lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_video.read())
                        tmp_path = tmp.name

                    with open(tmp_path, "rb") as audio_file:
                        transcription = gc.audio.transcriptions.create(
                            file=(uploaded_video.name, audio_file.read()),
                            model="whisper-large-v3",
                            response_format="text"
                        )
                    os.unlink(tmp_path)
                    transcript = transcription if isinstance(transcription, str) else transcription.text

                st.success("✅ Transcription complete!")
                with st.expander("📄 View transcript"):
                    st.write(transcript)

            # ── STEP B: Build prompt & call LLaMA ────────────────
            json_structure = """{
  "summary": "2-3 sentence summary of what was discussed and where the deal stands",
  "action_items": "Bullet list: • [Owner]: [Task] — [Deadline if mentioned]",
  "follow_up_email": "Ready-to-send email. First line must be: Subject: ...",
  "next_steps": "2-3 next steps plus any deal signals or risk flags"
}"""

            if uploaded_video is not None:
                # Use transcript as text input
                with st.spinner("🤖 Analyzing transcript with Groq AI..."):
                    prompt = f"""You are a sales assistant for a service-based company.
Below is the transcript of a sales meeting. Analyze it and return ONLY a valid JSON object.
No markdown, no backticks, no extra text — just the JSON.

Meeting: Client={client_name}, Date={meeting_date}
Transcript:
{transcript}

Return this JSON:
{json_structure}"""

                    response = gc.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=1024
                    )

            elif uploaded_img is not None:
                # Use vision model for image
                with st.spinner("🤖 Reading notes image with Groq AI..."):
                    img_bytes = uploaded_img.getvalue()
                    b64_img   = base64.b64encode(img_bytes).decode("utf-8")
                    ext       = uploaded_img.name.split(".")[-1].lower()
                    mime_type = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"

                    prompt = f"""You are a sales assistant for a service-based company.
The image contains handwritten or printed meeting notes. Read ALL text carefully.
Return ONLY a valid JSON object — no markdown, no backticks, no extra text.

Meeting: Client={client_name}, Date={meeting_date}

Return this JSON:
{json_structure}"""

                    response = gc.chat.completions.create(
                        model="meta-llama/llama-4-scout-17b-16e-instruct",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text",      "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}}
                            ]
                        }],
                        temperature=0.3,
                        max_tokens=1024
                    )

            else:
                # Text notes
                with st.spinner("🤖 Analyzing notes with Groq AI..."):
                    prompt = f"""You are a sales assistant for a service-based company.
Process these raw meeting notes and return ONLY a valid JSON object.
No markdown, no backticks, no extra text — just the JSON.

Meeting: Client={client_name}, Date={meeting_date}
Notes:
{notes}

Return this JSON:
{json_structure}"""

                    response = gc.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=1024
                    )

            # ── STEP C: Show results ──────────────────────────────
            raw    = response.choices[0].message.content.strip()
            raw    = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)

            st.divider()
            st.subheader("📋 Results")

            sections = [
                ("🗒️ Meeting Summary",         "summary",          "summary.txt"),
                ("✅ Action Items",              "action_items",     "action_items.txt"),
                ("📧 Follow-up Email Draft",     "follow_up_email",  "followup_email.txt"),
                ("🚀 Next Steps & Deal Signals", "next_steps",       "next_steps.txt"),
            ]

            for label, key, fname in sections:
                st.markdown(f"**{label}**")
                content = result.get(key, "")
                st.markdown(f'<div class="output-box">{content}</div>', unsafe_allow_html=True)
                st.download_button("⬇ Download", content, file_name=fname, mime="text/plain", key=fname)
                st.markdown("---")

            st.success("✅ Done! Copy or download any section above.")

        except json.JSONDecodeError:
            st.error("Could not parse the AI response. Please try again.")
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()
st.caption("Built with Streamlit + Groq AI (Whisper + LLaMA 3.3)")
