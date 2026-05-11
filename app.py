import streamlit as st
from groq import Groq
import json
import base64
from PIL import Image
import io

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
    .stDownloadButton button {
        font-size: 13px;
        padding: 4px 14px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📝 Sales Meeting Note Agent")
st.caption("Fill in meeting details, upload your notes image or type notes → click Analyze.")
st.divider()

# ── API key from Streamlit Secrets only ─────────────────────────
if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY not found in Streamlit Secrets. Please add it in app settings.")
    st.stop()

api_key = st.secrets["GROQ_API_KEY"]

# ── Step 1: Meeting Details (no rep name) ────────────────────────
st.subheader("Step 1 — Meeting Details")
col1, col2 = st.columns(2)
with col1:
    client = st.text_input("Client name", placeholder="e.g. Acme Corp")
with col2:
    date = st.text_input("Meeting date", placeholder="e.g. 11 May 2026")

st.divider()

# ── Step 2: Upload image OR type notes ──────────────────────────
st.subheader("Step 2 — Upload Notes Image or Type Notes")

uploaded_img = st.file_uploader(
    "Upload a photo of your handwritten or printed notes",
    type=["jpg", "jpeg", "png"],
    help="Take a photo of your paper notes and upload here"
)

if uploaded_img:
    img = Image.open(uploaded_img)
    st.image(img, caption="Uploaded notes", use_container_width=True)

st.markdown("<p style='text-align:center; color: gray; font-size:13px;'>— or type your notes below —</p>", unsafe_allow_html=True)

notes = st.text_area(
    "Type your notes here",
    height=160,
    label_visibility="collapsed",
    placeholder=(
        "- Client needs 20 new hires\n"
        "- Budget 5L, CFO approval needed\n"
        "- Pain: slow recruitment\n"
        "- Send pricing deck by Friday\n"
        "- Demo next week"
    )
)

st.divider()

# ── Step 3: Analyze ─────────────────────────────────────────────
st.subheader("Step 3 — Analyze")

if st.button("✨ Analyze Notes", use_container_width=True, type="primary"):

    client_name  = client.strip() or "the client"
    meeting_date = date.strip()   or "today"
    has_image    = uploaded_img is not None
    has_notes    = notes.strip() != ""

    if not has_image and not has_notes:
        st.error("Please upload a notes image or type your notes before analyzing.")
    else:
        with st.spinner("Analyzing with Groq AI..."):
            try:
                gc = Groq(api_key=api_key)

                if has_image:
                    # Convert image to base64 for vision model
                    img_bytes = uploaded_img.getvalue()
                    b64_img   = base64.b64encode(img_bytes).decode("utf-8")
                    ext       = uploaded_img.name.split(".")[-1].lower()
                    mime_type = "image/jpeg" if ext in ["jpg","jpeg"] else "image/png"

                    prompt_text = f"""You are a sales assistant for a service-based company.
The image contains handwritten or printed meeting notes from a sales meeting.

Meeting details:
- Client: {client_name}
- Date: {meeting_date}

Read ALL the text in the image carefully, then return ONLY a valid JSON object with exactly these 4 keys.
No markdown, no backticks, no extra text — just the JSON.

{{
  "summary": "2-3 sentence summary of what was discussed and where the deal stands",
  "action_items": "Bullet list: • [Owner]: [Task] — [Deadline if mentioned]",
  "follow_up_email": "Ready-to-send email. First line must be: Subject: ...",
  "next_steps": "2-3 next steps plus any deal signals or risk flags"
}}"""

                    response = gc.chat.completions.create(
                        model="meta-llama/llama-4-scout-17b-16e-instruct",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text",       "text": prompt_text},
                                {"type": "image_url",  "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}}
                            ]
                        }],
                        temperature=0.3,
                        max_tokens=1024
                    )

                else:
                    # Text-only path
                    prompt_text = f"""You are a sales assistant for a service-based company.
Process these raw meeting notes and return ONLY a valid JSON object.
No markdown, no backticks, no extra text — just the JSON.

Meeting: Client={client_name}, Date={meeting_date}
Notes: {notes}

{{
  "summary": "2-3 sentence summary of what was discussed and where the deal stands",
  "action_items": "Bullet list: • [Owner]: [Task] — [Deadline if mentioned]",
  "follow_up_email": "Ready-to-send email. First line must be: Subject: ...",
  "next_steps": "2-3 next steps plus any deal signals or risk flags"
}}"""

                    response = gc.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt_text}],
                        temperature=0.3,
                        max_tokens=1024
                    )

                raw    = response.choices[0].message.content.strip()
                raw    = raw.replace("```json", "").replace("```", "").strip()
                result = json.loads(raw)

                st.divider()
                st.subheader("📋 Results")

                sections = [
                    ("🗒️ Meeting Summary",        "summary",          "summary.txt"),
                    ("✅ Action Items",             "action_items",     "action_items.txt"),
                    ("📧 Follow-up Email",          "follow_up_email",  "followup_email.txt"),
                    ("🚀 Next Steps & Deal Signals","next_steps",       "next_steps.txt"),
                ]

                for label, key, fname in sections:
                    st.markdown(f"**{label}**")
                    content = result.get(key, "")
                    st.markdown(f'<div class="output-box">{content}</div>', unsafe_allow_html=True)
                    st.download_button(f"⬇ Download", content, file_name=fname, mime="text/plain", key=fname)
                    st.markdown("---")

                st.success("✅ Done! Copy or download any section above.")

            except json.JSONDecodeError:
                st.error("Could not parse the AI response. Please try again.")
            except Exception as e:
                st.error(f"Error: {e}")

st.divider()
st.caption("Built with Streamlit + Groq AI")
