import streamlit as st
from groq import Groq
import PyPDF2
import docx
import io
import base64
import PyPDF2

st.set_page_config(page_title="Sales Agent", layout="wide")
st.title("🤝 Sales Agent — powered by Groq AI")
st.caption("Upload image / transcript / type notes → Get full sales analysis")

groq_key = st.secrets["GROQ_API_KEY"]

# ── Helper: image to base64 ────────────────────────────────────
def image_to_base64(file):
    return base64.b64encode(file.read()).decode("utf-8")

# ── Helper: extract PDF ────────────────────────────────────────
def extract_pdf(file):
    reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

# ── Helper: extract DOCX ──────────────────────────────────────
def extract_docx(file):
    doc = docx.Document(io.BytesIO(file.read()))
    return "\n".join([p.text for p in doc.paragraphs]).strip()

# ── Helper: extract TXT ───────────────────────────────────────
def extract_txt(file):
    return file.read().decode("utf-8").strip()

# ── Helper: read any file ─────────────────────────────────────
def read_file(file):
    name = file.name.lower()
    if name.endswith(".pdf"):
        return extract_pdf(file)
    elif name.endswith(".docx"):
        return extract_docx(file)
    else:
        return extract_txt(file)

# ── Extract text from image using Groq vision ─────────────────
def extract_text_from_image(base64_img):
    client = Groq(api_key=groq_key)
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_img}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Read all the text from this image carefully. This is a sales note or meeting transcript. Extract every word exactly as written."
                    }
                ]
            }
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content

# ── Full sales analysis ────────────────────────────────────────
def run_sales_agent(transcript, notes):
    client = Groq(api_key=groq_key)

    combined = f"""
MEETING TRANSCRIPT:
{transcript if transcript else "Not provided"}

SALES NOTES:
{notes if notes else "Not provided"}
"""

    prompt = f"""
You are an expert Sales Agent AI. Analyze the meeting transcript and sales notes below.

Give a complete structured analysis:

## 📋 Meeting Summary
5-6 bullet points of what was discussed.

## 🎯 Key Sales Opportunities
For each opportunity:
- Opportunity name
- Potential value
- Probability (High/Medium/Low)

## ✅ Action Items
For each action item:
- Task
- Owner
- Deadline

## 🔄 Follow-up Plan
What to do in:
- Next 24 hours
- Next 1 week
- Next 1 month

## 😊 Customer Sentiment
- Overall sentiment (Positive/Neutral/Negative)
- Key concerns raised
- Buying signals detected

## 💡 Top 3 Recommendations
Best moves to close this deal.

## ⚠️ Risks & Blockers
What could stop this sale.

Be specific, professional, and actionable.

{combined}
"""

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000
    )
    return response.choices[0].message.content

# ── UI ─────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns(2)

# ── LEFT: Meeting Transcript ───────────────────────────────────
with col1:
    st.subheader("📹 Meeting Transcript")

    transcript_method = st.radio(
        "Choose input method:",
        [
            "📷 Upload image of transcript",
            "📁 Upload file (TXT/PDF/DOCX)",
            "✏️ Type or paste manually"
        ],
        key="t_method"
    )

    transcript_text = ""

    if transcript_method == "📷 Upload image of transcript":
        img_file = st.file_uploader(
            "Upload photo of transcript",
            type=["jpg", "jpeg", "png"],
            key="t_img"
        )
        if img_file:
            st.image(img_file, caption="Uploaded transcript image", width=300)
            with st.spinner("Reading text from image..."):
                b64 = image_to_base64(img_file)
                transcript_text = extract_text_from_image(b64)
            st.success("✅ Text extracted from image!")
            with st.expander("👀 View extracted text"):
                st.text(transcript_text)

    elif transcript_method == "📁 Upload file (TXT/PDF/DOCX)":
        t_file = st.file_uploader(
            "Upload transcript file",
            type=["txt", "pdf", "docx"],
            key="t_file"
        )
        if t_file:
            transcript_text = read_file(t_file)
            st.success(f"✅ File loaded! ({len(transcript_text)} characters)")
            with st.expander("👀 Preview"):
                st.text(transcript_text[:500] + "..." if len(transcript_text) > 500 else transcript_text)

    else:
        transcript_text = st.text_area(
            "Paste or type transcript here:",
            height=250,
            placeholder="Paste your Teams meeting transcript here...",
            key="t_text"
        )

# ── RIGHT: Sales Notes ─────────────────────────────────────────
with col2:
    st.subheader("📝 Sales Notes")

    notes_method = st.radio(
        "Choose input method:",
        [
            "📷 Upload image of notes",
            "📁 Upload file (TXT/PDF/DOCX)",
            "✏️ Type or paste manually"
        ],
        key="n_method"
    )

    notes_text = ""

    if notes_method == "📷 Upload image of notes":
        notes_img = st.file_uploader(
            "Upload photo of your handwritten notes",
            type=["jpg", "jpeg", "png"],
            key="n_img"
        )
        if notes_img:
            st.image(notes_img, caption="Uploaded notes image", width=300)
            with st.spinner("Reading text from image..."):
                b64 = image_to_base64(notes_img)
                notes_text = extract_text_from_image(b64)
            st.success("✅ Text extracted from image!")
            with st.expander("👀 View extracted text"):
                st.text(notes_text)

    elif notes_method == "📁 Upload file (TXT/PDF/DOCX)":
        n_file = st.file_uploader(
            "Upload notes file",
            type=["txt", "pdf", "docx"],
            key="n_file"
        )
        if n_file:
            notes_text = read_file(n_file)
            st.success(f"✅ File loaded! ({len(notes_text)} characters)")
            with st.expander("👀 Preview"):
                st.text(notes_text[:500] + "..." if len(notes_text) > 500 else notes_text)

    else:
        notes_text = st.text_area(
            "Type or paste your sales notes here:",
            height=250,
            placeholder="e.g. Customer interested in enterprise plan, budget 50k, decides end of month...",
            key="n_text"
        )

# ── Run Analysis ───────────────────────────────────────────────
st.markdown("---")

if st.button("🚀 Run Sales Analysis", use_container_width=True, type="primary"):
    if not transcript_text and not notes_text:
        st.error("❌ Please provide at least a transcript or sales notes!")
    else:
        with st.spinner("🤖 Sales Agent is analyzing everything... please wait..."):
            try:
                result = run_sales_agent(transcript_text, notes_text)

                st.success("✅ Analysis complete!")
                st.markdown("---")
                st.markdown("## 📊 Sales Agent Full Analysis")
                st.markdown(result)
                st.markdown("---")

                st.download_button(
                    label="⬇️ Download Full Analysis",
                    data=result,
                    file_name="sales_analysis.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ── How to get Teams transcript ────────────────────────────────
with st.expander("❓ How to get transcript from Microsoft Teams"):
    st.markdown("""
    **Method 1 — Download from Teams:**
    1. Open Microsoft Teams
    2. Go to the meeting chat
    3. Click the recording
    4. Click **Transcript** on the right panel
    5. Click **Download** → save as `.docx`
    6. Upload the `.docx` file here

    **Method 2 — Copy and paste:**
    1. Open the meeting recording in Teams
    2. Click **Transcript** panel
    3. Select all → Copy
    4. Paste in the text area above

    **Method 3 — Take a photo:**
    1. Take a photo of your screen showing the transcript
    2. Upload the image here
    3. AI will read the text automatically
    """)
