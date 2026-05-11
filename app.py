import streamlit as st
from groq import Groq
import json

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
</style>
""", unsafe_allow_html=True)

st.title("📝 Sales Meeting Note Agent")
st.caption("Paste rough notes → get summary, action items, email & next steps")
st.divider()

st.subheader("Step 1 — Groq API Key")
st.markdown("Get free key at [console.groq.com](https://console.groq.com)")

if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
    st.success("✅ API key loaded from secrets")
else:
    api_key = st.text_input("Enter Groq API key", type="password", placeholder="gsk_...")

st.divider()
st.subheader("Step 2 — Meeting Details")
col1, col2, col3 = st.columns(3)
with col1: client  = st.text_input("Client name",   placeholder="e.g. Acme Corp")
with col2: rep     = st.text_input("Sales rep name", placeholder="e.g. Ravi Kumar")
with col3: date    = st.text_input("Meeting date",   placeholder="e.g. 11 May 2026")

st.divider()
st.subheader("Step 3 — Paste Your Raw Notes")
st.caption("Messy bullet points, fragments, shorthand — all fine.")
notes = st.text_area("Notes", height=200, label_visibility="collapsed",
    placeholder="- Client needs 20 new hires\n- Budget 5L, CFO approval needed\n- Pain: slow recruitment\n- Send pricing deck by Friday\n- Demo next week")

st.divider()
st.subheader("Step 4 — Analyze")

if st.button("✨ Analyze my notes", use_container_width=True, type="primary"):
    if not api_key:
        st.error("Please enter your Groq API key.")
    elif not notes.strip():
        st.error("Please paste your meeting notes.")
    else:
        cn = client.strip() or "the client"
        rn = rep.strip()    or "our rep"
        dt = date.strip()   or "today"

        prompt = f"""You are a sales assistant for a service-based company.
Process these raw meeting notes and return ONLY a valid JSON object.
No markdown, no backticks, no extra text — just the JSON.

Meeting: Client={cn}, Rep={rn}, Date={dt}
Notes: {notes}

Return this exact JSON structure:
{{
  "summary": "2-3 sentence summary of discussion and deal status",
  "action_items": "Bullet list: • [Owner]: [Task] — [Deadline]",
  "follow_up_email": "Ready-to-send email. First line: Subject: ...",
  "next_steps": "2-3 next steps + deal signals or risk flags"
}}"""

        with st.spinner("Analyzing your notes with Groq AI..."):
            try:
                gc  = Groq(api_key=api_key)
                res = gc.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.3, max_tokens=1024
                )
                raw    = res.choices[0].message.content.strip()
                raw    = raw.replace("```json","").replace("```","").strip()
                result = json.loads(raw)

                st.divider()
                st.subheader("📋 Results")

                sections = [
                    ("🗒️ Meeting Summary",       "summary",       "summary.txt"),
                    ("✅ Action Items",            "action_items",  "action_items.txt"),
                    ("📧 Follow-up Email",         "follow_up_email","followup_email.txt"),
                    ("🚀 Next Steps & Deal Signals","next_steps",    "next_steps.txt"),
                ]
                for label, key, fname in sections:
                    st.markdown(f"**{label}**")
                    content = result.get(key, "")
                    st.markdown(f'<div class="output-box">{content}</div>', unsafe_allow_html=True)
                    st.download_button(f"⬇ Download", content, file_name=fname, mime="text/plain", key=fname)
                    st.markdown("---")

                st.success("Done! Copy or download any section above.")

            except json.JSONDecodeError:
                st.error("Could not parse response. Please try again.")
            except Exception as e:
                st.error(f"Error: {e}")

st.divider()
st.caption("Built with Streamlit + Groq AI (LLaMA 3.3 70B)")
