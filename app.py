import streamlit as st
from google import genai
from google.genai import types
from PIL import Image

st.set_page_config(page_title="JEE/NEET First-Principles Physics Mentor", page_icon="⚛️", layout="centered")

st.title("⚛️ First-Principles Physics Mentor")
st.caption("Built by Harpreet Prajapati | Structured step-by-step Physics solver for JEE & NEET")

api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("Missing GEMINI_API_KEY. Please add it in your Streamlit app Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

with st.sidebar:
    st.header("⚙️ Mentor Settings")
    exam_mode = st.selectbox("Target Exam", ["JEE Main", "JEE Advanced", "NEET"])
    help_mode = st.radio("Help Mode", ["Full Solution", "Hint Only"])
    st.markdown("---")
    st.markdown("**How to use:** Type a question, upload a diagram screenshot, or both.")

question_text = st.text_area(
    "Enter your Physics question:",
    placeholder="Example: A uniform rod of mass M and length L is hinged at one end...",
    height=130
)

uploaded_file = st.file_uploader("Upload question or diagram image (optional)", type=["png", "jpg", "jpeg"])

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image.thumbnail((800, 800))
    st.image(image, caption="Uploaded Question", use_container_width=True)

SYSTEM_PROMPT = """
You are an expert JEE & NEET Physics Mentor. Your goal is to build first-principles intuition, never just dump final answers.

Rules:
1. Never state the final numerical answer at the beginning.
2. Always format equations cleanly using LaTeX ($...$ for inline math, $$...$$ for block equations).
3. Adapt rigor to the selected exam:
   - NEET: Focus on conceptual clarity, direct formula intuition, and fast calculation tricks.
   - JEE Main: Balance conceptual setup with clean algebraic manipulation.
   - JEE Advanced: Emphasize multi-concept synthesis, calculus-based derivations, and edge cases.
4. If the user selects "Hint Only", provide ONLY Section 1 and Section 2, then ask a guiding question to help them finish the problem.
5. If the user selects "Full Solution", output strictly in these four markdown sections:
   ### 1. Core Concept & Why It Applies
   ### 2. System Setup & Reference Frame
   ### 3. Step-by-Step Derivation
   ### 4. Common Exam Trap & Sanity Check
"""

if st.button("Solve with First Principles", type="primary", use_container_width=True):
    if not question_text.strip() and image is None:
        st.warning("Please type a Physics question or upload an image first.")
    else:
        user_prompt = f"Target Exam: {exam_mode}\nHelp Mode: {help_mode}\n\nQuestion:\n{question_text}"
        
        contents = [user_prompt]
        if image is not None:
            contents.append(image)

        st.markdown("---")
        output_placeholder = st.empty()
        full_text = ""

        try:
            with st.spinner("Connecting to Physics Mentor..."):
                stream = client.models.generate_content_stream(
                    model="gemini-3.8-flash",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.2,
                        thinking_config=types.ThinkingConfig(thinking_level="LOW"),
                    ),
                )
            for chunk in stream:
                if chunk.text:
                    full_text += chunk.text
                    output_placeholder.markdown(full_text)
        except Exception as e:
            st.error(f"Could not complete request: {e}")
