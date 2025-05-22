import streamlit as st
import google.generativeai as genai
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Note Maker",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📝 AI Note Maker with Gemini")
st.markdown("Enter your text or topic below, and let AI generate concise notes for you!")

# --- Sidebar for API Key Configuration ---
st.sidebar.header("🔑 API Configuration")
st.sidebar.markdown("""
You need a Gemini API key to use this app.
You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).
""")

api_key = st.sidebar.text_input(
    "Enter your Gemini API Key:",
    type="password",
    key="api_key_input",
    help="Your API key is stored temporarily and used only for this session."
)

gemini_configured = False
if api_key:
    try:
        genai.configure(api_key=api_key)
        # Optional: A light check to see if the API key is somewhat valid by listing models
        # model_list = [m.name for m in genai.list_models()]
        # if not model_list:
        #     raise ValueError("No models found. API key might be invalid or have no permissions.")
        st.sidebar.success("Gemini API Key configured successfully!")
        gemini_configured = True
    except Exception as e:
        st.sidebar.error(f"Failed to configure Gemini API: {str(e)}")
        gemini_configured = False
else:
    st.sidebar.info("Please enter your Gemini API Key to enable note generation.")

# --- Main Area for Input and Output ---
st.subheader("🖋️ Your Input")
user_input_text = st.text_area(
    "Enter the text, topic, or question you want notes on:",
    height=200,
    key="user_note_input",
    placeholder="For example: 'The future of renewable energy sources' or paste a long article here..."
)

if 'generated_notes' not in st.session_state:
    st.session_state.generated_notes = ""

col1, col2 = st.columns([1,5]) # Adjust column ratio as needed

with col1:
    if st.button("✨ Generate Notes", type="primary", disabled=not gemini_configured or not user_input_text.strip(), use_container_width=True):
        if gemini_configured and user_input_text.strip():
            prompt = f"""
            You are an expert note-taking assistant.
            Based on the following input, please generate clear, concise, and well-structured notes.
            The notes should capture the key points, main ideas, and any important details.
            Use bullet points, numbered lists, or other structured formats where appropriate for readability.
            Ensure the notes are easy to understand and summarize the core information effectively.

            Input:
            ---
            {user_input_text}
            ---

            Generated Notes:
            """
            try:
                with st.spinner("AI is crafting your notes... ✨"):
                    model = genai.GenerativeModel('gemini-1.5-flash-latest') # Or 'gemini-pro'
                    response = model.generate_content(prompt)
                    st.session_state.generated_notes = response.text
                st.success("Notes generated successfully!")
            except Exception as e:
                st.error(f"An error occurred while generating notes: {str(e)}")
                st.session_state.generated_notes = f"Error: Could not generate notes. {str(e)}"

if st.session_state.generated_notes:
    st.subheader("📄 Generated Notes")
    st.markdown(st.session_state.generated_notes)
    if st.button("🗑️ Clear Notes", use_container_width=True):
        st.session_state.generated_notes = ""
        st.rerun()
