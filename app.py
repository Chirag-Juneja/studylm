import streamlit as st
from studylm.utils.logger import get_logger
from studylm.studylm import StudyLM
from studylm.utils.file import to_tempfile
from studylm.utils.parser import stream_parser, remove_think_block


logger = get_logger(__name__)


st.set_page_config(
    page_title="StudyLM - Agentic AI Study Assistant", page_icon="📚", layout="wide"
)

# Initialize Session
if "studylm" not in st.session_state:
    with st.spinner("🚀 Initializing StudyLM Agent..."):
        st.session_state.studylm = StudyLM()

st.title("StudyLM - Agentic AI Study Assistant")
st.markdown("*Powered by Ollama with Intelligent Tool Selection*")
st.markdown("---")


col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.header("📁 Document Library")
    uploaded_file = st.file_uploader(
        "Upload PDF Documents",
        type="pdf",
        accept_multiple_files=False,
        help="Upload PDF documents to chat with their content",
    )
    if uploaded_file:
        if not any(
            doc == uploaded_file.name for doc in st.session_state.studylm.uploaded_docs
        ):
            if st.button(
                f"📄 Process {uploaded_file.name}", key=f"process_{uploaded_file.name}"
            ):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    temp_path = to_tempfile(uploaded_file.getvalue())
                    success = st.session_state.studylm.upload_document(
                        temp_path, uploaded_file.name
                    )

                    if success:
                        st.success(f"✅ Processed successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Processing failed")

    if st.session_state.studylm.uploaded_docs:
        st.subheader("📚 Uploaded Documents")
        for doc in st.session_state.studylm.uploaded_docs:
            st.write(f"📄 {doc}")

if "prompt_buffer" not in st.session_state:
    st.session_state.prompt_buffer = None


with col2:
    st.header("AI Assistant")
    with st.container(height=720):
        try:
            for msg in st.session_state.studylm.get_state().values["messages"]:
                if msg.type == "tool":
                    continue
                role = "user" if msg.type == "human" else "assistant"
                with st.chat_message(role):
                    if role == "human":
                        st.markdown(msg.content)
                    else:
                        st.markdown(remove_think_block(msg.content))
        except Exception as e:
            pass

        if st.session_state.prompt_buffer:
            _prompt = st.session_state.prompt_buffer
            with st.chat_message("user"):
                st.markdown(_prompt)
            with st.chat_message("assistant"):
                response = st.session_state.studylm.stream(_prompt)
                st.write_stream(stream_parser(response))
            st.session_state.prompt_buffer = None

    prompt = st.chat_input("Ask a question")
    if prompt:
        st.session_state.prompt_buffer = prompt
        st.rerun()

with col3:
    st.header("Relavent Youtube Videos")
    try:
        for link in st.session_state.studylm.get_state().values["yt_links"]:
            st.video(link)
    except:
        pass

