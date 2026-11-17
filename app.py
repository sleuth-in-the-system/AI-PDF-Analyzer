from __future__ import annotations
import streamlit as st

from app.core import (
    MAX_FILES, PDFProcessingError, extract_pdf_pages, split_pages,
)
from app.services import (
    answer_question, create_vector_store, generate_quiz, get_api_key,
    get_llm, summarize_documents,
)

st.set_page_config(page_title="PDF Chatbot", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    div[data-baseweb="input"]:focus-within {
        border-color: #22c55e !important;
        box-shadow: 0 0 0 1px #22c55e !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("📚 AI-Powered PDF Chatbot")
st.caption("Upload your PDFs, ask questions, create summaries and generate practice quizzes.")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = get_api_key()
    if not api_key:
        api_key = st.text_input("Google Gemini API key", type="password",
                                help="Used only for this session. Prefer an environment variable or Streamlit secrets.")
    model_name = st.text_input("Gemini model", value="gemini-3.5-flash-lite")
    st.divider()
    st.caption("PDF limit: 25 MB per file • Up to 10 files")
    st.caption("Scanned-PDF OCR is not included.")

uploaded_files = st.file_uploader(
    "Upload one or more PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="Use text-based PDFs. Maximum 10 files, 25 MB each.",
)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chat" not in st.session_state:
    st.session_state.chat = []
if "processed_names" not in st.session_state:
    st.session_state.processed_names = []

col1, col2 = st.columns([1, 1])
with col1:
    process_clicked = st.button("🔎 Process PDFs", type="primary",
                                disabled=not uploaded_files or not api_key)
with col2:
    if st.button("♻️ Reset session"):
        for key in ("vector_store", "chunks", "chat", "processed_names"):
            st.session_state[key] = [] if key in ("chunks", "chat", "processed_names") else None
        st.rerun()

if uploaded_files and len(uploaded_files) > MAX_FILES:
    st.error(f"Please upload no more than {MAX_FILES} PDFs.")

if process_clicked:
    if not api_key:
        st.error("Enter your Gemini API key in the sidebar.")
    elif len(uploaded_files) > MAX_FILES:
        st.error(f"Please upload no more than {MAX_FILES} PDFs.")
    else:
        all_pages = []
        errors = []
        progress = st.progress(0)
        for i, uploaded in enumerate(uploaded_files):
            try:
                pages = extract_pdf_pages(uploaded.name, uploaded.getvalue())
                all_pages.extend(pages)
            except PDFProcessingError as exc:
                errors.append(f"{uploaded.name}: {exc}")
            progress.progress((i + 1) / len(uploaded_files))
        for error in errors:
            st.warning(error)
        if all_pages:
            try:
                chunks = split_pages(all_pages)
                with st.spinner("Creating embeddings and FAISS index..."):
                    vector_store = create_vector_store(chunks)
                st.session_state.chunks = chunks
                st.session_state.vector_store = vector_store
                st.session_state.processed_names = list(dict.fromkeys(
                    d.metadata["source"] for d in all_pages
                ))
                st.session_state.chat = []
                st.success(
                    f"Processed {len(st.session_state.processed_names)} PDF(s), "
                    f"{len(all_pages)} text page(s), and {len(chunks)} chunks."
                )
            except Exception as exc:
                st.error(f"Could not build the search index: {exc}")
        else:
            st.error("No usable PDF text was extracted.")

if st.session_state.vector_store is not None:
    st.success("Knowledge base ready: " + ", ".join(st.session_state.processed_names))
    tab_chat, tab_summary, tab_quiz = st.tabs(["💬 Ask questions", "📝 Summary", "🧠 Quiz"])

    try:
        llm = get_llm(api_key, model_name)
    except Exception as exc:
        llm = None
        st.error(f"Could not initialize Gemini: {exc}")

    with tab_chat:
        st.subheader("Chat with your documents")
        for message in st.session_state.chat:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    st.caption("Sources: " + " • ".join(message["sources"]))
        question = st.chat_input("Ask something about your uploaded PDFs...")
        if question:
            st.session_state.chat.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            if llm:
                with st.chat_message("assistant"):
                    with st.spinner("Searching your PDFs and generating an answer..."):
                        try:
                            chat_history = "\n".join(
                                f"{message['role'].upper()}: {message['content']}"
                                for message in st.session_state.chat[:-1]
                            )

                            answer, sources = answer_question(
                                st.session_state.vector_store,
                                llm,
                                question,
                                chat_history=chat_history
                            )
                            st.markdown(answer)
                            if sources:
                                st.caption("Sources: " + " • ".join(sources))
                            st.session_state.chat.append({
                                "role": "assistant", "content": answer, "sources": sources
                            })
                        except Exception as exc:
                            st.error(f"Answer generation failed: {exc}")

    with tab_summary:
        st.subheader("Document summary")
        if st.button("Generate summary"):
            if llm:
                with st.spinner("Preparing your summary..."):
                    try:
                        st.markdown(summarize_documents(st.session_state.chunks, llm))
                    except Exception as exc:
                        st.error(f"Summary generation failed: {exc}")

    
    with tab_quiz:
        if "quiz_question_history" not in st.session_state:
            st.session_state.quiz_question_history = []
        st.subheader("Practice quiz")

        mcq_count = st.number_input(
            "Number of MCQs",
            min_value=1,
            max_value=15,
            value=5
        )

        short_count = st.number_input(
            "Number of short-answer questions",
            min_value=0,
            max_value=15,
            value=5
        )

        if st.button("Generate quiz"):
            if llm:
                with st.spinner("Creating questions from your PDFs..."):
                    try:
                        quiz = generate_quiz(
                            st.session_state.chunks,
                            llm,
                            int(mcq_count),
                            int(short_count),
                            st.session_state.quiz_question_history
                        )

                        new_questions = [
                            item["question"].strip()
                            for section in ("mcqs", "short_answers")
                            for item in quiz.get(section, [])
                            if isinstance(item, dict)
                            and isinstance(item.get("question"), str)
                            and item["question"].strip()
                        ]

                        st.session_state.quiz_question_history.extend(new_questions)
                        # Keep only the latest 30 questions.
                        st.session_state.quiz_question_history = (
                            st.session_state.quiz_question_history[-30:]
                        )

                        st.session_state.quiz_data = quiz
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_version = (
                            st.session_state.get("quiz_version", 0) + 1
                        )

                        st.success("Quiz generated successfully!")

                    except Exception as exc:
                        st.error(f"Quiz generation failed: {exc}")

        quiz = st.session_state.get("quiz_data")

        if quiz:
            mcqs = quiz.get("mcqs", [])
            short_questions = quiz.get("short_answers", [])

            st.divider()
            st.subheader("Multiple-Choice Questions")

            version = st.session_state.get("quiz_version", 0)

            with st.form(f"mcq_form_{version}"):

                for i, question in enumerate(mcqs):
                    st.markdown(
                        f"**{i + 1}. {question['question']}**"
                    )

                    options = question["options"]

                    st.radio(
                        "Choose your answer:",
                        options=list(options.keys()),
                        index=None,
                        format_func=lambda key, opts=options: (
                            f"{key}) {opts[key]}"
                        ),
                        key=f"mcq_{version}_{i}"
                    )

                    st.write("")

                submitted = st.form_submit_button(
                    "Submit Answers",
                    type="primary"
                )

            if submitted:
                st.session_state.quiz_submitted = True

            if st.session_state.get("quiz_submitted", False):

                score = 0

                st.divider()
                st.subheader("Quiz Results")

                for i, question in enumerate(mcqs):
                    selected = st.session_state.get(
                        f"mcq_{version}_{i}"
                    )

                    correct = question["correct_answer"]

                    if selected == correct:
                        score += 1
                        st.success(f"Question {i + 1}: Correct!")
                    else:
                        st.error(f"Question {i + 1}: Incorrect")

                    st.markdown(
                        f"**Correct answer: {correct}) "
                        f"{question['options'][correct]}**"
                    )

                    st.markdown(
                        f"**Explanation:** {question['explanation']}"
                    )

                    st.divider()

                st.subheader(
                    f"Your Score: {score}/{len(mcqs)}"
                )

            st.divider()
            st.subheader("Short-Answer Questions")

            for i, question in enumerate(short_questions):
                st.markdown(
                    f"**{i + 1}. {question['question']}**"
                )

                st.markdown(
                    f"**Model Answer:** {question['model_answer']}"
                )

                st.write("")

else:
    st.info("Upload your PDF files and select **Process PDFs** to begin.")
