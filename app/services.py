"""AI and vector-search services."""
from __future__ import annotations

import time
import json
import os
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

from app.core import build_rag_prompt, format_sources

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


def get_api_key() -> str | None:
    """Read the key from environment or Streamlit secrets."""
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        return None


def get_llm(api_key: str, model_name: str = DEFAULT_GEMINI_MODEL):
    if not api_key:
        raise ValueError("Google Gemini API key is missing.")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2,
        max_retries=2,
        request_timeout=60,
    )


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vector_store(chunks: list[Document]) -> FAISS:
    if not chunks:
        raise ValueError("No readable text chunks were created from the PDFs.")
    return FAISS.from_documents(chunks, get_embeddings())


def contextualize_question(
    llm: Any,
    question: str,
    chat_history: str
) -> str:
    if not chat_history.strip():
        return question

    prompt = f"""You are a question reformulation assistant.

Rewrite the latest user question as a standalone question
using the conversation history to resolve references.

Rules:
- Do not answer the question.
- Preserve the original meaning.
- Resolve references such as "it", "that", "the second one".
- If the question is already self-contained, return it unchanged.
- Return ONLY the rewritten question.

CONVERSATION HISTORY:
{chat_history}

LATEST QUESTION:
{question}

REWRITTEN QUESTION:"""

    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))

        if isinstance(content, list):
            content = "\n".join(
                str(part.get("text", part))
                if isinstance(part, dict) else str(part)
                for part in content
            )

        rewritten = str(content).strip()

        return rewritten if rewritten else question

    except Exception:
        return question


def answer_question(
    vector_store: FAISS,
    llm: Any,
    question: str,
    k: int = 8,
    chat_history: str = ""
) -> tuple[str, list[str]]:

    contextualized_question = contextualize_question(
        llm, question, chat_history
    )

    matches = vector_store.similarity_search(
        contextualized_question, k=k
    )

    if not matches:
        return "Information not found in the uploaded document.", []

    context = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'Unknown')}, "
        f"page: {d.metadata.get('page', 'Unknown')}]\n{d.page_content}"
        for d in matches
    )

    response = llm.invoke(
        build_rag_prompt(question, context, chat_history)
    )

    answer = getattr(response, "content", str(response))

    if isinstance(answer, list):
        answer = "\n".join(
            str(part.get("text", part)) if isinstance(part, dict) else str(part)
            for part in answer
        )

    return str(answer), format_sources(matches)





def summarize_documents(chunks: list[Document], llm: Any) -> str:
    """Generate a concise, topic-wise overview of the uploaded PDF."""

    if not chunks:
        raise ValueError("There is no document text to summarize.")

    def extract_text(response: Any) -> str:
        """Convert Gemini's response into clean Markdown text."""
        content = getattr(response, "content", response)

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
                else:
                    parts.append(str(item))
            return "\n".join(parts).strip()

        return str(content).strip()

    batch_size = 6
    partials = []

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        context = "\n\n".join(
            f"[Page {d.metadata.get('page', 'Unknown')}]\n{d.page_content}"
            for d in batch
        )

        prompt = f"""
You are an expert academic tutor.

Create a CONCISE TOPIC-WISE OVERVIEW of the provided
PDF excerpts using ONLY the source material.

STRICT REQUIREMENTS:

1. Identify all major topics covered in the excerpts.
2. Organise relevant subtopics under their major topics.
3. Give each topic and subtopic a brief, clear description.
4. Explain concepts in simple, student-friendly language.
5. Preserve important technical terms and definitions.
6. Cover the important concepts without unnecessary detail.
7. Use clear Markdown headings and subheadings.
8. Avoid repeating the same information.

DO NOT INCLUDE:

- Numerical problems or calculations.
- Worked-out solutions.
- Examples or case studies.
- Practice questions or quiz questions.
- Lengthy mathematical derivations.
- Additional information not present in the source.

IMPORTANT:
This is NOT a detailed study guide or textbook.
It must communicate the gist of the material, not
reproduce or expand upon the original PDF.

SOURCE EXCERPTS:

{context}
"""

        start_time = time.time()
        print(f"Generating summary for batch {start // batch_size + 1}...")

        response = llm.invoke(prompt)

        print(
            f"Batch completed in {time.time() - start_time:.2f} seconds."
        )

        partials.append(extract_text(response))

    if len(partials) == 1:
        return partials[0]

    final_prompt = f"""
You are an academic editor.

Combine the following partial topic-wise overviews
into ONE concise and complete PDF summary.

REQUIREMENTS:

1. Group related topics together.
2. Arrange topics in a logical order.
3. Place subtopics under their appropriate topics.
4. Give each topic and subtopic a brief description.
5. Remove repeated or overlapping information.
6. Retain all important topics from the source.
7. Use simple, student-friendly language.
8. Use Markdown headings and subheadings.

STRICTLY EXCLUDE:

- Numerical problems and calculations.
- Worked solutions.
- Examples and case studies.
- Practice or quiz questions.
- Lengthy explanations and derivations.

Do not introduce outside information.

Ensure that every major topic and important subtopic present in the 
source document is represented in the summary. 
Do not omit a topic simply to make the summary shorter. 
Prioritise complete topic coverage while keeping individual descriptions concise.

The final result must be a concise overview of WHAT
the PDF covers, not a detailed explanation of HOW
to solve its problems.

PARTIAL SUMMARIES:

{"".join(chr(10) + chr(10) + p for p in partials)}
"""

    start_time = time.time()
    print("Combining partial summaries...")

    response = llm.invoke(final_prompt)

    print(
        f"Final summary completed in {time.time() - start_time:.2f} seconds."
    )

    return extract_text(response)




def generate_quiz(
    chunks: list[Document],
    llm: Any,
    mcq_count: int = 5,
    short_count: int = 5,
    previous_questions: list[str] | None = None
) -> dict:
    if not chunks:
        raise ValueError("There is no document text for quiz generation.")

    # Select up to 20 chunks evenly across the entire document.
    max_chunks = 20

    if len(chunks) <= max_chunks:
        selected_chunks = chunks
    else:
        indices = [
            round(i * (len(chunks) - 1) / (max_chunks - 1))
            for i in range(max_chunks)
        ]
        selected_chunks = [chunks[i] for i in indices]

    excerpt = "\n\n".join(d.page_content for d in selected_chunks)
    previous_questions = previous_questions or []

    previous_question_text = "\n".join(
        f"- {question}" for question in previous_questions[-30:]
    )

    if not previous_question_text:
        previous_question_text = "No previous questions."

    prompt = f"""
Create a study quiz using ONLY the source text below.

Generate exactly {mcq_count} multiple-choice questions
and {short_count} short-answer questions.

Return ONLY valid JSON. Do not use Markdown code fences.

Follow this exact structure:

{{
    "mcqs": [
        {{
            "question": "Question text",
            "options": {{
                "A": "Option A",
                "B": "Option B",
                "C": "Option C",
                "D": "Option D"
            }},
            "correct_answer": "C",
            "explanation": "Brief explanation of the correct answer."
        }}
    ],
    "short_answers": [
        {{
            "question": "Question text",
            "model_answer": "Concise model answer."
        }}
    ]
}}

Rules:
- Every MCQ must have exactly four options.
- Only one option must be correct.
- correct_answer must contain only A, B, C or D.
- Explanations must be based on the source.
- Short answers must be concise and accurate.
- Avoid duplicate questions.
- Do not introduce facts absent from the source.
- Cover different topics and concepts across the entire source document.
- Distribute questions across the document rather than focusing on its opening sections.
- Avoid testing the same concept more than once unless necessary.
- Vary question styles, including conceptual understanding,
  application, comparison, and factual recall.
- Do not simply rephrase another question.
- Prioritise concepts that have not been tested in the other questions.

PREVIOUSLY GENERATED QUESTIONS:
{previous_question_text}

IMPORTANT:
- Do not repeat any previously generated question.
- Do not create paraphrased versions of previous questions.
- Test different concepts from those covered previously.
- If a concept has already been tested, prioritise another concept.

SOURCE TEXT:
{excerpt}
"""

    response = llm.invoke(prompt)
    content = response.content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")
            else:
                text = getattr(item, "text", "")

            if text:
                text_parts.append(text)

        content = "\n".join(text_parts)

    if not isinstance(content, str):
        raise ValueError("Gemini returned an unsupported quiz response.")

    # Remove Markdown code fences if Gemini includes them.
    content = content.strip()

    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content[:-3]

    try:
        quiz = json.loads(content.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Could not interpret the quiz generated by Gemini. Please try again."
        ) from exc

    
    if not isinstance(quiz, dict):
        raise ValueError("Gemini returned an invalid quiz structure.")

    mcqs = quiz.get("mcqs")
    short_answers = quiz.get("short_answers")

    if not isinstance(mcqs, list):
        raise ValueError("MCQ section is missing or invalid.")

    if not isinstance(short_answers, list):
        raise ValueError("Short-answer section is missing or invalid.")

    # Verify the requested question counts.
    if len(mcqs) != mcq_count:
        raise ValueError(
            f"Expected {mcq_count} MCQs, but Gemini generated {len(mcqs)}."
        )

    if len(short_answers) != short_count:
        raise ValueError(
            f"Expected {short_count} short-answer questions, "
            f"but Gemini generated {len(short_answers)}."
        )

    # Validate every multiple-choice question.
    for i, question in enumerate(mcqs, start=1):
        if not isinstance(question, dict):
            raise ValueError(f"MCQ {i} has an invalid structure.")

        if not isinstance(question.get("question"), str) or not question["question"].strip():
            raise ValueError(f"MCQ {i} is missing its question text.")

        options = question.get("options")

        if not isinstance(options, dict) or set(options.keys()) != {"A", "B", "C", "D"}:
            raise ValueError(f"MCQ {i} must contain exactly four options: A, B, C and D.")

        if not all(
            isinstance(options[key], str) and options[key].strip()
            for key in ("A", "B", "C", "D")
        ):
            raise ValueError(f"MCQ {i} contains an empty or invalid option.")

        if question.get("correct_answer") not in ("A", "B", "C", "D"):
            raise ValueError(f"MCQ {i} has an invalid correct answer.")

        if not isinstance(question.get("explanation"), str) or not question["explanation"].strip():
            raise ValueError(f"MCQ {i} is missing its explanation.")

    # Validate every short-answer question.
    for i, question in enumerate(short_answers, start=1):
        if not isinstance(question, dict):
            raise ValueError(f"Short-answer question {i} has an invalid structure.")

        if not isinstance(question.get("question"), str) or not question["question"].strip():
            raise ValueError(f"Short-answer question {i} is missing its question text.")

        if not isinstance(question.get("model_answer"), str) or not question["model_answer"].strip():
            raise ValueError(f"Short-answer question {i} is missing its model answer.")

    return quiz

