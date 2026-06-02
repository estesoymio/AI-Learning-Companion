import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful learning companion in an ongoing conversation.
    Answer the student's latest question using only the provided textbook context.
    Use the conversation history to understand follow-up questions. If the answer
    is not supported by the context, say that you could not find it in the material.

    Conversation history:
    {chat_history}

    <context>
    {context}
    </context>

    Latest question: {input}
    """
)

quiz_prompt = ChatPromptTemplate.from_template(
    """
    Create a {difficulty} {quiz_type} quiz about the topic "{topic}".
    Use only the provided context. Do not add facts that are not supported
    by the context. Create exactly {question_count} questions.

    Format the response with these exact headings:
    QUESTIONS
    1. ...

    ANSWER KEY
    1. ...

    For multiple choice questions, provide four options labeled A, B, C,
    and D under each question. Do not reveal answers in the QUESTIONS section.
    If there is not enough relevant context for the topic, say so instead of
    creating unsupported questions.

    <context>
    {context}
    </context>
    """
)

flashcard_prompt = ChatPromptTemplate.from_template(
    """
    Create exactly {card_count} {detail_level} flashcards about the topic "{topic}".
    Use only the provided context. Each flashcard should help a student study an
    important idea, definition, process, example, or relationship from the topic.
    Do not add facts that are not supported by the context.

    Format each card exactly like this:
    CARD 1
    FRONT: Question or term
    BACK: Clear answer or explanation

    CARD 2
    FRONT: Question or term
    BACK: Clear answer or explanation

    If there is not enough relevant context for the topic, say so instead of
    creating unsupported flashcards.

    <context>
    {context}
    </context>
    """
)


@st.cache_resource(show_spinner="Loading PDF chapters...")
def load_pdf_documents():
    docs = []

    for pdf_file in Path("data").glob("*.pdf"):
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            loaded_docs = loader.load()
            chapter_name = pdf_file.stem

            for doc in loaded_docs:
                doc.metadata["chapter"] = chapter_name

            docs.extend(loaded_docs)
        except Exception as e:
            st.error(f"Failed: {pdf_file.name}")
            st.write(e)

    return docs


def create_vector_embedding():
    docs = load_pdf_documents()

    if not docs:
        st.error("No PDF documents were loaded from the data folder.")
        return

    selected_chapter = st.session_state.selected_chapter
    if selected_chapter != "All Chapters":
        docs = [
            doc for doc in docs
            if doc.metadata.get("chapter") == selected_chapter
        ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    final_documents = text_splitter.split_documents(docs)

    st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.final_documents = final_documents
    st.session_state.vectors = FAISS.from_documents(
        final_documents,
        st.session_state.embeddings
    )
    st.session_state.indexed_chapter = selected_chapter
    st.session_state.chat_messages = []
    st.session_state.pop("quiz_response", None)
    st.session_state.pop("flashcard_response", None)


def vector_database_ready():
    if "vectors" not in st.session_state:
        st.warning("Please click 'Document Embedding' first and make sure PDFs load successfully.")
        return False

    if st.session_state.get("indexed_chapter") != st.session_state.selected_chapter:
        st.warning("The selected chapter changed. Click 'Document Embedding' again before continuing.")
        return False

    return True


def split_quiz_response(quiz_response):
    heading = "ANSWER KEY"

    if heading not in quiz_response:
        return quiz_response, ""

    questions, answer_key = quiz_response.split(heading, maxsplit=1)
    questions = questions.replace("QUESTIONS", "", 1).strip()
    return questions, answer_key.strip()


def parse_flashcards(flashcard_response):
    cards = []

    for block in flashcard_response.split("CARD ")[1:]:
        if "FRONT:" not in block or "BACK:" not in block:
            continue

        front_and_back = block.split("FRONT:", maxsplit=1)[1]
        front, back = front_and_back.split("BACK:", maxsplit=1)
        cards.append({"front": front.strip(), "back": back.strip()})

    return cards


def format_chat_history(messages):
    recent_messages = messages[-8:]
    return "\n".join(
        f"{message['role'].title()}: {message['content']}"
        for message in recent_messages
    )


def answer_chat_question(user_prompt):
    chat_history = format_chat_history(st.session_state.chat_messages[:-1])
    retrieval_query = f"{chat_history}\nStudent: {user_prompt}".strip()
    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 4})
    context_docs = retriever.invoke(retrieval_query)
    document_chain = create_stuff_documents_chain(llm, prompt)

    start = time.process_time()
    answer = document_chain.invoke(
        {
            "input": user_prompt,
            "chat_history": chat_history or "No earlier messages.",
            "context": context_docs,
        }
    )

    return {
        "content": answer,
        "context": context_docs,
        "response_time": time.process_time() - start,
    }


st.title("AI Learning Companion")

all_docs = load_pdf_documents()
chapters = sorted({doc.metadata["chapter"] for doc in all_docs})

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

with st.sidebar:
    st.header("Study Material")
    st.selectbox(
        "Select Chapter",
        ["All Chapters"] + chapters,
        key="selected_chapter"
    )

    if st.button("Document Embedding", use_container_width=True):
        create_vector_embedding()

        if "vectors" in st.session_state:
            st.success(f"Ready for {st.session_state.indexed_chapter}.")
        else:
            st.error("Vector Database was not created. Check PDF loading errors.")

    if "vectors" in st.session_state:
        st.caption(f"Embedded: {st.session_state.indexed_chapter}")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_messages = []

ask_tab, quiz_tab, flashcard_tab = st.tabs(
    ["Chat", "Generate Quiz", "Flashcards"]
)

with ask_tab:
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                if "response_time" in message:
                    st.caption(f"Response time: {message['response_time']:.2f} seconds")

                with st.expander("Source Passages"):
                    for doc in message.get("context", []):
                        st.write(f"Chapter: {doc.metadata.get('chapter', 'Unknown')}")
                        st.write(doc.page_content)
                        st.write("------------------------")

    user_prompt = st.chat_input("Ask about your study material")

    if user_prompt:
        st.session_state.chat_messages.append(
            {"role": "user", "content": user_prompt}
        )

        with st.chat_message("user"):
            st.markdown(user_prompt)

        if vector_database_ready():
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = answer_chat_question(user_prompt)
                st.markdown(response["content"])
                st.caption(f"Response time: {response['response_time']:.2f} seconds")

                with st.expander("Source Passages"):
                    for doc in response["context"]:
                        st.write(f"Chapter: {doc.metadata.get('chapter', 'Unknown')}")
                        st.write(doc.page_content)
                        st.write("------------------------")

            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": response["content"],
                    "context": response["context"],
                    "response_time": response["response_time"],
                }
            )

with quiz_tab:
    with st.form("quiz_form"):
        topic = st.text_input("Quiz topic", placeholder="For example: force and pressure")
        question_count = st.slider("Number of questions", 3, 10, 5)
        difficulty = st.selectbox("Difficulty", ["Beginner", "Intermediate", "Advanced"])
        quiz_type = st.selectbox("Question type", ["Multiple choice", "Short answer", "Mixed"])
        quiz_submitted = st.form_submit_button("Generate Quiz")

    if quiz_submitted and topic and vector_database_ready():
        retriever = st.session_state.vectors.as_retriever(
            search_kwargs={"k": min(max(question_count * 2, 4), 10)}
        )

        with st.spinner("Creating your quiz..."):
            context_docs = retriever.invoke(topic)
            quiz_chain = create_stuff_documents_chain(llm, quiz_prompt)
            quiz_response = quiz_chain.invoke(
                {
                    "context": context_docs,
                    "topic": topic,
                    "difficulty": difficulty.lower(),
                    "quiz_type": quiz_type.lower(),
                    "question_count": question_count,
                }
            )
            st.session_state.quiz_response = {
                "content": quiz_response,
                "topic": topic,
                "context": context_docs,
            }

    if "quiz_response" in st.session_state:
        quiz_response = st.session_state.quiz_response
        questions, answer_key = split_quiz_response(quiz_response["content"])

        st.subheader(f"Quiz: {quiz_response['topic']}")
        st.markdown(questions)

        if answer_key:
            with st.expander("Show Answer Key"):
                st.markdown(answer_key)

        with st.expander("Quiz Source Passages"):
            for doc in quiz_response["context"]:
                st.write(f"Chapter: {doc.metadata.get('chapter', 'Unknown')}")
                st.write(doc.page_content)
                st.write("------------------------")

with flashcard_tab:
    with st.form("flashcard_form"):
        flashcard_topic = st.text_input(
            "Flashcard topic",
            placeholder="For example: crop production"
        )
        card_count = st.slider("Number of cards", 3, 12, 6)
        detail_level = st.selectbox(
            "Card style",
            ["Quick review", "Detailed study", "Exam preparation"]
        )
        flashcard_submitted = st.form_submit_button("Generate Flashcards")

    if flashcard_submitted and flashcard_topic and vector_database_ready():
        retriever = st.session_state.vectors.as_retriever(
            search_kwargs={"k": min(max(card_count, 4), 10)}
        )

        with st.spinner("Preparing your flashcards..."):
            context_docs = retriever.invoke(flashcard_topic)
            flashcard_chain = create_stuff_documents_chain(llm, flashcard_prompt)
            flashcard_content = flashcard_chain.invoke(
                {
                    "context": context_docs,
                    "topic": flashcard_topic,
                    "detail_level": detail_level.lower(),
                    "card_count": card_count,
                }
            )
            st.session_state.flashcard_response = {
                "content": flashcard_content,
                "topic": flashcard_topic,
                "context": context_docs,
            }

    if "flashcard_response" in st.session_state:
        flashcard_response = st.session_state.flashcard_response
        cards = parse_flashcards(flashcard_response["content"])

        st.subheader(f"Flashcards: {flashcard_response['topic']}")

        if cards:
            for index, card in enumerate(cards, start=1):
                st.markdown(f"**Card {index}: {card['front']}**")
                with st.expander("Reveal Answer"):
                    st.write(card["back"])
        else:
            st.write(flashcard_response["content"])

        with st.expander("Flashcard Source Passages"):
            for doc in flashcard_response["context"]:
                st.write(f"Chapter: {doc.metadata.get('chapter', 'Unknown')}")
                st.write(doc.page_content)
                st.write("------------------------")
