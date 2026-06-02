# AI Learning Companion

An AI-powered educational assistant built using Retrieval-Augmented Generation (RAG) that enables students to interact with learning materials through natural language.

## Overview

AI Learning Companion allows users to upload educational documents and ask questions about the content. The application uses semantic search and large language models to generate context-aware answers grounded in the uploaded material.

The project is designed to evolve beyond a generic PDF chatbot into a chapter-aware learning platform supporting quizzes, revision notes, flashcards, and personalized learning experiences.

## Features

### Current Features

* Upload educational PDFs
* Automatic document ingestion and processing
* Text chunking and embedding generation
* Semantic search using vector similarity
* Context-aware question answering
* Conversational interface built with Streamlit
* Groq LLM integration via LangChain

### Planned Features

* Chapter-aware retrieval
* Quiz generation
* Flashcard generation
* Revision note generation
* Difficulty-based explanations
* Student progress tracking
* Personalized learning recommendations

## Tech Stack

### Frontend

* Streamlit

### AI & NLP

* LangChain
* Groq LLM
* Embeddings

### Retrieval

* Retrieval-Augmented Generation (RAG)
* Vector Similarity Search

### Document Processing

* PyPDF
* Text Chunking

### Storage

* ChromaDB / Vector Database

## Architecture

User Query
→ Retriever
→ Relevant Document Chunks
→ Groq LLM
→ Context-Aware Response

Document Upload
→ PDF Processing
→ Chunking
→ Embeddings
→ Vector Store

## Project Structure

```text
project/
│
├── app.py
├── requirements.txt
├── data/
├── vectorstore/
├── utils/
│   ├── loader.py
│   ├── splitter.py
│   ├── embeddings.py
│   └── retrieval.py
│
└── README.md
```

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd ai-learning-companion
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate environment:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure environment variables:

```env
GROQ_API_KEY=your_api_key
```

Run the application:

```bash
streamlit run app.py
```

## Example Workflow

1. Upload a chapter PDF
2. Generate embeddings
3. Ask questions about the content
4. Receive contextual answers grounded in the document
5. Generate quizzes and revision materials (upcoming)

## Future Roadmap

### Phase 1

* PDF Question Answering
* Semantic Retrieval
* Groq Integration

### Phase 2

* Chapter-Aware Retrieval
* Quiz Generation
* Flashcards

### Phase 3

* Personalized Learning
* Progress Analytics
* Adaptive Tutoring

### Phase 4

* Multi-Agent Learning System
* Learning Recommendations
* Study Planning Assistant

## Author

Sindooja Govindaraju

Built as part of a journey into Generative AI, RAG systems, and AI-powered educational technology.
