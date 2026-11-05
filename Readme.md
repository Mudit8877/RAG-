# RAG Project using LangChain, ChromaDB, and Multiple LLM Providers (Under Construction)

## Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline using LangChain, vector embeddings, ChromaDB, and Large Language Models (LLMs).

The application processes multiple text documents, converts them into vector embeddings, stores them in a persistent ChromaDB vector database, retrieves relevant information based on user queries, and generates context-aware answers using an LLM.

The project provides flexibility to choose between different embedding providers and language models. Users can select their preferred models based on their requirements, API availability, and preferences.

## Technologies Used

- Python
- LangChain
- OpenAI Embeddings
- Voyage AI Embeddings
- ChromaDB
- Groq API
- OpenAI ChatGPT API
- RecursiveCharacterTextSplitter
- Python Dotenv

## Features

- Loads multiple text documents from the `docs/` directory.
- Splits documents into smaller chunks using RecursiveCharacterTextSplitter.
- Supports multiple embedding providers:
  - OpenAI Embeddings
  - Voyage AI Embeddings
- Stores document embeddings in a persistent ChromaDB vector database.
- Uses cosine similarity for vector retrieval.
- Retrieves the top 5 relevant document chunks for a user query.
- Supports multiple LLM providers:
  - Groq
  - OpenAI ChatGPT
- Allows users to change embedding models and LLMs according to their preferences.
- Generates context-aware answers using retrieved document information.
- Maintains a local daily request counter as an additional usage safeguard.

## Project Structure

```text
RAG/
│
├── docs/
│   ├── Google.txt
│   ├── Microsoft.txt
│   ├── Nvidia.txt
│   ├── SpaceX.txt
│   └── Tesla.txt
│
├── db/
│   └── chroma_db_retry/
│
├── ingestion_pipeline.py
├── LLM_retrieval.py
├── requirements.txt
├── .env
├── .gitignore
└── groq_usage.json
```

**Note:** The `db/`, `.env`, and `groq_usage.json` files are maintained locally and should not be committed to GitHub.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Mudit8877/RAG-.git
```

### 2. Navigate to the project directory

```bash
cd RAG-
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment (Windows)

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root directory and add the API keys for the providers you want to use.

```env
VOYAGE_API_KEY=your_voyage_api_key
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
```

**Note:** You only need the API keys for the providers you choose to use. Never upload your actual API keys to GitHub.

## Model Configuration

One of the main features of this project is the flexibility to select different embedding models and LLM providers.

### 1. Embedding Model Selection

The project supports:

- **OpenAI Embeddings**
- **Voyage AI Embeddings**

Users can choose their preferred embedding provider in the ingestion and retrieval code.

**Important:** Use the same embedding model and compatible configuration during document ingestion and query retrieval. If you change the embedding model, you must regenerate the vector database using the selected embedding model.

### 2. LLM Selection

The project supports two LLM providers:

- **Groq:** For fast LLM inference using supported Groq models.
- **OpenAI ChatGPT:** For response generation using OpenAI chat models.

Users can select their preferred LLM provider and model in `LLM_retrieval.py`.

Changing the LLM provider does not require recreating the ChromaDB vector database because the LLM is used only after document retrieval.

**Note:** Model names and availability depend on the respective provider's API and account access.

## Running the Project

The project works in two stages.

### Stage 1: Document Ingestion

Run the ingestion pipeline:

```bash
python ingestion_pipeline.py
```

This script:

1. Loads documents from the `docs/` directory.
2. Splits documents into smaller chunks.
3. Generates embeddings using the selected embedding provider.
4. Stores the embeddings in ChromaDB.
5. Saves the vector database locally.

### Stage 2: Retrieval and Answer Generation

After completing document ingestion, run:

```bash
python LLM_retrieval.py
```

This script:

1. Loads the existing ChromaDB vector database.
2. Converts the user query into an embedding using the selected embedding model.
3. Retrieves the top 5 relevant document chunks.
4. Passes the retrieved context and query to the selected LLM.
5. Generates and displays an answer based on the retrieved information.

## Vector Database

The project uses ChromaDB as its persistent vector database.

- **Vector Database:** ChromaDB
- **Collection Name:** `company_documents`
- **Similarity Metric:** Cosine similarity
- **Persistence Directory:** `db/chroma_db_retry/`
- **Embedding Providers:** OpenAI and Voyage AI

The vector database is stored locally and is not included in the GitHub repository.

To recreate the database, run the ingestion pipeline after selecting your preferred embedding model.

## Usage Tracking

The retrieval script maintains a local usage counter in `groq_usage.json`.

This counter acts as an additional safeguard for tracking daily generation attempts when using the Groq configuration. Groq independently enforces its API rate limits, including requests per minute and token limits.

## Future Improvements

- Add support for PDF and other document formats.
- Implement a web-based interface for user queries.
- Improve retrieval accuracy using advanced retrieval techniques.
- Add conversation history and memory.
- Implement source citations in generated answers.
- Improve error handling and API usage monitoring.

## Project Status

**Under Construction** — The basic document ingestion, vector storage, retrieval, and LLM-based answer generation pipeline has been implemented. Additional features and improvements are planned.