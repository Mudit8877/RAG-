# RAG Project using LangChain, ChromaDB, and Multiple LLM Providers (Under Construction)

## Overview

This project implements a History-Aware Retrieval-Augmented Generation (RAG) pipeline using LangChain, vector embeddings, ChromaDB, and Large Language Models (LLMs).

It processes documents, converts their content into vector embeddings, and stores them in a persistent ChromaDB vector database. Relevant information is retrieved based on user queries and used by an LLM to generate context-aware answers.

The project also integrates **Multi-Query Retrieval and Reciprocal Rank Fusion (RRF)** to improve retrieval coverage and ranking. Conversational history enables users to ask follow-up questions while maintaining context.

Users can configure their preferred embedding and LLM providers based on their requirements and API availability.

## Technologies Used

- Python
- LangChain
- ChromaDB
- OpenAI Embeddings
- Voyage AI Embeddings
- Groq API
- OpenAI ChatGPT API
- Pydantic
- Multi-Query Retrieval
- Reciprocal Rank Fusion (RRF)
- RecursiveCharacterTextSplitter
- Python Dotenv

## Features

- Loads multiple text documents from the `docs/` directory.
- Splits documents into chunks using RecursiveCharacterTextSplitter.
- Supports OpenAI and Voyage AI embedding providers.
- Stores embeddings in a persistent ChromaDB vector database.
- Uses cosine similarity for document retrieval.
- Supports Groq and OpenAI LLM providers.
- Allows users to configure their preferred embedding models and LLMs.
- Maintains conversational history using LangChain message types.
- Uses the last three previous user questions for history-aware retrieval.
- Generates three alternative queries using an LLM.
- Embeds the original query and three alternatives in a single Voyage AI batch request.
- Retrieves three document chunks per query.
- Applies Reciprocal Rank Fusion (RRF) to combine retrieval rankings.
- Selects the top three documents after RRF for final answer generation.
- Uses retrieved documents and conversation history to generate answers.
- Maintains a local daily Groq request counter as an additional usage safeguard.

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
├── Final_retrieval.py
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

### 3. Create and activate a virtual environment (Windows)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root directory:

```env
VOYAGE_API_KEY=your_voyage_api_key
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
```

**Note:** Only the API keys for the providers you choose are required. Never upload actual API keys to GitHub.

## Model Configuration

### Embedding Models

The project supports:

- **OpenAI Embeddings**
- **Voyage AI Embeddings**

Use the same embedding model and compatible configuration during ingestion and retrieval. Changing the embedding model requires regenerating the document embeddings.

### LLM Providers

The project supports:

- **Groq:** Fast LLM inference using supported Groq models.
- **OpenAI ChatGPT:** LLM-based response generation.

Users can configure their preferred provider and model. Changing the LLM does not require recreating the vector database.

## Running the Project

### Stage 1: Document Ingestion

```bash
python ingestion_pipeline.py
```

This script:

1. Loads documents from `docs/`.
2. Splits documents into smaller chunks.
3. Generates embeddings using the selected embedding provider.
4. Stores the embeddings in ChromaDB.

### Stage 2: Conversational RAG Pipeline

```bash
python Final_retrieval.py
```

The integrated retrieval pipeline follows these steps:

1. Accepts a user question and checks the conversation history.
2. Combines the current question with the last three previous user questions.
3. Uses an LLM to generate three alternative search queries.
4. Combines the original query with the three alternatives.
5. Generates all four query embeddings in a single Voyage AI batch request.
6. Performs four independent similarity searches in ChromaDB, retrieving three documents per query.
7. Applies Reciprocal Rank Fusion (RRF) to combine the retrieval rankings.
8. Selects the top three documents based on their RRF scores.
9. Sends the retrieved context and conversation history to the LLM.
10. Generates the final answer and updates the chat history.

## Multi-Query Retrieval and Reciprocal Rank Fusion (RRF)

Multi-Query Retrieval generates alternative formulations of the original question to improve retrieval coverage.

For example:

**Original Query:**

```text
How does Tesla make money?
```

**Alternative Queries:**

```text
1. What are Tesla's primary sources of revenue?
2. How does Tesla generate income from its automotive and energy businesses?
3. What products and services contribute to Tesla's revenue?
```

The original query and three alternatives are embedded together in one batch request. ChromaDB independently retrieves the top three document chunks for each query.

RRF combines these rankings using the following formula:

\[
RRF(d)=\sum_{r \in R}\frac{1}{k+rank_r(d)}
\]

Where:

- \(d\) = Retrieved document chunk.
- \(R\) = Set of retrieval result lists.
- \(rank_r(d)\) = Position of the document in a result list.
- \(k\) = Ranking constant, set to 60.

Documents appearing in multiple result lists can receive higher combined scores. The final three documents are selected for answer generation.

## History-Aware Retrieval

The project maintains conversational history throughout a chat session.

For example:

```text
User: How much did Microsoft pay to acquire GitHub?

AI: Microsoft acquired GitHub for $7.5 billion.

User: When did it happen?
```

The retrieval process incorporates the last three previous user questions into the current search query to help resolve follow-up questions.

The LLM also receives the conversation history and retrieved documents when generating the final answer.

## Vector Database

| Configuration | Value |
|---|---|
| Vector Database | ChromaDB |
| Collection | `company_documents` |
| Similarity Metric | Cosine |
| Persistence Directory | `db/chroma_db_retry/` |
| Embedding Providers | OpenAI, Voyage AI |

The vector database is stored locally and is not included in the GitHub repository.

## Usage Tracking

The project maintains a local `groq_usage.json` file to track daily Groq API request attempts.

This acts as an additional usage safeguard and does not replace the rate limits independently enforced by Groq.

## Future Improvements

- Add PDF and DOCX document support.
- Add image processing using vision models.
- Implement a web-based interface.
- Add persistent chat history across sessions.
- Implement source citations in generated answers.
- Improve retrieval accuracy and error handling.

## Project Status

**Under Construction** — The project currently supports document ingestion, persistent vector storage, history-aware conversational retrieval, Multi-Query Retrieval, Reciprocal Rank Fusion, and context-based answer generation. Additional document formats and image processing are planned.
