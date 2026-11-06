# RAG Project using LangChain, ChromaDB, and Multiple LLM Providers (Under Construction)

## Overview

This project implements a History-Aware Retrieval-Augmented Generation (RAG) pipeline using LangChain, vector embeddings, ChromaDB, and Large Language Models (LLMs).

The application processes multiple text documents, converts them into vector embeddings, and stores them in a persistent ChromaDB vector database. It retrieves relevant information based on user queries and generates context-aware answers using an LLM.

The project also supports conversational chat history, allowing users to ask follow-up questions while maintaining the context of previous interactions.

To improve retrieval accuracy, the project also implements Multi-Query Retrieval and Reciprocal Rank Fusion (RRF), which generate alternative search queries and combine their retrieval results into a unified ranking.

It provides flexibility to choose between different embedding providers and language models. Users can select their preferred models based on their requirements, API availability, and preferences.

## Technologies Used

- Python
- LangChain
- OpenAI Embeddings
- Voyage AI Embeddings
- ChromaDB
- Groq API
- OpenAI ChatGPT API
- Pydantic
- Reciprocal Rank Fusion (RRF)
- Multi-Query Retrieval
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
- Retrieves the top relevant document chunks for a user query (configurable).
- Supports multiple LLM providers:
  - Groq
  - OpenAI ChatGPT
- Allows users to change embedding models and LLMs according to their preferences.
- Maintains conversational chat history using LangChain message types.
- Supports history-aware retrieval by incorporating previous user questions into the current search query.
- Handles follow-up questions using previous conversational context.
- Combines chat history, retrieved documents, and the current question into a single prompt.
- Uses only one LLM API call per user question for answer generation.
- Maintains a local daily request counter as an additional usage safeguard when using Groq.
- Implements Multi-Query Retrieval by generating alternative variations of the original user query using an LLM.
- Performs independent similarity searches for each generated query variation.
- Implements Reciprocal Rank Fusion (RRF) to combine retrieval results from multiple query variations.
- Assigns higher combined scores to documents that appear in multiple retrieval result lists.
- Ranks retrieved chunks based on their combined RRF scores to improve retrieval relevance.

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
├── reciprocal_rank_fusion.py
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

Users can choose their preferred embedding provider and model in the ingestion and retrieval code.

**Important:** Use the same embedding model and compatible configuration during document ingestion and query retrieval. If you change the embedding model, you must regenerate the vector database using the selected embedding model.

### 2. LLM Selection

The project supports two LLM providers:

- **Groq:** For fast LLM inference using supported Groq models.
- **OpenAI ChatGPT:** For response generation using OpenAI chat models.

Users can select their preferred LLM provider and model in `LLM_retrieval.py` and the Multi-Query Retrieval implementation.

Changing the LLM provider does not require recreating the ChromaDB vector database because the LLM is used only after document retrieval or for generating query variations.

**Note:** Model names and availability depend on the respective provider's API and account access.

## Running the Project

The project currently has three main execution stages.

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

### Stage 2: History-Aware Retrieval and Answer Generation

After completing ingestion, run:

```bash
python LLM_retrieval.py
```

This script:

1. Loads the existing ChromaDB vector database.
2. Accepts a question from the user.
3. Checks whether previous conversation history is available.
4. Incorporates previous user questions into the current search query to provide conversational context.
5. Converts the search query into an embedding using the selected embedding model.
6. Retrieves the most relevant document chunks from ChromaDB.
7. Combines the conversation history, retrieved documents, and current question into a single prompt.
8. Sends the prompt to the selected LLM provider.
9. Generates a context-aware answer using a single LLM API call per question.
10. Stores the user's question and the AI-generated answer in chat history for subsequent interactions.

### Stage 3: Multi-Query Retrieval and Reciprocal Rank Fusion

Run:

```bash
python reciprocal_rank_fusion.py
```

This script:

1. Takes the original user query as input.
2. Uses the selected LLM to generate three alternative query variations.
3. Uses Pydantic structured output to obtain query variations in JSON format.
4. Performs independent similarity searches in ChromaDB for each generated query.
5. Retrieves the top relevant document chunks for every query variation.
6. Applies Reciprocal Rank Fusion (RRF) to combine the retrieval results.
7. Calculates a combined ranking score for each unique document chunk.
8. Ranks the retrieved chunks based on their RRF scores.
9. Displays the final ranked retrieval results.

## Multi-Query Retrieval and Reciprocal Rank Fusion (RRF)

The project implements Multi-Query Retrieval to improve the retrieval of relevant information from the vector database.

A single user query may not always retrieve all relevant documents because different wording can produce different semantic search results.

To address this, the application generates multiple variations of the original query using an LLM.

For example:

**Original Query:**

```text
How does Tesla make money?
```

**Generated Query Variations:**

```text
1. What are Tesla's primary sources of revenue?

2. How does Tesla generate income from its automotive and energy businesses?

3. What products and services contribute to Tesla's overall revenue?
```

Each query variation is independently searched against the ChromaDB vector database.

### Reciprocal Rank Fusion (RRF)

After retrieving results for each query variation, the application applies Reciprocal Rank Fusion to combine the individual rankings into a unified result list.

RRF calculates a score for each document based on its position in the different retrieval result lists.

The formula used is:

\[
RRF(d)=\sum_{r \in R}\frac{1}{k+rank_r(d)}
\]

Where:

- \(d\) = Retrieved document chunk.
- \(R\) = Set of retrieval result lists.
- \(rank_r(d)\) = Position of the document in a particular result list.
- \(k\) = Ranking constant, set to 60 in this implementation.

Documents appearing in multiple retrieval lists receive higher combined scores, while documents appearing near the top of individual lists also receive greater weight.

The final results are sorted in descending order of their RRF scores.

**Key Benefits:**

- Reduces dependence on a single search query.
- Captures relevant documents through different query formulations.
- Combines multiple retrieval rankings into one result list.
- Gives higher priority to consistently retrieved documents.
- Improves retrieval coverage and ranking quality.

**Note:** RRF improves the ranking of retrieved documents but does not guarantee that every relevant document will be retrieved. Its effectiveness depends on the generated queries and the initial retrieval results.

## History-Aware Retrieval

The project supports conversational question answering by maintaining chat history throughout a session.

For example:

**First question:**

```text
User: How much did Microsoft pay to acquire GitHub?
AI: Microsoft acquired GitHub for $7.5 billion.
```

**Follow-up question:**

```text
User: When did it happen?
```

The retrieval process incorporates previous user questions to provide additional context when searching ChromaDB.

The LLM also receives the previous conversation, retrieved documents, and current question to generate an appropriate response.

This allows the application to handle follow-up questions without making a separate LLM API call for query rewriting.

**Note:** The current implementation uses simple question concatenation for contextual retrieval rather than a separate AI-based query rewriting step. Complex follow-up questions may require more advanced retrieval techniques.

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

This counter acts as an additional safeguard for tracking daily generation attempts when using the Groq configuration.

Groq independently enforces its API rate limits, including requests per minute, requests per day, and token limits.

The local counter does not replace Groq's official API rate limits.

## Future Improvements

- Add support for PDF and other document formats.
- Add support for image-based document descriptions using vision models.
- Implement a web-based interface for user queries.
- Further improve retrieval accuracy using advanced retrieval techniques.
- Integrate Multi-Query Retrieval and RRF into the main conversational RAG pipeline.
- Implement persistent chat history across different sessions.
- Add conversation summarization for long chat sessions.
- Implement source citations in generated answers.
- Improve error handling and API usage monitoring.

## Project Status

**Under Construction** — The project currently implements document ingestion, vector storage, history-aware retrieval, conversational question answering, Multi-Query Retrieval, and Reciprocal Rank Fusion. Additional document formats, image processing, and further retrieval improvements are planned.