
# RAG Project using LangChain and ChromaDB

## Overview
This project implements a Retrieval-Augmented Generation (RAG)
pipeline using LangChain, Voyage AI embeddings, and ChromaDB.

## Technologies Used
- Python
- LangChain
- Voyage AI Embeddings
- ChromaDB
- RecursiveCharacterTextSplitter
- Python Dotenv

## Features
- Loads multiple text documents.
- Splits documents into smaller chunks.
- Generates embeddings using Voyage AI.
- Stores embeddings in ChromaDB.
- Uses cosine similarity for vector storage.

## Project Structure

    RAG/
    ├── docs/
    ├── db/
    ├── main.py
    ├── requirements.txt
    ├── .env
    └── .gitignore

## Installation

Clone the repository:

    git clone YOUR_REPOSITORY_URL

Move into the project directory:

    cd RAG

Create a virtual environment:

    python -m venv venv

Activate it on Windows:

    .\venv\Scripts\Activate.ps1

Install dependencies:

    pip install -r requirements.txt

## Environment Variables

Create a `.env` file and add your Voyage AI API key:

    VOYAGE_API_KEY=your_api_key_here

## Run the Project

    python main.py

## Vector Database

The ChromaDB database is generated locally and is not included
in this repository. Run the ingestion script to recreate it.
