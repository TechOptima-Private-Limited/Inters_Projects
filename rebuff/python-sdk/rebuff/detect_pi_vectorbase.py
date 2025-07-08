from typing import Dict
from langchain.vectorstores.pinecone import Pinecone as LC_Pinecone
from langchain_openai import OpenAIEmbeddings

from openai import OpenAI
from pinecone import Pinecone as PC, ServerlessSpec


def detect_pi_using_vector_database(
    input: str, similarity_threshold: float, vector_store: LC_Pinecone
) -> Dict:
    """
    Detects Prompt Injection using similarity search with vector database.

    Args:
        input (str): user input to be checked for prompt injection
        similarity_threshold (float): The threshold for similarity between entries in vector database and the user input.
        vector_store (Pinecone): Vector database of prompt injections

    Returns:
        Dict: top_score (float) and count_over_max_vector_score (int)
    """

    top_k = 20
    results = vector_store.similarity_search_with_score(input, top_k)

    top_score = 0
    count_over_max_vector_score = 0

    for _, score in results:
        if score is None:
            continue

        if score > top_score:
            top_score = score

        if score >= similarity_threshold and score > top_score:
            count_over_max_vector_score += 1

    return {
        "top_score": top_score,
        "count_over_max_vector_score": count_over_max_vector_score,
    }


def init_pinecone(api_key: str, index: str, openai_api_key: str) -> LC_Pinecone:
    """
    Initializes connection with the Pinecone vector database using existing index.

    Args:
        api_key (str): Pinecone API key
        index (str): Pinecone index name
        openai_api_key (str): Open AI API key

    Returns:
        vector_store (langchain Pinecone wrapper)
    """

    if not api_key:
        raise ValueError("Pinecone API key is missing.")

    pc = PC(api_key=api_key)

    if index not in pc.list_indexes().names():
        pc.create_index(
            name=index,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    pc_index = pc.Index(index)

    # Initialize OpenAI embeddings with client
    client = OpenAI(api_key=openai_api_key)

    embeddings = OpenAIEmbeddings(
        client=client,
        model="text-embedding-ada-002"
    )

    # LangChain Pinecone vector store initialization
    vector_store = LC_Pinecone(
        index=pc_index,
        embedding=embeddings,
        text_key="input"
    )

    return vector_store

