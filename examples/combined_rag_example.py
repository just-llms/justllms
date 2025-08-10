from justllms import Client


def create_pinecone_client():
    """Create a JustLLMs client with Pinecone RAG configuration."""
    config = {
        "providers": {
            "google": {
                "api_key": "<gemini api key>",
                "enabled": True
            }
        },
        "cache": {
            "enabled": True,
            "type": "disk"
        },
        "retrieval": {
            "vector_store": {
                "type": "pinecone",
                "api_key": "<pinecone api key>",
                "environment": "<pinecone host url>",
                "index_name": "<pinecone index url>"
            },
            "default_k": 5,
            "similarity_threshold": 0.0,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "splitting_strategy": "recursive",
            "clean_text": True,
            "extract_metadata": True
        }
    }
    
    return Client(config=config)


def create_chromadb_client():
    """Create a JustLLMs client with ChromaDB RAG configuration."""
    
    config = {
        "providers": {
            "google": {
                "api_key": "<gemini api key>",
                "enabled": True
            }
        },
        "cache": {
            "enabled": True,
            "type": "disk"
        },
        "retrieval": {
            "vector_store": {
                "type": "chroma"
            },
            "default_k": 5,
            "similarity_threshold": 0.0,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "splitting_strategy": "recursive",
            "clean_text": True,
            "extract_metadata": True
        }
    }
    
    return Client(config=config)


def run_rag_example(client, collection_name):
    """Run RAG example with the given client and vector store."""
    
    # Create a collection for our documents
    success = client.retrieval.create_collection(collection_name)
    print(f"Collection '{collection_name}' created: {success}")
    
    # PDF files to ingest
    pdf_files = [
        "JustLLMs HLD 1.1.0.pdf",
        "Unified LLM Gateway Design.pdf"
    ]
    
    # Ingest PDF documents
    print("Ingesting PDF documents...")
    result = client.retrieval.ingest_documents(pdf_files, collection_name)
    print(f"Processed {result.documents_processed} documents")
    print(f"Created {result.chunks_created} chunks")
    print(f"Generated {result.vectors_generated} embeddings")
    
    query = "What are the strategic recommendations for JustLLMs and key features of the unified LLM gateway design?"
    
    response = client.completion.retrieve_and_complete(
        query=query,
        collection=collection_name,
        model="gemini-2.5-flash",  
        k=3,  # Number of documents to retrieve
        include_metadata=True,
        temperature=0.7
    )
    
    print(f"Query: {query}")
    print(f"Answer: {response.choices[0].message.content}")
    print(f"Model used: {response.model} (Provider: {response.provider})")

def pinecone_rag_example():
    """Run RAG example with Pinecone."""    
    try:
        client = create_pinecone_client()
        run_rag_example(client, "pinecone_docs")
        print("Pinecone RAG example completed successfully!")
        
    except Exception as e:
        print(f"Pinecone Error: {e}")

def chromadb_rag_example():
    """Run RAG example with ChromaDB."""    
    try:
        client = create_chromadb_client()
        run_rag_example(client, "chromadb_docs")
        print("ChromaDB RAG example completed successfully!")
    except Exception as e:
        print(f"ChromaDB Error: {e}")

def main():
    """Run combined RAG examples."""    
    pinecone_rag_example()
    chromadb_rag_example()


if __name__ == "__main__":
    main()