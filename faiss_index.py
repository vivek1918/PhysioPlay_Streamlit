import faiss
import numpy as np

def create_faiss_index(dimension=768):
    """
    Create a FAISS index for L2 distance on CPU.

    Args:
        dimension (int): The dimension of the vectors to be indexed.

    Returns:
        faiss.Index: The created FAISS index.
    """
    # Create a flat (L2) index on the CPU
    index = faiss.IndexFlatL2(dimension)
    return index

def add_to_index(cpu_index, embeddings):
    """
    Add embeddings to the FAISS index.

    Args:
        cpu_index (faiss.Index): The FAISS index to add embeddings to.
        embeddings (list or np.ndarray): The embeddings to add, shape (N, dimensions).

    Returns:
        None
    """
    # Ensure embeddings are in the correct format
    cpu_index.add(np.array(embeddings, dtype=np.float32))  # Convert to float32 if not already

def search_index(cpu_index, query, top_k=5):
    """
    Search the FAISS index for the nearest neighbors of the query vector.

    Args:
        cpu_index (faiss.Index): The FAISS index to search.
        query (np.ndarray): The query vector, shape (1, dimensions).
        top_k (int): The number of nearest neighbors to retrieve.

    Returns:
        List[Tuple[int, float]]: A list of tuples containing the indices and distances of the nearest neighbors.
    """
    distances, indices = cpu_index.search(np.array(query, dtype=np.float32).reshape(1, -1), top_k)
    return list(zip(indices[0], distances[0]))  # Return a list of (index, distance) tuples
