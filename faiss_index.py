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

# Example usage
if __name__ == "__main__":
    # Create a FAISS index for 768-dimensional vectors
    index = create_faiss_index(dimension=768)

    # Generate random embeddings (for testing purposes)
    embeddings = np.random.rand(1000, 768).astype('float32')  # 1000 random 768-dim vectors
    add_to_index(index, embeddings)

    # Output the number of embeddings in the index
    print("Number of embeddings in index:", index.ntotal)
