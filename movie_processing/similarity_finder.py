import chromadb
import cohere


class MovieSimilarityFinder:
    """
    A class to find similar movies based on a search query.
    """

    __MAX_TEXT_LENGTH = 512
    __EMBEDDING_MODEL = "embed-english-v3.0"
    __COLLECTION_NAME = "movies"

    def __init__(self, cohere_key: str):
        self.__chroma_db = chromadb.Client()
        self.__cohere_client = cohere.Client(cohere_key)
        self.__collection = self.__chroma_db.get_or_create_collection(
            self.__COLLECTION_NAME
        )

    def generate_query_embedding(self, query_str: str):
        """
        Generate an embedding for a search query.

        Args:
            query_str: The search query.

        Returns:
            A list of floats representing the embedding.
        """
        if len(query_str) > self.__MAX_TEXT_LENGTH:
            raise ValueError(
                f"Text length exceeds the maximum limit of {self.__MAX_TEXT_LENGTH} characters."
            )

        # Use the Cohere API to generate an embedding for the search query
        query_embedding = self.__cohere_client.embed(
            texts=[query_str],
            model=self.__EMBEDDING_MODEL,
            input_type="search_query",
        )
        return query_embedding

    def search(self, query_str: str, top_n=5):
        """
        Search for similar movies based on a query.

        Args:
            query_str: The search query.
            top_n: The number of similar movies to return.

        Returns:
            A list of similar movies.
        """
        # Generate an embedding for the search query
        query_embedding = self.generate_query_embedding(query_str)

        # Conduct a cosine similarity search in the collection
        similar_movies = self.__collection.query(
            query_embeddings=query_embedding.embeddings,
            n_results=top_n,
        )
        return [similar_movies["metadatas"][0][i]["Title"] for i in range(top_n)]
