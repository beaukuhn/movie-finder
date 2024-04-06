import json
import uuid
import cohere
import chromadb
import datasets
from typing import List
from utils import retry_with_exponential_backoff


class MovieEmbeddingStoragePipeline:
    """
    EmbeddingStoragePipeline class to process JSONL files and store embeddings in ChromaDB.
    """

    __MOVIE_DATASET_NAME = "SandipPalit/Movie_Dataset"
    __EMBEDDING_MODEL = "embed-english-v3.0"
    __COLLECTION_NAME = "movies"
    __BATCH_SIZE = 96

    def __init__(self, cohere_key: str):
        self.__cohere_client = cohere.Client(cohere_key)
        self.__chroma_db = chromadb.Client()
        self.__collection = self.__chroma_db.get_or_create_collection(
            self.__COLLECTION_NAME
        )

    def __create_batches(self, array, batch_size):
        """
        Create batches of a given size from an array.

        Args:
            array: The input array.
            batch_size: The size of each batch.

        Returns:
            A list of batches.
        """
        for i in range(0, len(array), batch_size):
            yield array[i : i + batch_size]

    def load_movie_dataset_from_hugging_face(
        self,
        dataset_name: str,
        streaming: bool = False,
    ):
        """
        Loads a movie dataset from the Hugging Face.

        Args:
            dataset_name: Name of the dataset
            streaming: Whether to stream the dataset for efficient processing

        Returns:
            Movie dataset
        """
        return datasets.load_dataset(dataset_name, streaming=streaming)

    @retry_with_exponential_backoff(max_retries=10, backoff_factor=0.5)
    def get_embeddings_from_cohere(self, batch) -> List[float]:
        """
        Get embeddings for a batch using the Cohere API.

        Args:
            batch:  A list of records to get embeddings for.
                    A record is a dictionary containing movie details.
                    Expected keys in the dictionary are:
                    'Release Date' (str): The release date of the movie in 'YYYY-MM-DD' format.
                    'Title' (str): The title of the movie.
                    'Overview' (str): A brief description of the movie.
                    'Genre' (str): The genre(s) of the movie, formatted as a string of list.
                    'Vote Average' (float): The average vote or rating of the movie.
                    'Vote Count' (int): The count of votes or ratings received by the movie.
        """
        # Stringify each individual record in the batch
        texts = [json.dumps(record) for record in batch]

        return self.__cohere_client.embed(
            texts=[texts],
            model=self.__EMBEDDING_MODEL,
            input_type="search_document",
        )

    @retry_with_exponential_backoff(max_retries=10, backoff_factor=0.5)
    def store_embeddings_in_chromadb(self, embeddings):
        """
        Store embeddings in ChromaDB.

        Args:
            embeddings: A dictionary containing the texts and embeddings.
        """
        texts, embeddings = embeddings["texts"], embeddings["embeddings"]
        for text, embedding in zip(texts, embeddings):
            self.__collection.add(
                embeddings=[embedding],
                metadatas=[text],
                documents=[text],
                ids=[uuid.uuid4()],
            )

    def run(self):
        """
        Run the pipeline to store movie embeddings in ChromaDB.
        """
        # Load the movie dataset from Hugging Face
        movie_dataset = self.load_movie_dataset_from_hugging_face(
            self.__MOVIE_DATASET_NAME,
            streaming=True,
        )

        for batch in self.__create_batches(
            movie_dataset["train"][:: self.__BATCH_SIZE]
        ):
            # 1) Get embeddings for the record
            embeddings_dict = self.get_embeddings_from_cohere(batch)

            # 2) Store the embeddings in ChromaDB
            self.store_embeddings_in_chromadb(embeddings_dict)
