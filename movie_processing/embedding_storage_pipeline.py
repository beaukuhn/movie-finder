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

    __MAX_TEXT_LENGTH = 512
    __MOVIE_DATASET_NAME = "SandipPalit/Movie_Dataset"
    __EMBEDDING_MODEL = "embed-english-v3.0"
    __COLLECTION_NAME = "movies"

    def __init__(self, cohere_key: str):
        self.__cohere_client = cohere.Client(cohere_key)
        self.__chroma_db = chromadb.Client()
        self.__collection = self.__chroma_db.get_or_create_collection(
            self.__COLLECTION_NAME
        )

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
    def get_embeddings_from_cohere(self, record_str: str) -> List[float]:
        """
        Get embeddings for a record using the Cohere API.

        Args:
            record_str:
                    A stringified dictionary containing movie details. Expected keys in the dictionary are:
                   'Release Date' (str): The release date of the movie in 'YYYY-MM-DD' format.
                   'Title' (str): The title of the movie.
                   'Overview' (str): A brief description of the movie.
                   'Genre' (str): The genre(s) of the movie, formatted as a string of list.
                   'Vote Average' (float): The average vote or rating of the movie.
                   'Vote Count' (int): The count of votes or ratings received by the movie.
        """
        if len(record_str) > self.__MAX_TEXT_LENGTH:
            raise ValueError(
                f"Text length exceeds the maximum limit of {self.__MAX_TEXT_LENGTH} characters."
            )

        return self.__cohere_client.embed(
            texts=[record_str],
            model=self.__EMBEDDING_MODEL,
            input_type="search_document",
        )["embeddings"][0]

    def run(self):
        """
        Run the pipeline to store movie embeddings in ChromaDB.
        """
        # Load the movie dataset from Hugging Face
        movie_dataset = self.load_movie_dataset_from_hugging_face(
            self.__MOVIE_DATASET_NAME,
            streaming=True,
        )

        for record in movie_dataset["train"]:
            # 1) Convert the record to a string
            record_str = json.dumps(record)

            # 2) Get embeddings for the record
            embedding = self.get_embeddings_from_cohere(record_str)

            # 3) Store the embeddings in ChromaDB
            self.__collection.add(
                embeddings=[embedding],
                metadatas=[record],
                documents=[record_str],
                ids=[uuid.uuid4()],
            )
