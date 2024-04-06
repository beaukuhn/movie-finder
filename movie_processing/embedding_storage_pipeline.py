import os
import uuid
import cohere
import chromadb
import datasets
from typing import List
from utils import retry_with_exponential_backoff


class MovieEmbeddingStoragePipeline:
    """
    A pipeline to store movie embeddings in ChromaDB.
    """

    __MOVIE_DATASET_NAME = "SandipPalit/Movie_Dataset"
    __EMBEDDING_MODEL = "embed-english-v3.0"
    __COLLECTION_NAME = "movies"
    __COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
    __BATCH_SIZE = 96
    __LIMIT = 1000

    def __init__(self, cohere_key: str = __COHERE_API_KEY):
        self.__cohere_client = cohere.Client(cohere_key)
        self.__chroma_db = chromadb.Client()
        self.__collection = self.__chroma_db.get_or_create_collection(
            self.__COLLECTION_NAME
        )

    def __get_subset(self, dataset, limit=__LIMIT):
        """
        Get a subset of the dataset of size limit.

        Args:
            dataset: The input dataset.
            limit: The number of items to include in the subset.

        Returns:
            An iterator that yields up to limit items from the dataset.
        """
        count = 0
        for item in dataset:
            if count >= limit:
                break
            yield item
            count += 1

    def __create_batches(self, dataset, batch_size=__BATCH_SIZE):
        """
        Create batches of a given size from a dataset.

        Args:
            dataset: The input dataset, which is an iterable.
            batch_size: The size of each batch.

        Returns:
            An iterator that yields batches.
        """
        batch = []
        for item in dataset:
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch  # Yield any remaining items as the last batch

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

    @retry_with_exponential_backoff(max_retries=10, backoff_factor=0.1)
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
        texts = [record["Overview"] for record in batch]
        print("Sending batch...")
        response = self.__cohere_client.embed(
            texts=texts,
            model=self.__EMBEDDING_MODEL,
            input_type="search_document",
            batching=True,
        )
        print("LOLOLO", response)
        return response

    @retry_with_exponential_backoff(max_retries=10, backoff_factor=0.5)
    def store_embeddings_in_chromadb(self, batch, embeddings_batch):
        """
        Store embeddings in ChromaDB.

        Args:
            embeddings: A dictionary containing the texts and embeddings.
        """
        print("Storing embeddings into ChromaDB...")
        texts, embeddings = embeddings_batch.texts, embeddings_batch.embeddings
        for idx, (text, embedding) in enumerate(zip(texts, embeddings)):
            record = batch[idx]
            print(record, "the record")
            self.__collection.add(
                embeddings=[embedding],
                metadatas=record,
                documents=[record],
                ids=[str(uuid.uuid4())],
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

        batch_number = 0

        for batch in self.__create_batches(self.__get_subset(movie_dataset["train"])):
            print(
                f"Processing batch {batch_number}, size: {len(batch)}, total: {batch_number * self.__BATCH_SIZE}"
            )
            # 1) Get embeddings for the record
            embeddings_batch = self.get_embeddings_from_cohere(batch)

            # 2) Store the embeddings in ChromaDB
            self.store_embeddings_in_chromadb(batch, embeddings_batch)

            batch_number += 1
