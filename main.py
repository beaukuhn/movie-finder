import os
from movie_processing import MovieEmbeddingStoragePipeline, MovieSimilarityFinder


def main():
    # Create an instance of the MovieEmbeddingStoragePipeline class
    print("Establishing pipeline...")
    movie_embedding_storage = MovieEmbeddingStoragePipeline()

    # Run the pipeline to store movie embeddings in ChromaDB
    print("Running the pipeline...")
    movie_embedding_storage.run()

    # Create an instance of the MovieSimilarityFinder class
    print("Launching movie similarity finder...")
    similarity_finder = MovieSimilarityFinder(os.environ.get("COHERE_API_KEY"))

    # Search for similar movies based on a query from the terminal and loop until the user exits
    while True:
        query = input("Enter a search query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        similar_movies = similarity_finder.search(query)
        print("Similar movies:\n")
        for movie in similar_movies:
            print(movie)
        print("\n")


# Run the main function if this script is executed
if __name__ == "__main__":
    main()
