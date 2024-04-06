from movie_processing import MovieEmbeddingStoragePipeline, MovieSimilarityFinder


# Create a main function that will run the program
def main():

    # Create an instance of the MovieEmbeddingStoragePipeline class
    movie_embedding_storage = MovieEmbeddingStoragePipeline()

    # Run the pipeline to store movie embeddings in ChromaDB
    movie_embedding_storage.run()

    # Create an instance of the MovieSimilarityFinder class
    similarity_finder = MovieSimilarityFinder(cohere_key="your_cohere_key_here")

    # Search for similar movies based on a query from the terminal and loop until the user exits
    while True:
        query = input("Enter a search query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        similar_movies = similarity_finder.search(query)
        print("Similar movies:")
        for movie in similar_movies:
            print(movie)
        print("\n")


# Run the main function if this script is executed
if __name__ == "__main__":
    main()
