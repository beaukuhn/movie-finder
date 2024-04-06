# Similar Movie Finder

## Overview
This project provides a comprehensive suite of tools for processing movie data, generating embeddings, and finding similar movies based on a given query. It consists of two main components:

- `EmbeddingStoragePipeline`: A class responsible for processing movies, generating embeddings using the Cohere API, and storing them in ChromaDB.
- `MovieSimilarityFinder`: A class that handles queries to find the most similar movies within the stored embeddings in ChromaDB, given a textual movie query.

## Features
- Load and process movie datasets from Hugging Face.
- Generate and store movie embeddings in ChromaDB.
- Query ChromaDB to find the top `n` similar movies to a given input.

## Structure
```
movie_finder/
│
├── movie_processing/
│   ├── __init__.py
│   ├── embedding_storage.py   # Contains EmbeddingStoragePipeline class
│   └── similarity_finder.py   # Contains MovieSimilarityFinder class
│
├── utils.py                   # Helper functions and utilities
├── main.py                    # Main script to run the application
└── requirements.txt           # Dependencies for the project
```

## Getting Started

### Prerequisites
- Python 3.8+
- Cohere API key
- Access to ChromaDB

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/<your-username>/<repository-name>.git
   ```
2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
### Usage
- Initialize and run the embedding storage pipeline:
```python
from movie_processing.embedding_storage import EmbeddingStoragePipeline
storage_pipeline = EmbeddingStoragePipeline('<cohere-api-key>')
storage_pipeline.run()
```
- Query for similar movies:
```
Enter a search query (or type 'exit' to quit): A movie about a group of allies embarking on an epic adventure through space
Similar movies:
The Terrornauts
Plan 9 from Outer Space
Gamera vs. Guiron
Voyage to the Planet of Prehistoric Women
Conquest of Space
```

## Authors
- Beau Kuhn

## Acknowledgements
- Cohere for the embedding API.
- ChromaDB for the vector database.
- Hugging Face for providing the datasets.

## Contact
- beaukuhn@proton.me

## License
MIT License

Copyright (c) [2024] [Beau Alexander Kuhn]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
