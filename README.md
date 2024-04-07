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
```python
python main.py
...
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


I joined my previous startup as the tenth employee, playing a pivotal role in its growth to a 50-strong team. My tenure there was marked by versatility, primarily focusing on backend and infrastructure, while also occasionally contributing to frontend development. As a member of the Platform and Infrastructure team, I championed the integration of Large Language Models (LLMs) into our financial data warehouse architecture. This included implementing Retrieval-Augmented Generation (RAG), significantly enhancing our users' ability to interactively query and understand their financial data, such as inquiring about annual expenses or specific quarterly revenues.

My dedication to innovation was further demonstrated when I significantly enhanced our transaction categorization accuracy by 100%, employing vector embeddings and a vector database, for which I won a company-wide hackathon.

In addition to practical experience, I have also completed a graduate course in Machine Learning. This theoretical background complements my hands-on skills.

Moreover, I am well-versed in thriving within a remote work culture, having demonstrated the ability to work effectively and diligently in such an environment. This experience has equipped me with the skills necessary to excel in a remote-friendly setting like Cohere's, particularly in teams spread across various time zones."