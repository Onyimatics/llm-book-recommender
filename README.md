# Semantic Book Recommender

<p align="center">
  <a href="https://huggingface.co/spaces/Onyimatics/Semantic_Book_Recommender">
    <img alt="HuggingFace Spaces" src="https://img.shields.io/badge/Gradio%20App%20-Running-green?logo=gradio&style=flat-square">
  </a>
  <a href="https://huggingface.co/spaces/Onyimatics/Semantic_Book_Recommender">
    <img alt="HuggingFace Spaces" src="https://img.shields.io/badge/Deployed%20on-Hugging%20Face-blue?logo=huggingface&style=flat-square">
  </a>
  <a href="https://github.com/Onyimatics/llm-book-recommender/actions">
    <img alt="GitHub Actions Build" src="https://github.com/Onyimatics/llm-book-recommender/actions/workflows/main.yml/badge.svg">
  </a>
  <a href="https://github.com/Onyimatics/llm-book-recommender/stargazers">
    <img alt="GitHub Stars" src="https://img.shields.io/github/stars/Onyimatics/llm-book-recommender?style=social">
  </a>
  <a href="https://github.com/Onyimatics/llm-book-recommender/fork">
    <img alt="GitHub Forks" src="https://img.shields.io/github/forks/Onyimatics/llm-book-recommender?style=social">
  </a>
  <a href="https://github.com/Onyimatics/llm-book-recommender/issues">
    <img alt="GitHub Issues" src="https://img.shields.io/github/issues/Onyimatics/llm-book-recommender?style=flat-square">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img alt="MIT License" src="https://img.shields.io/github/license/Onyimatics/llm-book-recommender?color=blue&style=flat-square">
  </a>
</p>


A smart book recommendation system that leverages Large Language Models (LLMs) to provide personalized book suggestions based on natural language descriptions, emotional tones, and categories.

## 🔗 Live Demo

Experience the Semantic Book Recommender in action:  
[Semantic Book Recommender on HuggingFace Spaces](https://huggingface.co/spaces/Onyimatics/Semantic_Book_Recommender)

## ✨ Features

- **Natural Language Search**: Describe the type of book you're looking for in plain language
- **Semantic Understanding**: Uses OpenAI embeddings to deeply understand book content and match your interests
- **Emotional Tone Filtering**: Find books with specific emotional qualities (Happy, Sad, Suspenseful, etc.)
- **Category Filtering**: Narrow recommendations by book genre/category
- **User-Friendly Interface**: Intuitive Gradio UI with visual book gallery and descriptions
- **Performance Optimization**: Implements smart caching to reduce API calls and speed up recommendations

## 📚 Dataset

The project uses the "7k Books with Metadata" dataset from Kaggle, enhanced with:
- Emotional analysis (joy, anger, sadness, fear, surprise)
- Simplified category classification
- Processed text descriptions optimized for semantic search

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Gradio**: Interactive UI with responsive gallery layout
- **LangChain**: Document processing and embedding integration
- **OpenAI**: API for generating high-quality semantic embeddings
- **ChromaDB**: Vector database for similarity search
- **Pandas & NumPy**: Data manipulation and analysis

## 🗂️ Project Structure

- **Core Application**:
  - `gradio-dashboard.py`: Main application with UI and recommendation engine

- **Data Processing & Analysis**:
  - `data-exploration.ipynb`: Initial dataset exploration and cleaning
  - `sentiment-analysis.ipynb`: Emotional tone extraction from book descriptions
  - `text-classification.ipynb`: Book category classification
  - `vector-search.ipynb`: Testing and development of semantic search functionality

- **Data Files**:
  - `books_cleaned.csv`: Processed dataset with complete metadata
  - `books_with_emotions.csv`: Dataset enhanced with emotional analysis
  - `books_with_categories.csv`: Dataset with categorization
  - `tagged_description.txt`: Text file with processed descriptions for embedding

- **System Files**:
  - `requirements.txt`: Project dependencies
  - `cache/`: Directory for stored embeddings to optimize performance
  - `chroma_db_books/`: ChromaDB vector database files

## 🚀 Getting Started

### Prerequisites

- Python 3.9+ 
- OpenAI API key (for embeddings)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/llm-book-recommender.git
   cd llm-book-recommender
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. Run the application:
   ```bash
   python gradio-dashboard.py
   ```

5. Access the UI in your browser at http://127.0.0.1:7860

## 💡 How It Works

1. **Data Processing**: Book descriptions are processed and embedded using OpenAI's embedding model.
2. **User Query**: You describe the kind of book you want in natural language.
3. **Semantic Search**: The system finds books with similar semantic meaning to your query.
4. **Filtering**: Results are filtered by category and emotional tone if specified.
5. **Presentation**: Top recommendations are displayed with book covers and descriptions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset from Kaggle's "7k Books with Metadata"
- Built with LangChain and OpenAI's embedding technologies
- UI created using Gradio

---

<p align="center">
  <b>Made with ❤️ by Onyinye Favour Ezike</b>
</p>
