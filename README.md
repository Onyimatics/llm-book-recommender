---
title: Semantic_Book_Recommender
app_file: gradio-dashboard.py
sdk: gradio
sdk_version: 4.44.1
---
# LLM Book Recommender

This project explores a dataset of books with metadata to build a book recommendation system using LLMs.

## Dataset

The dataset used in this project is the "7k Books with Metadata" dataset from Kaggle, which contains information about approximately 7,000 books including titles, authors, descriptions, ratings, and more.

## Project Structure

- `data-exploration.ipynb`: Jupyter notebook containing data exploration and cleaning
- `books_cleaned.csv`: Cleaned dataset with books that have complete metadata and descriptions with at least 25 words

## Getting Started

1. Clone this repository
2. Set up a virtual environment: `python -m venv .venv`
3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`
4. Install dependencies (requirements.txt will be added later)