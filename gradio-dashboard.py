import os
import pickle
import hashlib
from pathlib import Path
import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document

import gradio as gr

load_dotenv()

# Define cache directory
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


# Function to generate a cache key based on document content
def generate_cache_key(documents):
    content_hash = hashlib.md5()
    for doc in documents:
        content_hash.update(doc.page_content.encode())
    return content_hash.hexdigest()


# Load books
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# Load documents
raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)


# Batching function with caching
def batch_process_documents(documents: list[Document], batch_size: int = 200, sleep_time: int = 65) -> Chroma:
    """
    Process documents in smaller batches with built-in rate limiting.
    Uses caching to avoid reprocessing when possible.
    """
    # Generate a cache key based on document content
    cache_key = generate_cache_key(documents)
    chroma_path = CACHE_DIR / f"chroma_{cache_key}"

    # Check if cached Chroma DB exists
    if chroma_path.exists():
        print(f"Loading cached embeddings from {chroma_path}")
        embeddings = OpenAIEmbeddings()
        return Chroma(persist_directory=str(chroma_path), embedding_function=embeddings)

    print("No cache found. Processing documents...")
    embeddings = OpenAIEmbeddings()

    if not documents:
        raise ValueError("No documents provided for processing.")

    start_idx = 0
    end_idx = min(batch_size, len(documents))

    print(f"Processing initial batch: {start_idx} to {end_idx}")
    db = Chroma.from_documents(
        documents[start_idx:end_idx],
        embedding=embeddings,
        persist_directory=str(chroma_path)
    )

    while end_idx < len(documents):
        print(f"Sleeping for {sleep_time} seconds to respect rate limits...")
        time.sleep(sleep_time)

        start_idx = end_idx
        end_idx = min(start_idx + batch_size, len(documents))

        print(f"Adding batch: {start_idx} to {end_idx}")
        db.add_documents(documents[start_idx:end_idx])

    print(f"Finished processing all documents. Cached at {chroma_path}")
    return db


# Create database using batching (now with caching)
db_books = batch_process_documents(documents)


# Functions
def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results


# Gradio UI
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book:",
                                placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)

# Launch app
if __name__ == "__main__":
    dashboard.launch()