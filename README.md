# AI-Powered News Summarizer
An AI-powered news summarization tool that takes user queries, fetches relevant news articles, and provides concise summaries. Built using News API, Sentence Transformers, Cosine Similarity, Groq LLM, and Langchain, with a Streamlit UI for a seamless user experience.

## Features
- Query-based News Search: Fetches the latest news based on user queries.
- Advanced Similarity Matching: Uses Sentence Transformers to convert articles to vectors and compares them with the query using Cosine Similarity to find the most relevant news.
- Dynamic News Summaries: Groups the top articles' summaries and sends them to a Groq LLM model to generate concise, human-readable summaries.
- Interactive UI: A clean, interactive interface built using Streamlit, enabling users to input queries and view summarized news instantly.

## Tech Stack
- News API: Used to fetch real-time news articles based on user input.
- Sentence Transformers: Converts text to embeddings for similarity comparison.
- Cosine Similarity: Measures the similarity between the query and fetched news articles.
- Groq LLM Model: Powers the summarization of the grouped descriptions into short summaries.
- Langchain: Manages the pipeline and interaction between different models and tools.
- Streamlit: Used to build a simple and interactive user interface.

## Installation
1. Clone this repository:
```
git clone https://github.com/hemantkumarlearning/AI_Powered_News.git
cd AI_Powered_News
```
2. Create a virtual environment:
```
python -m venv venv
```
3. Activate the virtual environment:
```
venv\Scripts\activate
```
4. Install the dependencies:
```
pip install -r requirements.txt
```
5. Run the application:
```
streamlit run app.py
```
## Usage
1. Open your browser and go to the Streamlit interface.
2. Enter a query in the input box.
3. View the summarized news articles based on your query.
