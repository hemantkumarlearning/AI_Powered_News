import streamlit as st
import requests
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer, util


progress = st.progress(0)
progress.progress(50)
progress.progress(100)

API_KEY = 'Your NewsAPI key'

llm = ChatGroq(
    temperature=0,
    groq_api_key='Your GroqAPI key',
    model_name='llama-3.3-70b-versatile'
)

prompt_template = """Summarize the following articles related to "{query}". Here are the articles:
{articles}

Provide a concise summary of the topic based on the above content."""

prompt = PromptTemplate(input_variables=["query", "articles"], template=prompt_template)

llm_chain = LLMChain(prompt=prompt, llm=llm)

# Function to fetch news articles
@st.cache_data
def fetch_news(query, page_size=3):
    url = f'https://newsapi.org/v2/everything?q={query}&pageSize={page_size}&apiKey={API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json()['articles']
        return [
            {'title': article['title'], 'description': article['description'], 
            'url': article['url'], 'publishedAt': article['publishedAt']}
            for article in articles
        ]
    else:
        return []

st.title("AI-Powered News")
st.subheader("Search for the latest news on any topic!")

embedder = SentenceTransformer('all-MiniLM-L6-v2')

query = st.text_input("Enter your query", "")
query_embedding = embedder.encode(query, convert_to_tensor=True)

if query:
    articles = fetch_news(query)

    if articles:
       
        article_embeddings = [embedder.encode(article["description"], convert_to_tensor=True) for article in articles]


        similarities = [util.cos_sim(query_embedding, emb).item() for emb in article_embeddings]

        top_k = 2
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
        top_articles = [articles[i] for i in top_indices]
        combined_text = " ".join([article["description"] for article in top_articles])

        summary = llm_chain.run(query=query, articles=combined_text)
        st.write(f"### Summary of the top {len(articles)} articles on '{query}':")
        highlighted_paragraph = f"""
        <mark style="background-color: #ff5733; color: white;">{summary}</mark>
        """
        st.markdown(highlighted_paragraph, unsafe_allow_html=True)
    
        if st.checkbox(f"Found {len(articles)} articles. Displaying top results:"):

            for i, article in enumerate(top_articles):
                st.markdown(f"### {article['title']}")
                st.markdown(f"Published on: {article['publishedAt']}")
                st.markdown(f"[Read full article]({article['url']})")
                st.markdown("---")

    else:
        st.write("No articles found. Try a different query.")
