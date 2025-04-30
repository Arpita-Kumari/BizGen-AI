# BizGen-AI

BizGen-AI is an AI-powered business report generator that helps small businesses and entrepreneurs instantly generate detailed, data-driven feasibility reports. Built with GPT-4, Retrieval-Augmented Generation (RAG), and real-time web scraping, the tool automates market analysis, financial forecasting, and strategic recommendations.

# Key Features

- GPT-4 Integration: Uses advanced generative AI to compose structured, investor-ready business reports based on user input.

- Retrieval-Augmented Generation (RAG): Retrieves relevant Reddit discussions using FAISS vector databases for more personalized and context-aware analysis.

- Real-Time Data Collection:

    - Google Trends (via PyTrends): Shows current market interest and consumer behavior.

    - Yahoo Finance (via yFinance): Maps industry keywords to ETFs and provides sector-level performance.

    - Investopedia & News: Scrapes definitions and current headlines using BeautifulSoup.

- Natural Language Processing (NLP):

    - Keyword extraction, Named Entity Recognition (SpaCy), and Sentiment Analysis (TextBlob).

- Interactive Streamlit Web App: User-friendly interface for input, report visualization, and PDF export.

- PDF Export: Automatically formats the report and charts into a professional downloadable PDF.

# Technologies Used

- Python 3.9+

- OpenAI GPT-4 API

- Streamlit

- FAISS (Vector Search)

- PRAW (Reddit Scraper)

- PyTrends, yFinance, BeautifulSoup, Newspaper3k

- SpaCy, TextBlob

- Matplotlib

- xhtml2pdf

# Project Structure

├── app2.py                  # Main Streamlit web interface

├── rag_vector_DB.py         # FAISS-based vector store setup for Reddit post embeddings

├── reddit_rag_scraper.py    # Reddit post scraper and cleaner using PRAW

├── news_scraper.py          # Scrapers for Investopedia, Google News, and financial headlines

├── industry_growth.csv      # Static data with CAGR and competitor insights

├── company_with_sectors.csv # Industry-to-ETF mapping support

├── charts/                  # Auto-generated visualizations (PNG)

└── README.md

# Version Control & Collaboration

- Team collaboration was managed using GitHub branching and pull requests.

- Each member worked on individual branches (e.g., streamlit-ui, rag-system, pdf-export) and merged through reviewed pull requests to ensure clean integration.

# Sample Report Output

- Executive Summary

- SWOT Analysis

- Financial Forecast (Revenue, Cost, Breakeven)

- Market Trends (TAM/SAM/SOM, Google Trends, ETF)

- Reddit Insights (RAG)

- Strategic Recommendations

- Exported PDF with charts

# Run Locally

git clone https://github.com/<your-username>/BizGen-AI.git
cd BizGen-AI
pip install -r requirements.txt
streamlit run app2.py
