import openai
from openai import OpenAIError
import pandas as pd
from pytrends.request import TrendReq
import yfinance as yf
import spacy
from xhtml2pdf import pisa
import streamlit as st
import matplotlib.pyplot as plt
import time
from difflib import get_close_matches
from textblob import TextBlob
# from news_scraper import scrape_latest_business_news, company_sector_df # Assuming this file/data exists
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import praw # Import PRAW

# --- Configuration and Initialization ---
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Reddit API
# Make sure REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT are in your .env file
try:
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT")
    )
except Exception as e:
    st.error(f"Failed to initialize Reddit API: {e}. Please check your .env file and internet connection.")
    reddit = None # Set to None if initialization fails

# Assuming company_sector_df comes from news_scraper or define a dummy if not available
try:
    # If news_scraper is available, uncomment the import at the top and remove the dummy below
    # from news_scraper import company_sector_df
    # If not, use a dummy dataframe structure that includes 'clean_sector'
    if 'company_sector_df' not in locals():
         company_sector_df = pd.DataFrame({'Company': ['DummyCo'], 'Sector': ['Technology'], 'clean_sector': ['technology']})
         # st.warning("Using dummy company_sector_df. Ensure news_scraper.py or a proper df is available.")

except ImportError:
     company_sector_df = pd.DataFrame({'Company': ['DummyCo'], 'Sector': ['Technology'], 'clean_sector': ['technology']})
     # st.warning("Using dummy company_sector_df due to ImportError. Ensure news_scraper.py is available or replace with your data.")


# --- Data Fetching and Processing Functions ---

# 🔹 Web Scraping: Investopedia Definition
def scrape_investopedia_definition(industry_term):
    search_query = industry_term.replace(" ", "+")
    url = f"https://www.investopedia.com/search?q={search_query}"

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.text, "html.parser")

        # Get first result link
        result = soup.select_one("a[data-analytics-label='search-result']")
        if not result:
            return f"No Investopedia article found for '{industry_term}'."

        article_url = result["href"]
        article_response = requests.get(article_url, headers=headers, timeout=10)
        article_response.raise_for_status()
        article_soup = BeautifulSoup(article_response.text, "html.parser")

        # Extract summary paragraph
        paragraph = article_soup.find("p")
        return paragraph.text.strip() if paragraph else "No summary found."

    except requests.exceptions.RequestException as e:
        return f"⚠️ Error scraping Investopedia: Network or HTTP error: {e}"
    except Exception as e:
        return f"⚠️ Error scraping Investopedia: {e}"

# 🔹 Web Scraping: Google News Headlines
def scrape_google_news(industry):
    try:
        query = industry.replace(" ", "+")
        url = f"https://news.google.com/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.select("article h3")
        return [a.get_text() for a in articles[:5]] if articles else ["No Google News results."]
    except requests.exceptions.RequestException as e:
         return [f"⚠️ Google News scrape failed: Network or HTTP error: {e}"]
    except Exception as e:
        return [f"⚠️ Google News scrape failed: {e}"]

# 🔹 Web Scraping: Statista Placeholder (since real-time scraping often blocked)
def get_statista_placeholder(industry):
    # Simulate or return static sample for now
    return f"According to Statista-like projections, the {industry} industry is expected to grow steadily, with digital adoption and AI integration being key drivers."

# 🔹 Web Scraping: Reddit Sentiment Analysis (NEW)
def scrape_reddit_sentiment(industry, limit_posts=15, limit_comments_per_post=10):
    if reddit is None:
        return "Reddit API not initialized. Cannot scrape sentiment."

    sentiment_scores = []
    comment_texts = []
    summary = f"No recent Reddit discussion found for '{industry}'.\n"

    try:
        # Search for submissions related to the industry across Reddit
        # Using a generator for potentially better memory usage with large limits
        search_results = reddit.subreddit("all").search(
            query=industry,
            sort="hot",  # or 'new', 'top', 'relevance'
            limit=limit_posts
        )

        count_comments = 0
        count_posts = 0

        for submission in search_results:
            if submission.locked or submission.stickied:
                continue # Skip locked or stickied posts

            count_posts += 1
            # Load comments (handles "MoreComments" links)
            # Use list() to force loading, limit the number of loaded comments
            submission.comments.replace_more(limit=limit_comments_per_post)

            # Iterate through the flattened list of comments
            for comment in submission.comments.list():
                if isinstance(comment, praw.models.Comment) and comment.body:
                    # Basic sentiment analysis using TextBlob
                    analysis = TextBlob(comment.body)
                    sentiment_scores.append(analysis.sentiment.polarity)
                    comment_texts.append(comment.body)
                    count_comments += 1
                if count_comments >= limit_posts * limit_comments_per_post: # Stop after processing enough comments overall
                    break
            if count_comments >= limit_posts * limit_comments_per_post:
                 break


        if sentiment_scores:
            avg_polarity = sum(sentiment_scores) / len(sentiment_scores)

            # Categorize sentiment
            # Using slightly wider thresholds for "neutral" for social media text
            positive_count = sum(1 for score in sentiment_scores if score > 0.15)
            negative_count = sum(1 for score in sentiment_scores if score < -0.15)
            neutral_count = len(sentiment_scores) - positive_count - negative_count

            summary = (
                f"Analyzed {len(sentiment_scores)} comments from {count_posts} Reddit posts related to '{industry}'.\n"
                f"Overall Sentiment Polarity: {avg_polarity:.2f} (Range: -1 to 1, higher is more positive)\n"
                f"Distribution: Positive ({positive_count}), Negative ({negative_count}), Neutral ({neutral_count})\n"
            )
            # Include a few sample comments for context (shuffle slightly to get diverse samples)
            import random
            random.shuffle(comment_texts)
            sample_comments = "Sample comments:\n" + "\n---\n".join(comment_texts[:min(5, len(comment_texts))])
            summary += sample_comments

        return summary

    except Exception as e:
        return f"⚠️ Error scraping Reddit: {e}"


# 🔹 Data Normalization
def normalize_industry_term(term):
    mapping = {
        "ai-driven micro-investing": "fintech",
        "credit access": "fintech",
        "wealth-building": "personal finance",
        "financial inclusion": "emerging markets finance",
        "credit scoring": "fintech",
        "ai investment": "fintech" # Added a common variation
    }
    term = term.lower()
    for k, v in mapping.items():
        if k in term:
            return v
    return term

# 🔹 ETF Mapping
fallback_etf_map = {
    "technology": "XLK",
    "energy": "XLE",
    "healthcare": "XLV",
    "financial": "XLF",
    "real estate": "XLRE",
    "consumer discretionary": "XLY",
    "utilities": "XLU",
    "industrials": "XLI",
    "materials": "XLB",
    "communications": "XLC",
    "fintech": "FINX",
    "emerging markets finance": "EMFM", # Example, verify ticker
    "blockchain": "BLOK",
    "personal finance": "ARKF", # Example, verify ticker
    "robo advisors": "BOTZ",   # Example, verify ticker
    "sustainable packaging": "PKB", # Example, verify ticker
    "aerospace": "ITA",
    "semiconductors": "SOXX",
    "cybersecurity": "CIBR",
    "renewable energy": "ICLN",
    "clean energy": "PBW"
}

def map_industry_to_etf(industry):
    industry_clean = industry.lower().strip()

    # 1. Try direct match in fallback ETF map
    if industry_clean in fallback_etf_map:
        return fallback_etf_map[industry_clean]

    # 2. Try fuzzy match in fallback ETF map keys
    matches = get_close_matches(industry_clean, fallback_etf_map.keys(), n=1, cutoff=0.7)
    if matches:
        st.info(f"Fuzzy matched input **'{industry}'** to map key **'{matches[0]}'**, using ETF **{fallback_etf_map[matches[0]]}**.")
        return fallback_etf_map[matches[0]]

    # 3. Try fuzzy match against clean sector names from CSV
    if 'company_sector_df' in locals() and not company_sector_df.empty and "clean_sector" in company_sector_df.columns:
        sector_matches = get_close_matches(industry_clean, company_sector_df['clean_sector'].str.lower().unique(), n=1, cutoff=0.7)
        if sector_matches:
            sector = sector_matches[0]
            # Now check if the matched sector has an ETF mapping
            if sector in fallback_etf_map:
                 st.info(f"Fuzzy matched industry **'{industry}'** to sector **'{sector}'** (from data), using ETF **{fallback_etf_map[sector]}**.")
                 return fallback_etf_map[sector]
            else:
                 st.warning(f"Matched industry **'{industry}'** to sector **'{sector}'**, but no ETF found for this specific sector.")


    # 4. If nothing matched
    st.warning(f"No direct or fuzzy ETF mapping found for **'{industry}'**. Try using a more common or broad sector name or add it to the `fallback_etf_map`.")
    return None

# 🔹 Step 4: Static Market Data from CSV
def get_market_insights(industry):
    try:
        df = pd.read_csv("industry_growth.csv")
        if 'GrowthRate' not in df.columns or 'Industry' not in df.columns:
            return "Industry data format error: Missing 'GrowthRate' or 'Industry' column.", df
        filtered = df[df['Industry'].str.lower() == industry.lower()]
        if filtered.empty:
            # Attempt fuzzy match if direct match fails
            matches = get_close_matches(industry.lower(), df['Industry'].str.lower().tolist(), n=1, cutoff=0.8)
            if matches:
                st.info(f"Fuzzy matched industry **'{industry}'** to CSV entry **'{matches[0]}'**.")
                filtered = df[df['Industry'].str.lower() == matches[0]]

        if filtered.empty:
             return f"No industry insights available for '{industry}' in the CSV.", df

        stats = filtered.iloc[0]
        return (
            f"Based on available data, the {stats['Industry']} industry has an annual growth rate of {stats['GrowthRate']}%. Key players include {stats.get('TopCompetitors', 'N/A')}.", df
        )
    except FileNotFoundError:
        return "Error: industry_growth.csv not found. Market insights cannot be loaded.", pd.DataFrame()
    except Exception as e:
        return f"Error reading industry data CSV: {e}", pd.DataFrame()


# 🔹 Step 5: Google Trends
def get_google_trends(industry):
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload([industry], timeframe='today 12-m')
        time.sleep(2) # Be polite with API calls
        interest = pytrends.interest_over_time()
        if interest.empty or industry not in interest.columns:
            return f"No sufficient Google Trend data available for '{industry}'.", None

        # Drop the 'isPartial' column if it exists
        interest = interest.drop(columns=['isPartial'], errors='ignore')

        return (
            f"Search interest in '{industry}' is currently {interest[industry].iloc[-1]} (Index value over last 12 months).",
            interest
        )
    except Exception as e:
        return f"Google Trends error: {e}", None

# 🔹 Step 6: ETF Market Data (using yfinance)
def get_industry_market_summary(industry):
    ticker = map_industry_to_etf(industry)
    if not ticker:
        return f"No ETF mapping found for '{industry}'. Cannot fetch market data.", None
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        if hist.empty:
            return f"No historical data found for ETF '{ticker}'.", None
        start_price = hist["Close"].iloc[0]
        end_price = hist["Close"].iloc[-1]
        change = round(((end_price - start_price) / start_price) * 100, 2)
        return f"The {industry} sector via ETF **{ticker}** changed **{change}%** over the last 6 months (from ${start_price:.2f} to ${end_price:.2f}).", hist
    except Exception as e:
        return f"Error retrieving ETF data for {ticker}: {e}", None


# 🔹 Step 7: NLP (Defined but not used in the main report generation flow currently)
nlp = spacy.load("en_core_web_sm")

def extract_keywords(text):
    doc = nlp(text)
    return list(set(token.text for token in doc if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop))

def extract_named_entities(text):
    return [(ent.text, ent.label_) for ent in nlp(text).ents]

def get_sentiment_blob(text): # Renamed to avoid conflict if needed, using TextBlob
    return TextBlob(text).sentiment


# 🔹 Step 8: Full GPT Report with All Data Sources
def generate_full_report(industry, target_market, goal, budget):
    industry = normalize_industry_term(industry)

    # --- Gather all data sources ---
    with st.spinner(f"Fetching Investopedia data for '{industry}'..."):
        investopedia_insight = scrape_investopedia_definition(industry)
        if "Error" in investopedia_insight or "No Investopedia article" in investopedia_insight:
             investopedia_insight = "No expert insight from Investopedia available. Please consult additional sources."


    with st.spinner(f"Fetching static market insights for '{industry}'..."):
        static_insight, _ = get_market_insights(industry)

    with st.spinner(f"Fetching Google Trends data for '{industry}'..."):
        trend_data_summary, trend_df = get_google_trends(industry)
        if trend_df is None:
             trend_data_summary = "No sufficient Google Trend data available."

    with st.spinner(f"Fetching ETF market data for '{industry}'..."):
        market_summary_text, market_hist_df = get_industry_market_summary(industry)

    # Assuming scrape_latest_business_news exists from news_scraper.py
    try:
         from news_scraper import scrape_latest_business_news
         with st.spinner("Fetching latest business news..."):
             # news_articles is expected to be a list of dicts, e.g., [{'News': 'Headline 1'}, ...]
             news_articles = scrape_latest_business_news()
             news_section = "\n".join(f"- {a['News']}" for a in news_articles[:5]) if news_articles else "No recent news available from financial news sites."
    except ImportError:
         news_section = "Financial news scraper unavailable (news_scraper.py not found)."
         # st.warning("Financial news scraper unavailable. Ensure news_scraper.py is in the same directory.")
    except Exception as e:
         news_section = f"Error fetching financial news: {e}"


    with st.spinner(f"Fetching Google News headlines for '{industry}'..."):
        google_news_headlines_list = scrape_google_news(industry)
        google_headlines = "\n".join(f"- {headline}" for headline in google_news_headlines_list)


    with st.spinner(f"Fetching Statista-like insights for '{industry}'..."):
        statista_insight = get_statista_placeholder(industry)

    with st.spinner(f"Scraping Reddit for community sentiment on '{industry}'..."):
        reddit_sentiment_summary = scrape_reddit_sentiment(industry)
        if "Error scraping Reddit" in reddit_sentiment_summary or "Reddit API not initialized" in reddit_sentiment_summary:
             st.warning(reddit_sentiment_summary) # Show the warning in the UI
             reddit_sentiment_summary = "Could not retrieve Reddit community sentiment."


    # --- Construct the Prompt for GPT-4 ---
    prompt = f"""
    You are an expert AI business analyst with deep knowledge of industry trends, competitive analysis, and financial modeling.

    Based on the user inputs and real-world market data provided below, generate a professional and comprehensive business feasibility report.
    The report should be structured, detailed, and suitable for investor and stakeholder presentations.

    🔹 User Inputs:
    - Industry Focus: {industry}
    - Target Market: {target_market}
    - Business Goal: {goal}
    - Estimated Budget: {budget}

    🔹 Growth Rate/Market Insights (from Industry Data):
    {static_insight}

    🔹 Consumer Trend Insights (from Google Trends):
    {trend_data_summary}

    🔹 Financial Market Analysis (from ETF Performance):
    {market_summary_text}

    🔹 Expert Definition/Insight (from Investopedia):
    {investopedia_insight}

    🔹 Macro Industry Context (Statista-style):
    {statista_insight}

    🔹 Recent Industry Headlines (from Google News):
    {google_headlines}

    🔹 Latest Sector Headlines (from Financial News):
    {news_section}

    🔹 Reddit Community Sentiment Analysis:
    {reddit_sentiment_summary}


    ✅ Please include:
    - Competitor Overview with at least two similar startups or platforms relevant to the target market.
    - Specific numeric assumptions in financial projections (e.g., estimated monthly costs, revenue streams, projected growth rate for breakeven analysis, potential funding rounds). Assume a 6-12 month outlook where specific numbers are required.
    - Strategic recommendations for MVP launch, capital efficiency, and user traction specific to the industry and target market.
    - Use estimated figures for TAM/SAM/SOM if possible, based on industry context.

    📊 Report Structure:
    1. Executive Market Summary (Highlight the industry scope, relevance, and key findings from the data provided).
    2. SWOT Analysis (Strengths, Weaknesses, Opportunities, Threats based on the industry, target market, and gathered data).
    3. Financial Forecast (Provide a 6-12 month outlook including estimated startup costs based on budget, potential revenue streams, projected growth, and estimated breakeven point. State assumptions clearly).
    4. Industry Scale and Growth Trends (Discuss macro view, CAGR if indicated, expansion potential, TAM/SAM/SOM estimations).
    5. Competitive Landscape (Analyze direct/indirect alternatives and key players mentioned or known in the industry).
    6. Tactical Recommendations (Outline go-to-market strategy, marketing channels, user acquisition).
    7. Pre-launch Readiness & Capital Planning (Discuss key resources needed like tech stack considerations, potential team roles, phase rollout plan, and advice on managing budget/seeking further capital).
    8. Final Assessment (Summarize overall feasibility, potential contingencies, relevant compliance considerations, fallback paths, and a clear "Next Steps" recommendation).

    Use a formal, analytical, and confident business tone suitable for investor and stakeholder presentations. Ensure insights are backed by estimated figures and data points derived or inferred from the provided context.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Using gpt-4o for potentially better performance and cost
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=2000 # Increased max tokens to accommodate more detail
        )
        return response.choices[0].message.content.strip()
    except OpenAIError as e:
        return f"\u274c Error generating full report: {e}"
    except Exception as e:
         return f"\u274c An unexpected error occurred during report generation: {e}"


# 🔹 Step 9: Export Report to PDF (Formatted with charts)
# This version is designed to work with the Streamlit app and include saved chart images
def export_to_pdf(report_content, output_file="business_report.pdf", chart_paths=None):
    # Simple formatting for HTML
    formatted_content = report_content.replace("\n", "<br>")
    # Replace markdown headers with HTML tags for better PDF rendering
    formatted_content = formatted_content.replace("### ", "<h3>").replace("## ", "<h2>").replace("# ", "<h1>")
    formatted_content = formatted_content.replace("📊 Report Structure:", "<h2>Report Structure:</h2>")
    formatted_content = formatted_content.replace("✅ Please include:", "<h2>Key Requirements:</h2>")
     # Basic list formatting
    formatted_content = formatted_content.replace("<br>- ", "<br>• ")


    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 30px; line-height: 1.6; }}
            h1 {{ color: #1f4e79; }}
            h2 {{ color: #1f4e79; margin-top: 20px; }}
            h3 {{ color: #4472c4; margin-top: 15px; }}
            .report-block {{ background-color: #f0f2f6; padding: 15px; border-radius: 8px; }}
            img {{ max-width: 100%; height: auto; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <h1>AI-Generated Business Feasibility Report</h1>
        <div class="report-block">{formatted_content}</div>
    """
    # Add images if paths are provided
    if chart_paths:
        html_content += "<h2>Visual Market Analysis</h2>"
        for chart_path in chart_paths:
            if os.path.exists(chart_path):
                html_content += f'<img src="{chart_path}"><br>'
            else:
                 html_content += f'<p>Chart not found: {chart_path}</p><br>'


    html_content += "</body></html>"

    @st.cache_data # Cache the PDF creation process
    def create_pdf(source_html, output_filename):
        result_file = open(output_filename, "w+b")
        pisa_status = pisa.CreatePDF(source_html, dest=result_file)
        result_file.close()
        return pisa_status.err

    err = create_pdf(html_content, output_file)

    if err:
        st.error("Error creating PDF.")
        return False
    return True


# 🔹 Step 10: Charts (Functions to plot using matplotlib)
def plot_trends_chart(trend_df, industry):
    if trend_df is None or trend_df.empty:
        st.warning("No Google Trends data to plot.")
        return None # Return None if no plot is generated
    try:
        keyword_col = trend_df.columns[0] # The industry name will be the first column name
        fig, ax = plt.subplots(figsize=(10, 4))
        trend_df[keyword_col].plot(ax=ax, title=f"Google Trends for '{keyword_col}' (12 months)")
        ax.set_ylabel("Search Interest (Index)")
        ax.set_xlabel("Date")
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error in Google Trends chart: {e}")
        return None


def plot_etf_price_trend(hist, industry):
    if hist is None or hist.empty:
        st.warning("No ETF price data available to plot.")
        return None # Return None if no plot is generated
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        hist["Close"].plot(ax=ax, title=f"{industry.capitalize()} Sector - ETF Price Trend (6 months)")
        ax.set_ylabel("Price (USD)")
        ax.set_xlabel("Date")
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error plotting ETF price trend: {e}")
        return None


def plot_growth_bar(df):
    if df is None or df.empty or 'GrowthRate' not in df.columns or 'Industry' not in df.columns:
         st.warning("No industry growth data to plot.")
         return None
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Ensure data is sorted and take top N if needed
        df_sorted = df.sort_values(by="GrowthRate", ascending=False).head(15) # Limit to top 15 for clarity
        ax.barh(df_sorted["Industry"], df_sorted["GrowthRate"], color='skyblue')
        ax.set_title("Top Growing Industries by Growth Rate (%)")
        ax.set_xlabel("Growth Rate (%)")
        ax.invert_yaxis() # Highest growth at the top
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error plotting growth bar chart: {e}")
        return None


# --- Streamlit UI Layout ---

# Set page config
st.set_page_config(page_title="AI Business Report Generator", layout="wide", initial_sidebar_state="collapsed")

# Initialize state
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "report_content" not in st.session_state:
    st.session_state.report_content = None
if "chart_paths" not in st.session_state:
     st.session_state.chart_paths = {} # Dictionary to store saved chart paths


page = st.session_state.page

# Inject JS & CSS for feedback (assuming button-29.mp3 is accessible or replace with a local sound)
st.markdown("""
    <script>
        function playSound() {
            var audio = new Audio("https://www.soundjay.com/buttons/sounds/button-29.mp3"); // Ensure this URL is valid or use a local path
            audio.play();
        }
    </script>
    <style>
        div.stButton > button:first-child {
            height: 3em;
            font-weight: bold;
            border-radius: 8px;
            border: none;
            background-color: #dc3545; /* Red */
            color: white;
            transition: all 0.3s ease;
            padding: 0.5em 1em; /* Added padding */
        }
        div.stButton > button:first-child.enabled {
            background-color: #28a745 !important; /* Green when enabled */
        }
        .report-block {
            background-color: #0e1117; /* Dark background */
            color: #f0f2f6; /* Light text */
            padding: 1.5em;
            border-radius: 12px;
            border: 1px solid #444; /* Subtle border */
            white-space: pre-wrap; /* Preserve line breaks */
            word-wrap: break-word; /* Break long words */
        }
         .stSpinner > div {
            border-top-color: #28a745 !important; /* Green spinner */
        }
        h1, h2, h3, h4, h5, h6 {
             color: #1f4e79; /* Dark blue heading color */
        }
         .stMarkdown strong {
            color: #4CAF50; /* Highlight strong text (like % change) */
        }
         .stAlert {
             margin-top: 15px;
             margin-bottom: 15px;
         }
    </style>
""", unsafe_allow_html=True)

# Home Page
if page == "Home":
    st.markdown("""
        <h1 style='text-align: center; color: #1f4e79;'>AI-Powered Business Report Generator</h1>
        <h4 style='text-align: center; color: #ccc;'>Generate investor-ready business reports using GPT-4, market data, and community sentiment.</h4>
    """, unsafe_allow_html=True)

    with st.form("report_form"):
        st.markdown("### Enter Business Idea Details")
        industry = st.text_input("Industry Focus", placeholder="e.g. Sustainable Packaging, AI in Healthcare, Fintech")
        target_market = st.text_input("Target Market", placeholder="e.g. Small businesses in Europe, Gen Z in Southeast Asia")
        goal = st.text_area("Business Goal / Problem Solved", placeholder="Describe your product or service idea and what problem it solves")
        budget = st.text_input("Estimated Initial Budget", placeholder="e.g. $10,000, €50k")

        # Report Type Selection (removed Summary option to focus on Full)
        # report_type = st.selectbox("Report Type", ["Full Report"]) # Keep only 'Full'

        form_complete = all([industry, target_market, goal, budget])
        # Use the form_complete variable to control button state/style
        submit_button_html = f"""
            <button type="submit" class="stButton" {'enabled' if form_complete else ''} {'disabled' if not form_complete else ''}>
                Generate Full Report
            </button>
        """
        st.markdown(submit_button_html, unsafe_allow_html=True)
        # Use a hidden button purely for form submission detection
        submit = st.form_submit_button("Generate Full Report", type="primary", help="Fill all fields to enable", disabled=not form_complete)


    if submit and form_complete:
        st.session_state.industry = industry
        st.session_state.target_market = target_market
        st.session_state.goal = goal
        st.session_state.budget = budget
        st.session_state.report_content = None # Clear previous report
        st.session_state.chart_paths = {} # Clear previous charts
        st.session_state.page = "Generating Report"
        st.components.v1.html("<script>playSound();</script>", height=0)
        st.rerun()

# Generating Report Page (Intermediate state)
elif page == "Generating Report":
     st.markdown("<h2 style='text-align: center; color: #1f4e79;'>Generating Business Report...</h2>", unsafe_allow_html=True)
     st.info("Gathering data and generating insights. This may take a moment...")
     # The actual generation happens here
     industry = st.session_state.industry
     target_market = st.session_state.target_market
     goal = st.session_state.goal
     budget = st.session_state.budget

     # Call the main report generation function
     report_text = generate_full_report(industry, target_market, goal, budget)

     # Save report and charts to session state
     st.session_state.report_content = report_text

     # Generate and save charts
     st.session_state.chart_paths = {} # Clear previous chart paths

     # Get data again for plotting (could optimize by returning data from generate_full_report)
     _, growth_df = get_market_insights(industry)
     _, trend_df = get_google_trends(industry)
     _, market_hist_df = get_industry_market_summary(industry)

     # Plot and save charts
     fig_trend = plot_trends_chart(trend_df, industry)
     if fig_trend:
         fig_trend.savefig("trend_chart.png")
         st.session_state.chart_paths['trend'] = "trend_chart.png"
         plt.close(fig_trend) # Close the figure to free memory

     fig_growth = plot_growth_bar(growth_df)
     if fig_growth:
         fig_growth.savefig("growth_chart.png")
         st.session_state.chart_paths['growth'] = "growth_chart.png"
         plt.close(fig_growth)

     fig_etf = plot_etf_price_trend(market_hist_df, industry)
     if fig_etf:
         fig_etf.savefig("etf_chart.png")
         st.session_state.chart_paths['etf'] = "etf_chart.png"
         plt.close(fig_etf)


     st.session_state.page = "Generated Report"
     st.rerun()


# Generated Report Page
elif page == "Generated Report" and st.session_state.get("report_content") is not None:
    report_content = st.session_state.report_content
    industry = st.session_state.industry # Retrieve industry for displaying charts

    st.markdown("<h2 style='color:#1f4e79;'>Business Feasibility Report</h2>", unsafe_allow_html=True)
    st.markdown(f"<div class='report-block'>{report_content}</div>", unsafe_allow_html=True)


    # Display Visualizations
    st.markdown("---")
    st.markdown("<h3 style='color:#1f4e79;'>Visual Market Analysis</h3>", unsafe_allow_html=True)

    # Display charts saved during generation
    if 'trend' in st.session_state.chart_paths and os.path.exists(st.session_state.chart_paths['trend']):
         st.image(st.session_state.chart_paths['trend'], caption=f"Google Trends for '{industry}'")
    else:
         st.warning("Google Trends chart not available.")

    if 'growth' in st.session_state.chart_paths and os.path.exists(st.session_state.chart_paths['growth']):
         st.image(st.session_state.chart_paths['growth'], caption="Top Growing Industries")
    else:
         st.warning("Industry Growth chart not available.")

    if 'etf' in st.session_state.chart_paths and os.path.exists(st.session_state.chart_paths['etf']):
         st.image(st.session_state.chart_paths['etf'], caption=f"{industry.capitalize()} Sector - ETF Price Trend")
    else:
         st.warning("ETF Price Trend chart not available.")


    # Download Report Section
    st.markdown("---")
    st.markdown("<h3 style='color:#1f4e79;'>Download Report</h3>", unsafe_allow_html=True)

    # Create the PDF with the report text and chart images
    pdf_output_filename = "business_report.pdf"
    chart_paths_list = list(st.session_state.chart_paths.values())

    if export_to_pdf(report_content, pdf_output_filename, chart_paths=chart_paths_list):
        with open(pdf_output_filename, "rb") as f:
            st.download_button(
                "Download PDF Report with Charts",
                f,
                file_name=pdf_output_filename,
                mime="application/pdf"
            )
    else:
         st.error("Failed to generate PDF.")

    # Back Button
    if st.button("⬅ Back to Home"):
        # Clean up saved chart files on going back
        for path in st.session_state.chart_paths.values():
             if os.path.exists(path):
                 os.remove(path)
        st.session_state.chart_paths = {} # Clear session state
        st.session_state.report_content = None # Clear report content
        st.session_state.page = "Home"
        st.rerun()

# Fallback for direct access to generated page without generation
else:
    st.info("Please generate a report from the Home page first.")
    if st.button("Go to Home"):
         st.session_state.page = "Home"
         st.rerun()