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
# Assuming news_scraper.py provides scrape_latest_business_news and company_sector_df
# Assuming reddit_rag_scraper.py provides get_reddit_posts, build_vector_db_from_texts, retrieve_relevant_docs
from news_scraper import scrape_latest_business_news, company_sector_df
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from reddit_rag_scraper import get_reddit_posts
from rag_vector_DB import build_vector_db_from_texts, retrieve_relevant_docs

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("SpaCy model 'en_core_web_sm' not found. Please install it by running: python -m spacy download en_core_web_sm")
    st.stop()


def scrape_investopedia_definition(industry_term):
    """Scrapes the first paragraph definition from Investopedia for a given term."""
    search_query = industry_term.replace(" ", "+")
    url = f"https://www.investopedia.com/search?q={search_query}"

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        # Get first result link
        result = soup.select_one("a[data-analytics-label='search-result']")
        if not result:
            return f"No Investopedia article found for '{industry_term}'."

        article_url = result["href"]
        # Add a delay to be polite to the server
        time.sleep(1)
        article_response = requests.get(article_url, headers=headers)
        article_soup = BeautifulSoup(article_response.text, "html.parser")

        # Extract summary paragraph
        # Look for common tags used for definition paragraphs, e.g., <p>, <h2> + <p>
        paragraph = article_soup.find("p")
        if paragraph:
             # Basic cleaning: remove excessive whitespace
            return paragraph.text.strip().replace('\n', ' ')
        else:
            return "No summary found on the Investopedia page."

    except requests.exceptions.RequestException as e:
        return f"⚠️ Network error scraping Investopedia: {e}"
    except Exception as e:
        return f"⚠️ Error scraping Investopedia: {e}"


# 🔹 Web Scraping: Google News Headlines
def scrape_google_news(industry):
    """Scrapes top 5 Google News headlines for a given industry."""
    try:
        query = industry.replace(" ", "+")
        url = f"https://news.google.com/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.select("article h3")
        return [a.get_text() for a in articles[:5]] if articles else ["No Google News results."]
    except Exception as e:
        return [f"⚠️ Google News scrape failed: {e}"]

# 🔹 Web Scraping: Statista Placeholder (since real-time scraping often blocked)
def get_statista_placeholder(industry):
    """Provides a placeholder Statista-like insight."""
    # Simulate or return static sample for now
    return f"According to hypothetical data, the {industry} industry is expected to grow steadily, with digital adoption and AI integration being key drivers. Market size is estimated to be [Insert Estimated Market Size] by [Insert Year]."


def normalize_industry_term(term):
    """Normalizes industry terms to a standard format for better mapping."""
    mapping = {
        "ai-driven micro-investing": "fintech",
        "credit access": "fintech",
        "wealth-building": "personal finance",
        "financial inclusion": "emerging markets finance",
        "credit scoring": "fintech",
        "sustainable packaging": "packaging", # Mapping to a broader sector if specific ETF not available
        "renewable energy": "clean energy",
        "robo advisors": "fintech" # Often falls under fintech
    }
    term = term.lower().strip()
    for k, v in mapping.items():
        if k in term:
            return v
    return term

# Updated and expanded fallback ETF mapping
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
    "fintech": "FINX", # Global X FinTech ETF
    "emerging markets finance": "EMFM", # Global X Emerging Markets Financials ETF
    "blockchain": "BLOK", # Amplify Transformational Data Sharing ETF
    "personal finance": "ARKF", # ARK Fintech Innovation ETF (covers some personal finance tech)
    "packaging": "PKB", # Invesco Dynamic Building & Construction ETF (often includes packaging materials)
    "aerospace": "ITA", # iShares U.S. Aerospace & Defense ETF
    "semiconductors": "SOXX", # iShares Semiconductor ETF
    "cybersecurity": "CIBR", # First Trust NASDAQ Cybersecurity ETF
    "clean energy": "PBW", # Invesco WilderHill Clean Energy ETF
    "biotechnology": "IBB", # iShares Nasdaq Biotechnology ETF
    "cloud computing": "CLOU", # Global X Cloud Computing ETF
    "e-commerce": "IBUY", # Amplify Online Retail ETF
    "gaming": "GAMR", # ETFMG Video Game Tech ETF
    "robotics": "ROBO", # ROBO Global Robotics and Automation Index ETF
    "artificial intelligence": "BOTZ", # Global X Robotics & Artificial Intelligence ETF
    "space exploration": "ARKX", # ARK Space Exploration & Innovation ETF
    "ंकन": "CNCR" # ProShares Ultra Nasdaq Cybersecurity (example, need to confirm ticker)
}

def map_industry_to_etf(industry):
    """Maps an industry term to a relevant ETF ticker."""
    industry_clean = normalize_industry_term(industry)

    # 1. Try direct match in fallback ETF map
    if industry_clean in fallback_etf_map:
        return fallback_etf_map[industry_clean]

    # 2. Try fuzzy match in fallback ETF map keys
    matches_keys = get_close_matches(industry_clean, fallback_etf_map.keys(), n=1, cutoff=0.7)
    if matches_keys:
        matched_key = matches_keys[0]
        st.info(f"Fuzzy matched input **'{industry}'** to ETF map key **'{matched_key}'**, using ETF **{fallback_etf_map[matched_key]}**.")
        return fallback_etf_map[matched_key]

    # 3. Try fuzzy match against company_sector_df sectors (if available)
    if company_sector_df is not None and "clean_sector" in company_sector_df.columns:
        sector_matches = get_close_matches(industry_clean, company_sector_df['clean_sector'].str.lower().unique(), n=1, cutoff=0.7)
        if sector_matches:
            sector = sector_matches[0]
            # Check if the matched sector exists in the ETF map
            if sector in fallback_etf_map:
                st.info(f"Matched industry **'{industry}'** to CSV sector **'{sector}'**, using ETF **{fallback_etf_map[sector]}**.")
                return fallback_etf_map[sector]
            else:
                 st.warning(f"Matched industry **'{industry}'** to CSV sector **'{sector}'**, but no ETF mapping found for this sector.")


    # 4. If nothing matched
    st.warning(f"No direct or close ETF mapping found for **'{industry}'**. Market data for this sector may not be available.")
    return None


# 🔹 Step 3: GPT-4 Summary Report (Refined Prompt)
def get_business_feasibility_summary(industry, target_market, goal, budget):
    """Generates a concise business feasibility summary using GPT-4."""
    prompt = f"""
    You are a seasoned business analyst providing a high-level feasibility assessment.
    Analyze the following business concept and provide a concise summary report focused on key feasibility aspects.

    Business Idea Overview:
    - Industry: {industry}
    - Target Market: {target_market}
    - Business Goal/Concept: {goal}
    - Estimated Budget: {budget}

    Provide a summary with these sections:
    1. Executive Summary: Briefly state the business idea and its potential feasibility.
    2. Market Potential Snapshot: Comment on the apparent market opportunity.
    3. Primary Challenges/Risks: Identify the most significant hurdles.
    4. Feasibility Outlook: Give a brief assessment of the likelihood of success based on the limited information.
    5. Initial Recommendation: Suggest the immediate next step (e.g., further research, pilot program).

    Keep the response focused and brief, suitable for an initial review.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Using a potentially newer model if available and suitable
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=600 # Increased max tokens slightly for more detail if needed
        )
        return response.choices[0].message.content.strip()
    except OpenAIError as e:
        return f"\u274c Error generating summary report: {e}"


# 🔹 Step 4: Static Market Data from CSV
def get_market_insights(industry):
    """Reads industry growth data from a CSV file."""
    try:
        df = pd.read_csv("industry_growth.csv")
        if 'GrowthRate' not in df.columns or 'Industry' not in df.columns:
            return "Industry data format error: Missing 'GrowthRate' or 'Industry' column.", None
        # Ensure 'GrowthRate' is numeric, coercing errors
        df['GrowthRate'] = pd.to_numeric(df['GrowthRate'], errors='coerce')
        # Drop rows where GrowthRate is not a number
        df = df.dropna(subset=['GrowthRate'])

        # Fuzzy match the industry input to the CSV industries
        industry_lower = industry.lower().strip()
        csv_industries_lower = df['Industry'].str.lower().tolist()
        matches = get_close_matches(industry_lower, csv_industries_lower, n=1, cutoff=0.8) # Increased cutoff for better match

        if not matches:
            return f"No specific industry insights found for '{industry}'. Data available for other sectors.", df

        matched_industry = matches[0]
        filtered = df[df['Industry'].str.lower() == matched_industry]

        if filtered.empty:
             return f"Error retrieving data for matched industry '{matched_industry}'.", df

        stats = filtered.iloc[0]
        # Provide more detailed insight if available
        top_competitors = stats.get('TopCompetitors', 'N/A')
        market_size = stats.get('MarketSize', 'N/A') # Assuming MarketSize column might exist
        market_year = stats.get('MarketSizeYear', 'N/A') # Assuming MarketSizeYear column might exist


        insight = f"Based on available data (matched to '{matched_industry}'), the '{industry}' industry has an estimated annual growth rate of {stats['GrowthRate']}%. "
        if market_size != 'N/A' and market_year != 'N/A':
            insight += f"The market size was approximately ${market_size} in {market_year}. "
        if top_competitors != 'N/A':
            insight += f"Key players include: {top_competitors}."

        st.info(f"Matched your input '{industry}' to CSV data for '{matched_industry}'.")
        return insight, df

    except FileNotFoundError:
        return "Industry growth data file (industry_growth.csv) not found.", None
    except Exception as e:
        return f"Error reading industry data: {e}", None


# 🔹 Step 5: Google Trends
def get_google_trends(industry):
    """Fetches Google Trends data for a given industry term."""
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        # Use the normalized industry term for trends
        trends_term = normalize_industry_term(industry)
        pytrends.build_payload([trends_term], timeframe='today 12-m', geo='') # Added geo='' for worldwide trends

        time.sleep(2) # Add a delay
        interest = pytrends.interest_over_time()

        if interest.empty or trends_term not in interest.columns:
            return f"No trend data available for '{industry}' (using term '{trends_term}').", None

        # Drop the 'isPartial' column if it exists
        if 'isPartial' in interest.columns:
            interest = interest.drop(columns=['isPartial'])

        latest_interest = interest[trends_term].iloc[-1]
        # Calculate change over the period
        if len(interest) > 1:
             change_percentage = ((latest_interest - interest[trends_term].iloc[0]) / interest[trends_term].iloc[0]) * 100 if interest[trends_term].iloc[0] != 0 else 0
             trend_summary = f"Search interest in '{industry}' (as '{trends_term}') is currently {latest_interest} (last recorded week), showing a change of {change_percentage:.2f}% over the last 12 months."
        else:
            trend_summary = f"Search interest in '{industry}' (as '{trends_term}') is currently {latest_interest} (last recorded week)."


        return trend_summary, interest

    except Exception as e:
        return f"Google Trends error: {e}", None


def get_industry_market_summary(industry):
    """Fetches and summarizes ETF market data for a given industry."""
    ticker = map_industry_to_etf(industry)
    if not ticker:
        return f"Could not find a suitable ETF for '{industry}'. Market data unavailable.", None
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y") # Changed to 1 year for a broader view
        if hist.empty:
            return f"No historical data found for ETF '{ticker}'. Market data unavailable.", None
        start_price = hist["Close"].iloc[0]
        end_price = hist["Close"].iloc[-1]
        change_percent = round(((end_price - start_price) / start_price) * 100, 2) if start_price != 0 else 0

        # Get recent performance data (e.g., last month)
        hist_recent = stock.history(period="1mo")
        recent_change = None
        if not hist_recent.empty and len(hist_recent) > 1:
             recent_start_price = hist_recent["Close"].iloc[0]
             recent_end_price = hist_recent["Close"].iloc[-1]
             recent_change = round(((recent_end_price - recent_start_price) / recent_start_price) * 100, 2) if recent_start_price != 0 else 0


        market_summary = f"The **{industry}** sector, represented by ETF **{ticker}**, showed a change of **{change_percent}%** over the past year (from ${start_price:.2f} to ${end_price:.2f})."
        if recent_change is not None:
            market_summary += f" In the last month, it changed by **{recent_change}%**."


        return market_summary, hist
    except Exception as e:
        return f"Error retrieving ETF data for {ticker}: {e}", None


# 🔹 Step 7: NLP (Keep as is, potentially for later use or analysis within the report)
# nlp loaded at the beginning

def extract_keywords(text):
    """Extracts keywords (NOUN, PROPN) from text."""
    doc = nlp(text)
    return list(set(token.text for token in doc if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop))

def extract_named_entities(text):
    """Extracts named entities from text."""
    return [(ent.text, ent.label_) for ent in nlp(text).ents]

def get_sentiment(text):
    """Calculates sentiment score for text."""
    return TextBlob(text).sentiment

# 🔹 Step 8: Full GPT Report with Real-Time Data (Improved Prompt)
def generate_full_report(industry, target_market, goal, budget):
    """Generates a comprehensive business feasibility report using GPT-4 and various data sources."""
    normalized_industry = normalize_industry_term(industry)

    # Gather data from various sources
    static_insight, growth_df = get_market_insights(normalized_industry)
    trend_data_summary, trend_df = get_google_trends(normalized_industry)
    market_summary, etf_hist = get_industry_market_summary(normalized_industry)
    investopedia_insight = scrape_investopedia_definition(industry) # Use original industry for definition
    google_news_headlines = scrape_google_news(industry) # Use original industry for news
    statista_insight = get_statista_placeholder(industry) # Use original industry for placeholder
    latest_business_news = scrape_latest_business_news() # General business news

    # Process news and Reddit data for RAG context
    news_texts = [f"{article.get('Title', '')}: {article.get('News', '')}" for article in latest_business_news[:10]] # Limit news articles
    reddit_posts = get_reddit_posts(normalized_industry) # Use normalized industry for Reddit search

    rag_texts = news_texts + reddit_posts # Combine news and reddit for RAG
    rag_context = "No relevant real-time insights found."
    if rag_texts:
        try:
            reddit_db = build_vector_db_from_texts(rag_texts)
            relevant_docs = retrieve_relevant_docs(reddit_db, f"{industry} market trends and challenges") # Query vector DB
            rag_context = "\n".join([doc.page_content for doc in relevant_docs])
            if not rag_context.strip():
                 rag_context = "No highly relevant real-time insights found, but general discussions are available."
        except Exception as e:
            rag_context = f"Error processing real-time data for RAG: {e}"


    # Format data for the prompt
    google_news_section = "\n".join(f"- {headline}" for headline in google_news_headlines) if google_news_headlines else "No Google News headlines available."
    latest_business_news_section = "\n".join(f"- {article.get('Title', 'N/A')}: {article.get('News', 'No summary available.')}" for article in latest_business_news[:5]) if latest_business_news else "No recent general business news available."


    prompt = f"""
    You are an expert AI business analyst. Generate a professional, in-depth, and comprehensive business feasibility report for the proposed idea, incorporating the provided real-world data.

    Objective: Assess the feasibility and potential of a business idea within a specific industry and market.

    Target Audience: Investors, stakeholders, and executive leadership. The tone should be formal, analytical, and data-driven.

    Business Idea Details:
    - Industry Focus: {industry} (Analyzed as: {normalized_industry})
    - Target Market: {target_market}
    - Business Goal/Concept: {goal}
    - Estimated Budget: {budget}

    Market Data & Insights:
    - Industry Growth & Static Insights: {static_insight}
    - Google Trends (Search Interest): {trend_data_summary}
    - ETF Market Behavior ({normalized_industry} sector proxy): {market_summary}
    - Investopedia Expert Insight: {investopedia_insight}
    - Hypothetical Statista-style Summary: {statista_insight}
    - Recent Google News Headlines for {industry}:
    {google_news_section}
    - Latest General Business News Headlines:
    {latest_business_news_section}
    - Real-Time Community & Market Sentiment (Reddit/LinkedIn context):
    {rag_context}

    Report Structure:

    1. Executive Summary:
       - Briefly summarize the business idea.
       - Provide a high-level assessment of its feasibility based on the analysis.
       - State the key recommendation.

    2. Market Opportunity Analysis:
       - Detail the identified market gap and potential.
       - Incorporate insights from Industry Growth, Google Trends, and ETF Market Behavior.
       - Discuss the Target Market size and characteristics.
       - Reference Investopedia and Statista insights.

    3. Competitive Landscape:
       - Identify and analyze at least two direct competitors and two indirect alternatives.
       - Discuss their strengths, weaknesses, and market position.
       - Explain the proposed business's unique selling proposition (USP) and competitive advantage.

    4. SWOT Analysis:
       - Present a comprehensive analysis of the Strengths, Weaknesses, Opportunities, and Threats related to the business idea within the current market context. Use bullet points under each heading.

    5. Financial Considerations & Projections (3-6 month outlook):
       - Discuss the estimated budget and its allocation across key areas (e.g., development, marketing, operations).
       - Provide realistic, *estimated* financial projections for the initial 3-6 months, including:
         - Estimated startup costs.
         - Projected revenue streams (explain assumptions).
         - Estimated operating expenses.
         - Projected burn rate.
         - *Estimate* the time or traction needed to reach breakeven, based on assumptions.
       - Acknowledge limitations due to estimated nature.

    6. Go-to-Market Strategy & Tactical Recommendations:
       - Propose a phased approach (e.g., MVP launch).
       - Recommend key marketing and customer acquisition strategies for the Target Market.
       - Suggest strategies for capital efficiency given the budget.
       - Incorporate insights from Google Trends and real-time news/Reddit data regarding market sentiment or popular discussion points.

    7. Pre-launch Readiness & Operational Plan:
       - Briefly outline key operational aspects (e.g., required technology stack, potential team structure/hiring needs for MVP).
       - Discuss initial operational workflows.

    8. Risk Assessment & Mitigation:
       - Elaborate on the Key Challenges/Risks identified in the Executive Summary.
       - For each risk, suggest potential mitigation strategies.

    9. Final Feasibility Assessment & Next Steps:
       - Reiterate the overall feasibility assessment, summarizing the key findings supporting this conclusion.
       - Outline concrete next steps required to move forward (e.g., detailed business plan, prototype development, seed funding).
       - Include considerations for compliance and potential fallback paths.

    Formatting:
    - Use clear headings and subheadings.
    - Use bullet points for lists (e.g., SWOT, recommendations).
    - Ensure a professional and formal business tone throughout.
    - Where numerical estimates are provided (especially in Financial Considerations), clearly state that these are *estimates* based on the limited information available.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Using a potentially newer model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6, # Slightly lower temperature for a more focused report
            max_tokens=1800 # Increased max tokens to allow for a more detailed report structure
        )
        return response.choices[0].message.content.strip()
    except OpenAIError as e:
        return f"\u274c Error generating full report: {e}"

# (All other functions like trends chart, bar chart, Streamlit UI remain unchanged — but now this code will return actual ETF + growth rate results if the CSV and input are aligned.)
# 🔹 Step 9: Export Report to PDF (Formatted)
def export_to_pdf(report_content, output_file):
    """Exports the report content to a PDF file."""
    # Simple HTML formatting; advanced formatting would require more complex HTML/CSS
    formatted_content = report_content.replace("\n", "<br>")
    html_content = f"""
    <html>
    <head><style>body {{ font-family: Arial, sans-serif; padding: 30px; }} h1, h2 {{ color: #1f4e79; }} .report-block {{ background-color: #0e1117; color: #f0f2f6; padding: 1.5em; border-radius: 12px; border: 1px solid #444; }} img {{ max-width: 100%; height: auto; margin-top: 20px; }}</style></head>
    <body>
        <h1>AI-Generated Business Feasibility Report</h1>
        <div>{formatted_content}</div>
        </body>
    </html>"""
    try:
        with open(output_file, "w+b") as f:
            pisa_status = pisa.CreatePDF(src=html_content, dest=f)
        return pisa_status.err
    except Exception as e:
         st.error(f"Error exporting to PDF: {e}")
         return True # Indicate an error occurred

# 🔹 Step 10: Charts
def plot_trends_chart(trend_df, industry):
    """Plots Google Trends data."""
    if trend_df is None or trend_df.empty:
        st.warning("No Google Trends data to plot.")
        return None # Return None if no plot is generated
    try:
        # Ensure the column name matches the one used in get_google_trends
        trends_term = normalize_industry_term(industry)
        if trends_term not in trend_df.columns:
             # Fallback to the first column if the specific term is not found (might happen with fuzzy matching)
             keyword_col = trend_df.columns[0]
        else:
             keyword_col = trends_term

        fig, ax = plt.subplots(figsize=(10, 3))
        trend_df[keyword_col].plot(ax=ax, title=f"Google Trends for '{keyword_col}' (12 months)")
        ax.set_ylabel("Search Interest")
        ax.set_xlabel("Date")
        st.pyplot(fig)
        # Save the figure to include in PDF
        fig_path = "trend_chart.png"
        fig.savefig(fig_path)
        plt.close(fig) # Close the figure to free memory
        return fig_path # Return the path to the saved figure
    except Exception as e:
        st.error(f"Error generating Google Trends chart: {e}")
        return None


def plot_etf_price_trend(hist, industry):
    """Plots ETF price trend data."""
    if hist is None or hist.empty:
        st.warning("No ETF price data available to plot.")
        return None # Return None if no plot is generated
    try:
        fig, ax = plt.subplots(figsize=(10, 3))
        hist["Close"].plot(ax=ax, title=f"{industry.capitalize()} Sector - ETF Price Trend (1 Year)")
        ax.set_ylabel("Price (USD)")
        st.pyplot(fig)
         # Save the figure to include in PDF
        fig_path = "etf_chart.png"
        fig.savefig(fig_path)
        plt.close(fig) # Close the figure to free memory
        return fig_path # Return the path to the saved figure
    except Exception as e:
        st.error(f"Error generating ETF price chart: {e}")
        return None


def plot_growth_bar(df):
    """Plots a bar chart of top industries by growth rate."""
    if df is None or df.empty or 'GrowthRate' not in df.columns or 'Industry' not in df.columns:
        st.warning("Industry growth data not available or incorrectly formatted.")
        return None # Return None if no plot is generated
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        # Ensure GrowthRate is numeric before sorting
        df['GrowthRate'] = pd.to_numeric(df['GrowthRate'], errors='coerce')
        top_df = df.dropna(subset=['GrowthRate']).sort_values(by="GrowthRate", ascending=False).head(10)

        if top_df.empty:
             st.warning("No valid growth rate data to plot.")
             return None

        ax.barh(top_df["Industry"], top_df["GrowthRate"], color='skyblue')
        ax.set_title("Top 10 Fastest-Growing Industries (Based on Available Data)")
        ax.set_xlabel("Growth Rate (%)")
        ax.invert_yaxis() # Highest growth at the top
        plt.tight_layout() # Adjust layout to prevent labels overlapping
        st.pyplot(fig)
         # Save the figure to include in PDF
        fig_path = "growth_chart.png"
        fig.savefig(fig_path)
        plt.close(fig) # Close the figure to free memory
        return fig_path # Return the path to the saved figure
    except Exception as e:
        st.error(f"Error generating growth bar chart: {e}")
        return None


# stream lit
import streamlit as st
import matplotlib.pyplot as plt
from xhtml2pdf import pisa

st.set_page_config(page_title="AI Business Report Generator", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "Home"
if "generated" not in st.session_state:
    st.session_state.generated = False


page = st.session_state.page

# JavaScript and CSS for button styling and sound
st.markdown("""
    <script>
        function playSound() {
            var audio = new Audio("https://www.soundjay.com/buttons/sounds/button-29.mp3");
            audio.play();
        }
    </script>
    <style>
        div.stButton > button:first-child {
            height: 3em;
            font-weight: bold;
            border-radius: 8px;
            border: none;
            background-color: #dc3545;
            color: white;
            transition: all 0.3s ease;
        }
        div.stButton > button:first-child.enabled {
            background-color: #28a745 !important;
        }
        .report-block {
            background-color: #0e1117;
            color: #f0f2f6;
            padding: 1.5em;
            border-radius: 12px;
            border: 1px solid #444;
            white-space: pre-wrap; /* Preserve line breaks */
            word-wrap: break-word; /* Break long words */
        }
         .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #0e1117;
            border-radius: 4px 4px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
            margin: 0;
            width: fit-content;
            min-width: none;
        }

        .stTabs [aria-selected="true"] {
            background-color: #1f4e79;
            color: white;
        }
         h1, h2, h3 {
            color: #1f4e79;
         }
    </style>
""", unsafe_allow_html=True)


# Home Page
if page == "Home":
    st.markdown("""
        <h1 style='text-align: center; color: #1f4e79;'>AI-Powered Business Report Generator</h1>
        <h4 style='text-align: center; color: #ccc;'>Generate investor-ready business reports using GPT-4o and real-time market data.</h4>
    """, unsafe_allow_html=True)

    with st.form("report_form"):
        st.markdown("### Enter Business Details")
        industry = st.text_input("Industry", placeholder="e.g. Fintech, Renewable Energy, E-commerce")
        target_market = st.text_input("Target Market", placeholder="e.g. Small businesses in Europe, Gen Z in North America")
        goal = st.text_area("Business Goal / Concept", placeholder="Describe your product or service idea and its primary goal.")
        budget = st.text_input("Estimated Budget", placeholder="e.g. $10,000, £50,000, €100,000 (specify currency if possible)")
        report_type = st.selectbox("Report Type", ["Summary", "Full"])

        form_complete = all([industry, target_market, goal, budget])
        submit = st.form_submit_button("Generate Report", type="primary", disabled=not form_complete)

        # Apply the 'enabled' class based on form completion using JavaScript
        st.markdown(f"""
            <script>
                const submitButton = window.parent.document.querySelector('button[type="submit"]');
                if (submitButton) {{
                    if ({str(form_complete).lower()}) {{
                        submitButton.classList.add('enabled');
                    }} else {{
                        submitButton.classList.remove('enabled');
                    }}
                }}
            </script>
        """, unsafe_allow_html=True)


        if submit and form_complete:
            st.session_state.generated = True
            st.session_state.industry = industry
            st.session_state.target_market = target_market
            st.session_state.goal = goal
            st.session_state.budget = budget
            st.session_state.report_type = report_type
            st.session_state.page = "Generated Report"
            st.components.v1.html("<script>playSound();</script>", height=0)
            st.rerun()
        elif submit and not form_complete:
             st.warning("Please fill out all fields to generate the report.")


# Generated Report Page
elif page == "Generated Report" and st.session_state.get("generated"):
    industry = st.session_state.industry
    target_market = st.session_state.target_market
    goal = st.session_state.goal
    budget = st.session_state.budget
    report_type = st.session_state.report_type

    st.markdown("<h2 style='color:#1f4e79;'>Generating Business Report...</h2>", unsafe_allow_html=True)

    # Use a spinner while generating the report
    with st.spinner("Analyzing data and generating report..."):
        if report_type == "Summary":
            result = get_business_feasibility_summary(industry, target_market, goal, budget)
        else:
            result = generate_full_report(industry, target_market, goal, budget)

    st.markdown("<h2 style='color:#1f4e79;'>Business Report</h2>", unsafe_allow_html=True)
    st.markdown(f"<div class='report-block'>{result.replace(chr(10), '<br>').replace(chr(13), '')}</div>", unsafe_allow_html=True) # Handle different newlines

    # Visual Market Insights Tabs
    st.markdown("---")
    st.markdown("<h3 style='color:#1f4e79;'>Visual Market Insights</h3>", unsafe_allow_html=True)
    tabs = st.tabs(["📈 Google Trends", "💹 ETF Price Trend", "📊 Industry Growth", "💬 Community Trends"])

    chart_paths = {}

    # Tab 1: Google Trends
    with tabs[0]:
        trend_summary, trend_df = get_google_trends(industry)
        st.markdown(f"*{trend_summary}*")
        trends_chart_path = plot_trends_chart(trend_df, industry)
        if trends_chart_path:
             chart_paths['trends'] = trends_chart_path


    # Tab 2: ETF Market Behavior
    with tabs[1]:
        market_summary, hist = get_industry_market_summary(industry)
        st.markdown(f"*{market_summary}*")
        etf_chart_path = plot_etf_price_trend(hist, industry)
        if etf_chart_path:
             chart_paths['etf'] = etf_chart_path


    # Tab 3: Industry Growth Rate
    with tabs[2]:
        static_insight, growth_df = get_market_insights(industry) # Re-fetch just for the plot
        st.markdown(f"*{static_insight}*")
        growth_chart_path = plot_growth_bar(growth_df)
        if growth_chart_path:
             chart_paths['growth'] = growth_chart_path


    # Tab 4: Community Trends (Displaying RAG context used)
    with tabs[3]:
         st.markdown("Insights from real-time discussions (Reddit, etc.) that informed the report.")
         normalized_industry = normalize_industry_term(industry)
         reddit_posts = get_reddit_posts(normalized_industry)
         rag_texts = [f"Reddit Post: {post}" for post in reddit_posts]
         rag_context_display = "No relevant community discussions found."
         if rag_texts:
              try:
                  reddit_db = build_vector_db_from_texts(rag_texts)
                  relevant_docs = retrieve_relevant_docs(reddit_db, f"{industry} market trends and challenges")
                  rag_context_display = "\n".join([doc.page_content for doc in relevant_docs])
                  if not rag_context_display.strip():
                       rag_context_display = "No highly relevant community discussions found."
              except Exception as e:
                   rag_context_display = f"Error retrieving community trends: {e}"

         st.text_area("Relevant Community Insights (from Reddit/etc.)", rag_context_display, height=300)


    st.markdown("---")
    st.markdown("<h3 style='color:#1f4e79;'>Download Report</h3>", unsafe_allow_html=True)
    st.markdown("Click the button below to download your report as a PDF.")
    st.markdown("Note: Charts included in the PDF are based on the visualizations above.")
    st.markdown("**Disclaimer**: This report is for informational purposes only and should not be considered financial advice.")

    # Export to PDF with charts
    def export_report_with_charts_to_pdf(report_text, file_name, chart_paths):
        formatted_text = report_text.replace("\n", "<br>")
        html_content = f"""
        <html>
        <head><style>body {{ font-family: Arial, sans-serif; padding: 30px; }} h1, h2 {{ color: #1f4e79; }} .report-block {{ background-color: #0e1117; color: #f0f2f6; padding: 1.5em; border-radius: 12px; border: 1px solid #444; }} img {{ max-width: 100%; height: auto; margin-top: 20px; display: block; margin-left: auto; margin-right: auto; }}</style></head>
        <body>
            <h1>AI-Generated Business Feasibility Report</h1>
            <div>{formatted_text}</div>
        """
        # Add images if they exist
        if 'trends' in chart_paths and os.path.exists(chart_paths['trends']):
            html_content += f'<img src="data:image/png;base64,{get_image_as_base64(chart_paths["trends"])}"><br>'
        if 'etf' in chart_paths and os.path.exists(chart_paths['etf']):
             html_content += f'<img src="data:image/png;base64,{get_image_as_base64(chart_paths["etf"])}"><br>'
        if 'growth' in chart_paths and os.path.exists(chart_paths['growth']):
            html_content += f'<img src="data:image/png;base64,{get_image_as_base64(chart_paths["growth"])}"><br>'

        html_content += "</body></html>"

        try:
            with open(file_name, "w+b") as f:
                pisa_status = pisa.CreatePDF(src=html_content, dest=f)
            return pisa_status.err
        except Exception as e:
             st.error(f"Error exporting to PDF with charts: {e}")
             return True


    # Helper function to embed images in PDF
    import base64
    def get_image_as_base64(file_path):
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()


    pdf_file_name = "business_report.pdf"
    # Generate the PDF (this saves the file locally on the server/container)
    export_report_with_charts_to_pdf(result, pdf_file_name, chart_paths)

    # Provide the download button
    if os.path.exists(pdf_file_name):
        with open(pdf_file_name, "rb") as f:
            st.download_button("Download PDF with Charts", f, pdf_file_name, mime="application/pdf")
    else:
        st.error("Could not generate PDF for download.")


    # Clean up generated chart images
    for chart_path in chart_paths.values():
        if os.path.exists(chart_path):
            os.remove(chart_path)


    # Back button
    if st.button("Back to Home"):
        # Clean up session state related to the generated report
        for key in ["generated", "industry", "target_market", "goal", "budget", "report_type"]:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.page = "Home"
        st.rerun()

# Fallback
else:
    st.info("Please generate a report from the Home page first.")
    st.session_state.page = "Home"
    st.rerun()