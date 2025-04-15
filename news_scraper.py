import re
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import pandas as pd

# -------- Sector Keywords (fallback) --------
sector_keywords = {
    "Automotive": ["car", "vehicle", "automotive", "EV", "battery", "motors"],
    "Technology": ["AI", "tech", "software", "hardware", "cloud", "data", "SaaS", "machine learning", "cybersecurity"],
    "Finance": ["bank", "investment", "fintech", "stocks", "IPO", "merger", "fund", "equity", "crypto"],
    "Retail": ["retail", "ecommerce", "store", "shopping", "consumer goods", "mall"],
    "Energy": ["energy", "oil", "gas", "solar", "renewable", "electric", "power"],
    "Healthcare": ["health", "pharma", "biotech", "hospital", "vaccine", "clinic", "medical"],
    "Logistics / Transportation": ["shipping", "logistics", "freight", "transport", "delivery", "airlines", "railway"],
    "Media & Entertainment": ["media", "ad", "broadcast", "streaming", "tv", "film", "movie", "music", "podcast"],
    "Real Estate": ["real estate", "property", "housing", "rental", "mortgage", "land", "construction"],
    "Aerospace & Defense": ["aircraft", "aerospace", "defense", "military", "satellite", "missile"],
    "Telecom": ["telecom", "mobile", "5G", "network", "broadband", "wireless"],
    "Education": ["education", "school", "university", "edtech", "student", "course", "degree"],
    "Food & Beverage": ["food", "restaurant", "beverage", "grocery", "dining", "alcohol", "snack"],
    "Manufacturing / Industrial": ["factory", "manufacturing", "industry", "supply chain", "assembly"],
    "Travel & Hospitality": ["hotel", "travel", "tourism", "resort", "airbnb", "booking"],
    "Government / Policy": ["regulation", "policy", "government", "tax", "law", "minister", "sanction"],
    "Crypto / Web3": ["crypto", "bitcoin", "blockchain", "web3", "NFT", "ethereum"]
}

# -------- Business News Sites --------
business_sites = {
    "Bloomberg": "https://www.bloomberg.com/business",
    "Business Insider": "https://www.businessinsider.com/business",
    "CNBC": "https://www.cnbc.com/business/",
    "Financial Times": "https://www.ft.com/business",
    "Forbes": "https://www.forbes.com/business/",
    "Inc": "https://www.inc.com/"
}

# -------- Load and Clean Company-Sector CSV --------
try:
    df_raw = pd.read_csv("company_with_sectors.csv")
    df_raw['company_name'] = df_raw['company_name'].str.lower()
    df_raw['clean_sector'] = df_raw['sector'].str.extract(r"'([^']+)'", expand=False).fillna("Other")
    company_sector_df = df_raw[['company_name', 'clean_sector']]
except Exception as e:
    print(f"⚠️ Could not load or parse company_with_sectors.csv: {e}")
    company_sector_df = pd.DataFrame(columns=["company_name", "clean_sector"])

# -------- Helper Functions --------
def infer_sector(text):
    text_lower = text.lower()

    # Company name match from CSV
    for company in company_sector_df['company_name']:
        if company in text_lower:
            match = company_sector_df[company_sector_df['company_name'] == company]
            return match['clean_sector'].values[0]

    # Fallback: keyword-based sector inference
    for sector, keywords in sector_keywords.items():
        if any(k.lower() in text_lower for k in keywords):
            return sector

    return "Other"

def get_article_links(url, limit=5):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        links = set()
        for tag in soup.find_all('a', href=True):
            href = tag['href']
            if href.startswith("http"):
                links.add(href)
        return list(links)[:limit]
    except Exception:
        return []

def extract_article_data(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        title = article.title
        summary = article.summary
        full_text = article.text
        sector = infer_sector(full_text + " " + title)
        business_guess = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', title)[:2]
        return {
            "Sector": sector,
            "Business": ', '.join(business_guess) if business_guess else "Unknown",
            "News": f"{title}: {summary}"
        }
    except:
        return None

# -------- Exportable Function --------
def scrape_latest_business_news():
    all_articles = []
    for source_name, homepage_url in business_sites.items():
        urls = get_article_links(homepage_url, limit=3)
        for link in urls:
            result = extract_article_data(link)
            if result:
                all_articles.append(result)
    return all_articles
