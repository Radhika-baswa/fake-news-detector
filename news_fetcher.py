# news_fetcher.py
import requests

API_KEY = "d0e18d41074313148248038943cd5b90" # Replace this with your real key

def fetch_latest_news():
    url = f"https://gnews.io/api/v4/top-headlines?lang=en&country=in&max=5&apikey={API_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        return [f"❌ Error: {response.status_code} - {response.json().get('errors', ['Unknown error'])[0]}"]

    data = response.json()
    articles = data.get('articles', [])
    if not articles:
        return ["⚠️ No articles found. Try again later."]
    
    # Return full articles (title + description)
    return [f"{a['title']} — {a['description']}" for a in articles]