from newsapi import NewsApiClient
import pandas as pd

def fetch_macro_news(api_key, query="economy OR inflation OR recession"):
    newsapi = NewsApiClient(api_key=api_key)
    articles = newsapi.get_everything(
        q=query,
        language="en",
        sort_by="publishedAt",
        page_size=100
    )
    
    texts = []
    for article in articles["articles"]:
        title = article["title"] or ""
        desc = article["description"] or ""
        texts.append(title + " " + desc)

    return texts
