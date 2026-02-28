import torch
import numpy as np

from src.config import SYMBOL, START_DATE, NEWS_API_KEY
from src.data_loader import load_market_data
from src.news_ingestion import fetch_macro_news
from src.embedding_engine import FinBERTEmbedder
from src.regime_discovery import discover_regimes
from src.regime_classifier import RegimeClassifier
from src.walk_forward import walk_forward_split
from src.backtester import backtest
from src.metrics import evaluate
from src.explainability import shap_explain
from src.utils import clean_labels

# 1. Load Market Data
market_data = load_market_data(SYMBOL, START_DATE)

# 2. Fetch Real Macro News
news_texts = fetch_macro_news(NEWS_API_KEY)

# Align lengths
min_len = min(len(market_data), len(news_texts))
market_data = market_data.iloc[:min_len]
news_texts = news_texts[:min_len]

# 3. Embed Text
embedder = FinBERTEmbedder()
embeddings = embedder.embed(news_texts)

# 4. Discover Regimes
raw_labels = discover_regimes(embeddings)
labels = clean_labels(raw_labels)

returns = market_data["Returns"].values[:len(labels)]

X = torch.tensor(embeddings[:len(labels)]).float()
y = torch.tensor(labels).long()

# 5. Walk Forward Split
X_train, y_train, X_test, y_test = walk_forward_split(X, y)

# 6. Train
model = RegimeClassifier(X.shape[1], len(set(labels)))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 7. Evaluate
with torch.no_grad():
    test_preds = model(X_test).argmax(dim=1).numpy()

evaluate(y_test.numpy(), test_preds)

# 8. SHAP Explainability
shap_explain(model, X_test.numpy())

# 9. Backtest
backtest(test_preds, returns[-len(test_preds):])
