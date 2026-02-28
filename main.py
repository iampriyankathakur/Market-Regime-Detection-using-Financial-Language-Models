import torch
import numpy as np

from src.config import SYMBOL, START_DATE
from src.data_loader import load_market_data, generate_macro_news
from src.embedding_engine import FinBERTEmbedder
from src.regime_discovery import discover_regimes
from src.regime_classifier import RegimeClassifier
from src.backtester import backtest
from src.utils import clean_labels

# 1. Load Data
market_data = load_market_data(SYMBOL, START_DATE)
news_texts = generate_macro_news(market_data)

# 2. Embed News
embedder = FinBERTEmbedder()
embeddings = embedder.embed(news_texts)

# 3. Discover Regimes
raw_labels = discover_regimes(embeddings)
labels = clean_labels(raw_labels)

# Align data
labels = labels[:len(market_data)]
returns = market_data["Returns"].values[:len(labels)]

# 4. Train Classifier
X = torch.tensor(embeddings[:len(labels)]).float()
y = torch.tensor(labels).long()

model = RegimeClassifier(X.shape[1], len(set(labels)))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(25):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 5. Predict Regimes
with torch.no_grad():
    predictions = model(X).argmax(dim=1).numpy()

# 6. Backtest
backtest(predictions, returns)
