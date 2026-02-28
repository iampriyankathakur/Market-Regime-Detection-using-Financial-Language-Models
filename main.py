import torch
import numpy as np

from src.data_loader import load_market_data, load_mock_news
from src.embedding_engine import FinBERTEmbedder
from src.clustering import discover_regimes
from src.regime_model import RegimeClassifier
from src.backtester import backtest

# Load data
market_data = load_market_data()
news = load_mock_news()

# Embed text
embedder = FinBERTEmbedder()
embeddings = embedder.embed(news)

# Discover regimes
labels = discover_regimes(embeddings)

# Align length
labels = labels[:len(market_data)]
returns = market_data["Returns"].values[:len(labels)]

# Convert to tensor
X = torch.tensor(embeddings[:len(labels)]).float()
y = torch.tensor(labels).long()

# Train simple classifier
model = RegimeClassifier(X.shape[1], len(set(labels)))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# Predict regimes
with torch.no_grad():
    predictions = model(X).argmax(dim=1).numpy()

# Backtest
backtest(predictions, returns)
