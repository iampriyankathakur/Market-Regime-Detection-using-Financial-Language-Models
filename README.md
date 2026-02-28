# Transformer-Based Market Regime Detection using Financial Narrative Embeddings

## 📌 Overview

This project implements a financial market regime detection framework using domain-specific transformer embeddings derived from macroeconomic news narratives.

Instead of directly predicting asset prices, the system models latent macro regimes inferred from financial text and uses them to drive a rule-based portfolio allocation strategy.

The framework integrates:

- FinBERT-based financial embeddings
- Unsupervised clustering for regime discovery
- Temporal sequence modeling
- Regime-driven portfolio backtesting


## 🧠 Methodology

### 1. Financial Text Embedding
Financial news is embedded using a domain-specific transformer model (FinBERT).

### 2. Regime Discovery
Dimensionality reduction (UMAP) followed by clustering (HDBSCAN) is used to identify latent macroeconomic narrative regimes.

### 3. Regime Transition Modeling
A temporal classifier is trained to predict next-period regime labels.

### 4. Strategy Layer
Portfolio allocation is determined based on predicted regime:

- Risk-On → Equity Exposure
- Risk-Off → Cash Allocation

Strategy performance is evaluated against Buy & Hold.

## 📊 Evaluation

- Regime classification accuracy
- Regime stability analysis
- Strategy cumulative returns
- Sharpe ratio comparison

## 🛠 Technologies

- Python
- PyTorch
- Transformers (FinBERT)
- UMAP
- HDBSCAN
- yFinance
- Scikit-learn
- Pandas

## ▶️ Run

```bash
pip install -r requirements.txt
python main.py
