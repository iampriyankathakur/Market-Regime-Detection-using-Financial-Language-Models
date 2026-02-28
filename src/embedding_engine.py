from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class FinBERTEmbedder:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModel.from_pretrained("ProsusAI/finbert")

    def embed(self, texts):
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
            embeddings.append(cls_embedding[0])
        return np.array(embeddings)
