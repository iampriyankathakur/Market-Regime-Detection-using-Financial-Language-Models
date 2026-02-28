from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class FinBERTEmbedder:

    def __init__(self, device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModel.from_pretrained("ProsusAI/finbert").to(device)

    def embed(self, texts):
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls[0])
        return np.array(embeddings)
