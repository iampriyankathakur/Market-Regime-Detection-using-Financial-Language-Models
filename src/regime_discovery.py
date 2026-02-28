import umap
import hdbscan

def discover_regimes(embeddings):
    reducer = umap.UMAP(n_components=5, random_state=42)
    reduced = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=50)
    labels = clusterer.fit_predict(reduced)

    return labels
