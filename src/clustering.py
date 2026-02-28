import umap
import hdbscan

def discover_regimes(embeddings):

    reducer = umap.UMAP(n_components=5)
    reduced = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=20)
    labels = clusterer.fit_predict(reduced)

    return labels
