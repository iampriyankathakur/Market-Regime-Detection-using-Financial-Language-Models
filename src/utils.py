def clean_labels(labels):
    unique = sorted(set(labels))
    mapping = {old:i for i, old in enumerate(unique)}
    return [mapping[l] for l in labels]
