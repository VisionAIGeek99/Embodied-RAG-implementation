import numpy as np

def compute_spatial_similarity(positions, theta=1.0, mode="euclidean"):
    """
    Compute spatial similarity using either Euclidean or Haversine distance.
    
    positions:
        shape (N, 2) or (N, 3)
        - Euclidean: (x, y) in meters
        - Haversine: (lat, lon) in degrees
    
    theta:
        similarity decay parameter
        S = exp(-distance / theta)
    
    mode: "euclidean" or "haversine"
    """

    if mode == "euclidean":
        # Take only x, y
        pos = positions[:, :2].astype(float)

        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)

        return np.exp(-dist / theta)

    elif mode == "haversine":
        # -----------------------------------------
        # Haversine distance (lat/lon)
        # -----------------------------------------
        R = 6371.0  # Earth radius in km

        lat = np.radians(positions[:, 0]).reshape(-1, 1)
        lon = np.radians(positions[:, 1]).reshape(-1, 1)

        dlat = lat - lat.T
        dlon = lon - lon.T

        a = (
            np.sin(dlat / 2)**2
            + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2)**2
        )
        c = 2 * np.arcsin(np.sqrt(a))

        dist = R * c  # km
        return np.exp(-dist / theta)

    else:
        raise ValueError("mode must be 'euclidean' or 'haversine'")



def compute_semantic_similarity(embeddings):
    """
    embeddings: numpy array of shape (N, D)
    returns:
        cosine similarity matrix (N, N)
    """

    # Normalize embeddings to unit vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)

    # Cosine similarity matrix = dot product of normalized vectors
    sim_matrix = np.dot(normalized, normalized.T)

    return sim_matrix

def compute_hybrid_similarity(spatial, semantic, alpha=0.5):
    """
    spatial: (N, N) numpy array - spatial similarity matrix
    semantic: (N, N) numpy array - semantic similarity matrix
    alpha: float (0~1), weight for semantic similarity
    
    returns:
        hybrid similarity matrix (N, N)
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    
    if spatial.shape != semantic.shape:
        raise ValueError("spatial and semantic matrices must have the same shape")

    hybrid = (1 - alpha) * spatial + alpha * semantic
    return hybrid
