import numpy as np

def complete_linkage_clustering(sim_matrix, threshold=0.3):
    N = len(sim_matrix)
    clusters = [[i] for i in range(N)]

    while True:
        max_sim = -1
        merge_pair = None

        # find closest cluster pair
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                sim = _cluster_similarity(sim_matrix, clusters[i], clusters[j])
                if sim > max_sim:
                    max_sim = sim
                    merge_pair = (i, j)

        if max_sim < threshold:
            break

        i, j = merge_pair
        clusters[i] += clusters[j]
        del clusters[j]

        if len(clusters) == 1:
            break

    return clusters


def _cluster_similarity(S, c1, c2):
    # complete-link: min pairwise similarity between clusters
    return min(S[i][j] for i in c1 for j in c2)
