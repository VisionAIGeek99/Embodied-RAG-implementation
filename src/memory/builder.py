# src/memory/builder.py

import numpy as np
from src.memory.node import Node
from src.memory.summarizer import summarize_cluster
from src.memory.similarity import (
    compute_spatial_similarity,
    compute_semantic_similarity,
    compute_hybrid_similarity,
)
from src.memory.clustering import complete_linkage_clustering


def build_semantic_forest(
    positions,
    embeddings,
    captions,
    images,
    quaternions,
    theta_spatial=10.0,
    alpha=0.3,
    cluster_threshold=0.4,
):
    """
    Build a hierarchical semantic forest structure using Node class.

    Inputs:
        positions   : (N, 3) or (N, 2) numpy array
        embeddings  : (N, D) numpy array
        captions    : list[str], length N
        images      : list[str], length N (각 노드 이미지 경로)
        quaternions : (N, 4) numpy array, [x,y,z,w] (카메라 pose)

    Returns:
        forest_dict: { "root": node_id, "nodes": {node_id: {...}, ...} }
    """
    N = len(positions)
    assert len(captions) == N
    assert len(images) == N
    assert len(quaternions) == N

    nodes = {}

    # ---------------------------------------------------------
    # 1) Leaf Nodes 생성 (L0)  — pose + image 포함
    # ---------------------------------------------------------
    for i in range(N):
        node_id = f"L0_{i}"

        nodes[node_id] = Node(
            node_id=node_id,
            level=0,
            node_type="leaf",
            children=[],
            parent=None,
            summary=captions[i],              # 나중에 summary 대신 raw caption만 쓰고 싶으면 바꿔도 됨
            embedding=embeddings[i].tolist(),
            position=positions[i].tolist(),
            quaternion=quaternions[i].tolist(),
            image=images[i],
            raw_caption=captions[i],
        )

    # ---------------------------------------------------------
    # 2) Spatial + Semantic + Hybrid similarity 계산
    # ---------------------------------------------------------
    S_sp = compute_spatial_similarity(positions, theta=theta_spatial)
    S_sem = compute_semantic_similarity(embeddings)
    S_hybrid = compute_hybrid_similarity(S_sp, S_sem, alpha=alpha)

    print("S_spatial stats:", np.min(S_sp), np.max(S_sp), np.mean(S_sp))
    print("S_semantic stats:", np.min(S_sem), np.max(S_sem), np.mean(S_sem))
    print("S_hybrid stats:", np.min(S_hybrid), np.max(S_hybrid), np.mean(S_hybrid))

    # ---------------------------------------------------------
    # 3) 1단계 CLINK clustering → L1 노드 생성
    # ---------------------------------------------------------
    clusters = complete_linkage_clustering(S_hybrid, threshold=cluster_threshold)

    level = 1
    for idx, cluster in enumerate(clusters):
        node_id = f"L1_{idx}"
        child_ids = [f"L0_{i}" for i in cluster]

        # summary 생성 (LLM)
        area_captions = [nodes[c].raw_caption for c in child_ids]
        summary = summarize_cluster(area_captions)

        # centroid 계산
        centroid_pos = positions[cluster].mean(axis=0).tolist()
        centroid_emb = embeddings[cluster].mean(axis=0).tolist()

        nodes[node_id] = Node(
            node_id=node_id,
            level=level,
            node_type="area",
            children=child_ids,
            parent=None,
            summary=summary,
            embedding=centroid_emb,
            position=centroid_pos,   # area는 centroid position
        )

        # parent 연결
        for cid in child_ids:
            nodes[cid].parent = node_id

    # ---------------------------------------------------------
    # 4) Recursive merge until single root  (area 노드들만 합침)
    # ---------------------------------------------------------
    current_level_nodes = [nid for nid in nodes if nodes[nid].level == 1]
    level = 2

    while len(current_level_nodes) > 1:
        embs = np.array([nodes[nid].embedding for nid in current_level_nodes])
        S = np.dot(embs, embs.T) / (
            np.linalg.norm(embs, axis=1, keepdims=True)
            * np.linalg.norm(embs, axis=1).T
            + 1e-8
        )
        np.fill_diagonal(S, -1)

        i, j = np.unravel_index(np.argmax(S), S.shape)
        nid1 = current_level_nodes[i]
        nid2 = current_level_nodes[j]

        # 자식 합치기
        merged_children = nodes[nid1].children + nodes[nid2].children

        # summary 합치기 (상위 area 요약)
        merged_summary = summarize_cluster(
            [nodes[nid1].summary, nodes[nid2].summary]
        )

        # embedding merge
        merged_emb = (
            np.array(nodes[nid1].embedding) + np.array(nodes[nid2].embedding)
        ) / 2
        merged_emb = merged_emb.tolist()

        # position merge (centroid)
        merged_pos = (
            np.array(nodes[nid1].position) + np.array(nodes[nid2].position)
        ) / 2
        merged_pos = merged_pos.tolist()

        new_id = f"L{level}_{len([n for n in nodes if nodes[n].level == level])}"

        nodes[new_id] = Node(
            node_id=new_id,
            level=level,
            node_type="area",
            children=[nid1, nid2],
            parent=None,
            summary=merged_summary,
            embedding=merged_emb,
            position=merged_pos,
        )

        nodes[nid1].parent = new_id
        nodes[nid2].parent = new_id

        remain = [n for k, n in enumerate(current_level_nodes) if k not in (i, j)]
        remain.append(new_id)
        current_level_nodes = remain
        level += 1

    root = current_level_nodes[0]

    forest_dict = {
        "root": root,
        "nodes": {nid: nodes[nid].to_dict() for nid in nodes},
    }

    return forest_dict
