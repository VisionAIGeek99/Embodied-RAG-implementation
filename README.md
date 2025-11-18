# Embodied-RAG unofficial implementation

## Overview
This is an unofficial implementation of **Embodied-RAG**, a non-parametric memory
system for embodied agents.  

## Paper 
[Embodied-RAG: General Non-parametric Embodied Memory for Retrieval and Generation](https://arxiv.org/abs/2409.18313)

## Scripts usage

```
# 1. setup_dataset.py 
uv run python -m scripts.setup_dataset \
    --name coex_1f \
    --raw_path /disks/ssd1/kmw2622/dataset/coex_1F_release_mapping/1F/release/mapping \
    --target_cam 40027089_00 \
    --max_nodes 50

# 2. extract_viewpoints.py 
uv run python -m scripts.extract_viewpoints

# 3. caption_nodes.py
uv run python -m scripts.caption_nodes

# 4. build_edges.py
uv run python -m scripts.build_edges
cv
# 5. build_graph.py
uv run python -m scripts.build_graph

# 6. viz_graph.py
uv run python -m scripts.viz_graph

```

## Installation
```
git clone https://github.com/VisionAIGeek99/Embodied-RAG-implementation.git
cd Embodied-RAG-implementation
uv sync
```


## Dataset
[NAVER LABS Large-scale localization datasets in crowded indoor spaces](https://europe.naverlabs.com/blog/first-of-a-kind-large-scale-localization-datasets-in-crowded-indoor-spaces/)
