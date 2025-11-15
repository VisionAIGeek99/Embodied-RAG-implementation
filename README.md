# Embodied-RAG unofficial implementation

## Paper title: Embodied-RAG: General Non-parametric Embodied Memory for Retrieval and Generation

# Scripts usage

```
# 1. setup_dataset.py 
uv run setup_dataset.py \
    --name coex_1f \
    --raw_path /disks/ssd1/kmw2622/dataset/coex_1F_release_mapping/1F/release/mapping \
    --target_cam 40027089_00 \
    --max_nodes 100

# 2. extract_viewpoints.py 
uv run extract_viewpoints.py 

# 3. caption_nodes.py
uv run python -m scripts.caption_nodes

# 4. build_edges.py
uv run python -m scripts.build_edges

# 5. build_graph.py
uv run python -m scripts.build_graph

# 6. viz_graph.py
uv run python -m scripts.viz_graph

```

