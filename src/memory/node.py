# src/memory/node.py

class Node:
    def __init__(
        self,
        node_id,
        level,
        node_type,
        children=None,
        parent=None,
        summary=None,
        embedding=None,
        position=None,
        quaternion=None,
        image=None,
        raw_caption=None,
    ):
        self.id = node_id
        self.level = level          # 0,1,2,...
        self.type = node_type       # "leaf" or "area"
        self.children = children or []
        self.parent = parent

        self.summary = summary
        self.embedding = embedding

        # geometry / sensor
        self.position = position        # leaf: 실제 pose, area: centroid
        self.quaternion = quaternion    # leaf 에만 사용
        self.image = image              # leaf 에만 사용

        # 원본 캡션(leaf에서만 사용)
        self.raw_caption = raw_caption

    def is_leaf(self):
        return self.type == "leaf"

    def to_dict(self):
        data = {
            "id": self.id,
            "level": self.level,
            "type": self.type,
            "children": self.children,
            "parent": self.parent,
            "summary": self.summary,
            "embedding": self.embedding,
            "position": self.position,
        }

        if self.raw_caption is not None:
            data["raw_caption"] = self.raw_caption

        # leaf 전용 필드
        if self.type == "leaf":
            data["image"] = self.image
            data["quaternion"] = self.quaternion

        return data
