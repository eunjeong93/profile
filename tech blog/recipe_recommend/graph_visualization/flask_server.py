from flask import Flask, jsonify, request, render_template
import networkx as nx
import pickle
import json
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/graph": {"origins": "http://localhost:8000"}})

# ✅ 네트워크 그래프 로드
with open("graph.pkl", "rb") as f:
    G = pickle.load(f)


def get_nodes_within_k_hops(G, start_node, k=1):
    # ✅ 그래프 내 모든 노드를 소문자로 변환한 딕셔너리 생성
    graph_nodes = {node.lower(): node for node in G.nodes()}

    # ✅ 검색어를 소문자로 변환하여 그래프 내 노드와 비교
    start_node_lower = start_node.lower()

    if start_node_lower not in graph_nodes:
        raise ValueError(
            f"🚨 The node '{start_node}' is not in the graph. Available nodes: {list(graph_nodes.keys())[:10]}")

    # ✅ 원래 노드명을 찾기
    start_node = graph_nodes[start_node_lower]
    nodes_within_k = set([start_node])

    print(f"🔍 Found Node in Graph: {start_node}")

    for _ in range(k):
        new_nodes = set()
        for node in nodes_within_k:
            try:
                neighbors = list(G.neighbors(node))
                print(f"🕵️ Checking Neighbors for {node}: {neighbors}")
                new_nodes.update(neighbors)
            except KeyError:
                print(f"❌ KeyError: Node '{node}' not found in graph.")
                continue

        nodes_within_k.update(new_nodes)

    subG = G.subgraph(nodes_within_k)
    print(f"📡 Subgraph Nodes: {list(subG.nodes())[:10]}")
    return subG


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/graph', methods=['GET'])
def get_graph():
    """ JSON 변환 시 `numpy.float32` 변환 가능하도록 설정 """
    min_weight = float(request.args.get("min_weight", 0))
    min_stars = int(request.args.get("min_stars", 0))
    search_query = request.args.get("search", "").lower()
    filter_type = request.args.get("type", "")  # ✅ 노드 타입 필터링
    relation_filters = request.args.getlist("relation")  # ✅ relation 다중 선택 필터링
    k = int(request.args.get("k", 1))

    print(
        f"🔎 Received Filters - Weight: {min_weight}, Stars: {min_stars}, Search: {search_query}, Type: {filter_type}, Relations: {relation_filters}")

    subG = get_nodes_within_k_hops(G, search_query, k)

    # ✅ "All" 선택 시 모든 relation 허용
    all_relations = ["keyword", "category", "first", "second", "third"]
    if not relation_filters or "all" in relation_filters:
        relation_filters = all_relations  # ✅ 모든 relation 허용

    # ✅ 검색된 노드 필터링
    nodes = [node for node in subG.nodes() if search_query in node.lower()
             ] if search_query else list(G.nodes())

    # ✅ 노드 리스트 생성
    nodes = list(subG.nodes())

    # ✅ 존재하는 노드 목록을 집합으로 저장 (빠른 조회를 위해)
    valid_nodes = set(nodes)

    # ✅ 엣지 필터링 (weight, stars, relation 적용)
    edges = []
    for u, v in subG.edges():
        edge_data = subG[u][v]
        relation = edge_data.get("relation", "").lower()
        keys = list(edge_data.keys())

        if ('weight' in keys) & ('stars' in keys):
            if (
                edge_data.get("weight") >= min_weight and
                edge_data.get("stars") >= min_stars and
                relation in relation_filters  # ✅ relation 필터 적용
            ):
                edges.append({"source": u, "target": v, "weight": edge_data.get(
                    "weight"), "stars": edge_data.get("stars"), "relation": relation})
        elif 'weight' in keys:
            if (
                edge_data.get("weight") >= min_weight and
                relation in relation_filters  # ✅ relation 필터 적용
            ):
                edges.append({"source": u, "target": v, "weight": edge_data.get(
                    "weight"), "relation": relation})
        elif 'stars' in keys:
            if (
                edge_data.get("stars") >= min_stars and
                relation in relation_filters  # ✅ relation 필터 적용
            ):
                edges.append({"source": u, "target": v, "stars": edge_data.get(
                    "stars"), "relation": relation})
        else:
            if (
                relation in relation_filters  # ✅ relation 필터 적용
            ):
                edges.append({"source": u, "target": v, "relation": relation})

    # ✅ 특정 노드 타입 필터링 적용
    if filter_type != "":
        nodes = [node for node in nodes if subG.nodes[node].get(
            "type", "") == filter_type]
        edges = [edge for edge in edges if edge["source"]
                 in nodes and edge["target"] in nodes]

    print(f"📡 Filtered Data - Nodes: {len(nodes)}, Edges: {len(edges)}")

    return app.response_class(
        response=json.dumps(
            {
                "nodes": [{"id": node, "type": subG.nodes[node].get("type", "")} for node in nodes],
                "edges": edges
            },
            default=lambda x: float(x) if isinstance(x, np.float32) else x
        ),
        status=200,
        mimetype="application/json"
    )


if __name__ == "__main__":
    app.run(debug=True)
