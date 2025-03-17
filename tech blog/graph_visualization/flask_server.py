from flask import Flask, jsonify, request
import networkx as nx
import pickle
import json
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/graph": {"origins": "http://localhost:8000"}})



# ✅ 네트워크 그래프 G (이전에 생성한 G 활용)
with open("graph.pkl", "rb") as f:
    G = pickle.load(f)

# ✅ API: 전체 그래프 데이터 제공


@app.route('/')
def home():
    return """
    <h1>Flask Graph API</h1>
    <p>API is running! Try accessing <a href='/graph'>/graph</a> to see the graph data.</p>
    """

@app.route('/graph', methods=['GET'])
def get_graph():
    """ JSON 변환 시 `numpy.float32` 변환 가능하도록 설정 """
    min_weight = float(request.args.get("min_weight", 0))
    search_query = request.args.get("search", "").lower()

    nodes = [node for node in G.nodes() if search_query in node.lower()
             ] if search_query else list(G.nodes())

    edges = [
        {"source": u, "target": v, "weight": G[u][v].get("weight", 1)}
        for u, v in G.edges()
        if u in nodes and v in nodes and G[u][v].get("weight", 1) >= min_weight
    ]

    # 🔹 JSON 변환 시 `float32`를 처리할 수 있도록 `default=str` 사용
    return app.response_class(
        response=json.dumps({"nodes": [{"id": node} for node in nodes], "edges": edges},
                            default=lambda x: float(x) if isinstance(x, np.float32) else x),
        status=200,
        mimetype="application/json"
    )


# Flask 서버 실행
if __name__ == '__main__':
    app.run(debug=True)
