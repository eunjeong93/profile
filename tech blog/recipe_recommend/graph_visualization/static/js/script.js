document.addEventListener("DOMContentLoaded", function () {
    console.log("✅ DOM 로딩 완료 - 초기값 설정 시작");

    let defaultSearch = "Creamy White Chili";
    let defaultMinWeight = 0.98;
    let defaultType = "";
    let defaultMinStars = 4;
    let defaultK = 1;
    let defaultRelations = ["keyword", "category", "first", "second", "third"];

    document.getElementById("searchBox").value = defaultSearch;
    document.getElementById("weightFilter").value = defaultMinWeight;
    document.getElementById("typeFilter").value = defaultType;
    document.getElementById("starsFilter").value = defaultMinStars;
    document.getElementById("kFilter").value = defaultK;

    document.querySelectorAll(".relation-checkbox").forEach(cb => {
        cb.checked = defaultRelations.includes(cb.value);
    });

    console.log("🔍 초기값 설정 완료 - updateGraph 실행");
    updateGraph();
});

// ✅ "All" 체크박스 클릭 시 모든 relation 선택/해제
document.getElementById("relationAll").addEventListener("change", function () {
    let checkboxes = document.querySelectorAll(".relation-checkbox");
    checkboxes.forEach(cb => cb.checked = this.checked);
});

function updateGraph() {
    let searchQuery = document.getElementById("searchBox").value;
    let minWeight = document.getElementById("weightFilter").value;
    let nodeType = document.getElementById("typeFilter").value;
    let minStars = document.getElementById("starsFilter").value;
    let k = document.getElementById("kFilter").value;

    let selectedRelations = [];
    document.querySelectorAll(".relation-checkbox:checked").forEach(cb => {
        selectedRelations.push(cb.value);
    });

    console.log("📡 API 요청 데이터:");
    console.log(`🔍 검색어: ${searchQuery}`);
    console.log(`⚖️ 최소 Weight: ${minWeight}`);
    console.log(`⭐ 최소 Stars: ${minStars}`);
    console.log(`🔄 Hop Depth (K): ${k}`);
    console.log(`📌 선택된 Node Type: ${nodeType}`);
    console.log(`🔗 선택된 Relations: ${selectedRelations}`);

    let apiUrl = `http://127.0.0.1:5000/graph?search=${searchQuery}&min_weight=${minWeight}&type=${nodeType}&min_stars=${minStars}&k=${k}&relation=${selectedRelations.join("&relation=")}`;

    console.log("📡 API 요청 URL:", apiUrl);

    d3.json(apiUrl).then(function (data) {
        console.log("✅ 데이터 받아옴:", data);

        console.log("📌 노드 데이터:", data.nodes);
        console.log("📌 엣지 데이터:", data.edges);
        let missingNodes = new Set();

        // ✅ 엣지 데이터에서 참조되는 노드가 `data.nodes`에 존재하는지 확인
        data.edges.forEach(edge => {
            if (!data.nodes.find(node => node.id === edge.source)) {
                missingNodes.add(edge.source);
            }
            if (!data.nodes.find(node => node.id === edge.target)) {
                missingNodes.add(edge.target);
            }
        });

        if (missingNodes.size > 0) {
            console.error("❌ 누락된 노드:", missingNodes);
        }

        let svg = d3.select("#networkGraph");
        svg.selectAll("*").remove();

        // ✅ SVG 크기 동적 조정
        let width = document.getElementById("networkGraph").clientWidth;
        let height = document.getElementById("networkGraph").clientHeight;
        console.log(`📏 SVG 크기 조정: width=${width}, height=${height}`);

        // ✅ `g` 요소를 먼저 정의한 후 `zoom` 적용
        let g = svg.append("g");

        let zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on("zoom", function (event) {
                g.attr("transform", event.transform);  // ✅ `g`가 올바르게 참조됨
            });

        svg.call(zoom);

        // ✅ 시뮬레이션을 전역 변수로 설정 (드래그 이벤트에서 접근 가능)
        window.simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.edges).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collide", d3.forceCollide(50));

        let link = g.append("g")
            .selectAll("line")
            .data(data.edges)
            .enter().append("line")
            .style("stroke", "#aaa")
            .style("stroke-width", d => Math.sqrt(d.weight) + 1);

        let node = g.append("g")
            .selectAll("circle")
            .data(data.nodes)
            .enter().append("circle")
            .attr("r", 12)
            .style("fill", d => d.type === "user_name" ? "#e74c3c" : "#3498db")
            .call(d3.drag()
                .on("start", dragStarted)
                .on("drag", dragged)
                .on("end", dragEnded));

        // ✅ 노드 라벨 추가 (각 노드의 ID 표시)
        let labels = g.append("g")
            .selectAll("text")
            .data(data.nodes)
            .enter().append("text")
            .attr("dy", -15)  // ✅ 노드 위에 표시되도록 위치 조정
            .attr("text-anchor", "middle")
            .style("font-size", "12px")
            .style("fill", "#333")
            .text(d => d.id);

        // ✅ `tick` 이벤트에서 라벨 위치 업데이트 추가
        window.simulation.on("tick", () => {
            link.attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node.attr("cx", d => d.x)
                .attr("cy", d => d.y);

            labels.attr("x", d => d.x)
                .attr("y", d => d.y - 15);
        });

    }).catch(error => console.error("❌ API 호출 실패:", error));
}

// ✅ 드래그 함수 정의 추가
function dragStarted(event, d) {
    if (!event.active) window.simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
}

function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
}

function dragEnded(event, d) {
    if (!event.active) window.simulation.alphaTarget(0);

    // ✅ 사용자가 움직인 위치 고정
    if (event.subject) {
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
    }
}