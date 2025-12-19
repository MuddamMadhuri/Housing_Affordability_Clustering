// Global Data
let dashboardData = null;

// Init
document.addEventListener('DOMContentLoaded', () => {
    fetchData();
    setupTabs();
    setupForm();
});

// Fetch Data
async function fetchData() {
    const metrics = ['total-count', 'avg-burden', 'vulnerable-pct'];

    // Show Loading
    metrics.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = '<span class="spinner">↻</span> Loading...';
    });

    try {
        const response = await fetch('/api/data');
        if (!response.ok) throw new Error(`Server returned ${response.status}`);

        const data = await response.json();
        if (!data.stats || !data.scatter) throw new Error("Invalid data format received");

        dashboardData = data;

        renderMetrics(data);

        // Check if Plotly is loaded
        if (typeof Plotly !== 'undefined') {
            renderScatter(data.scatter);
            renderBar(data.stats);
        } else {
            console.warn("Plotly not loaded, skipping charts.");
            const err = "<p class='error' style='color: #ef4444; text-align: center; padding: 2rem;'>Charts unavailable (Plotly missing)</p>";
            document.getElementById('scatter-plot').innerHTML = err;
            document.getElementById('bar-plot').innerHTML = err;
        }

    } catch (error) {
        console.error("Error fetching data:", error);
        const errorMsg = `<span class="error" style="color: #ef4444;">Failed to load. <button onclick="fetchData()" style="background:none; border:1px solid #ef4444; color:#ef4444; cursor:pointer; padding:2px 5px; border-radius:4px;">Retry</button></span>`;
        metrics.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.innerHTML = errorMsg;
        });
    }
}

// Render Metrics
function renderMetrics(data) {
    let total = 0;
    let burdenedCount = 0;

    // stats is {label: {Count: ..., cost_burden_ratio: ...}}
    if (data.stats) {
        Object.values(data.stats).forEach(cluster => {
            total += cluster.Count || 0;
            if (cluster.cost_burden_ratio > 0.3) {
                burdenedCount += cluster.Count || 0;
            }
        });
    }

    const totalEl = document.getElementById('total-count');
    if (totalEl) totalEl.innerText = total.toLocaleString();

    // Weighted avg burden
    let weightedBurden = 0;
    if (data.stats && total > 0) {
        Object.values(data.stats).forEach(c => {
            weightedBurden += (c.cost_burden_ratio * (c.Count || 0));
        });
        const avgBurden = (weightedBurden / total).toFixed(2);
        const avgEl = document.getElementById('avg-burden');
        if (avgEl) avgEl.innerText = avgBurden;

        const vuln = ((burdenedCount / total) * 100).toFixed(1) + "%";
        const vulnEl = document.getElementById('vulnerable-pct');
        if (vulnEl) vulnEl.innerText = vuln;
    } else {
        document.getElementById('avg-burden').innerText = "-";
        document.getElementById('vulnerable-pct').innerText = "-";
    }
}

// Render Scatter
function renderScatter(scatterData) {
    if (!scatterData) return;
    const traces = [];
    const clusters = [...new Set(scatterData.map(d => d.Cluster_Label))].sort();

    clusters.forEach(cluster => {
        const clusterPoints = scatterData.filter(d => d.Cluster_Label === cluster);
        traces.push({
            x: clusterPoints.map(d => d.ZINC2),
            y: clusterPoints.map(d => d.cost_burden_ratio),
            mode: 'markers',
            type: 'scatter',
            name: `Cluster ${cluster}`,
            marker: { size: 6, opacity: 0.7 }
        });
    });

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#94a3b8' },
        xaxis: { title: 'Annual Income ($)', gridcolor: 'rgba(255,255,255,0.1)' },
        yaxis: { title: 'Cost Burden Ratio', gridcolor: 'rgba(255,255,255,0.1)', range: [0, 2] },
        margin: { t: 20, r: 20, b: 40, l: 50 }
    };

    Plotly.newPlot('scatter-plot', traces, layout);
}

// Render Bar
function renderBar(stats) {
    if (!stats) return;
    const labels = Object.keys(stats).map(k => `Cluster ${k}`);
    const incomes = Object.values(stats).map(d => d.ZINC2);
    const burdens = Object.values(stats).map(d => d.cost_burden_ratio);

    const trace1 = {
        x: labels,
        y: incomes,
        name: 'Avg Income',
        type: 'bar',
        marker: { color: '#38bdf8' }
    };

    const trace2 = {
        x: labels,
        y: burdens,
        name: 'Avg Burden',
        yaxis: 'y2',
        type: 'scatter',
        mode: 'lines+markers',
        marker: { color: '#ef4444' }
    };

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#94a3b8' },
        yaxis: { title: 'Income ($)', gridcolor: 'rgba(255,255,255,0.1)' },
        yaxis2: {
            title: 'Burden Ratio',
            overlaying: 'y',
            side: 'right',
            gridcolor: 'rgba(255,255,255,0)'
        },
        margin: { t: 20, r: 50, b: 40, l: 50 },
        showlegend: true,
        legend: { x: 0, y: 1.1, orientation: 'h' }
    };

    Plotly.newPlot('bar-plot', [trace1, trace2], layout);
}

// Tabs
function setupTabs() {
    window.openTab = (tabName) => {
        document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));

        document.getElementById(tabName).classList.add('active');
        event.currentTarget.classList.add('active');

        // Resize plots when tab becomes visible
        window.dispatchEvent(new Event('resize'));
    };
}

// Form
function setupForm() {
    const form = document.getElementById('predict-form');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const data = {
            income: document.getElementById('income').value,
            cost: document.getElementById('cost').value,
            age: document.getElementById('age').value,
            bedrooms: document.getElementById('bedrooms').value
        };

        const btn = e.target.querySelector('button');
        const originalText = btn.innerText;
        btn.innerText = "Analyzing...";

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            document.getElementById('result-card').style.display = 'block';
            document.getElementById('pred-cluster').innerText = `Cluster ${result.cluster}`;

            // Format burden as percentage with label
            const burdenPct = (result.cost_burden_ratio * 100).toFixed(1) + "%";
            document.getElementById('pred-burden').innerText = `${burdenPct} (${result.cost_burden_label})`;

            document.getElementById('pred-rec').innerText = result.recommendation;

            // Handle Anomaly Banner
            const banner = document.getElementById('anomaly-banner');
            if (result.anomaly_flag) {
                banner.style.display = 'block';
                banner.innerText = "⚠️ Critical: " + result.anomaly_reasons.join(". ");
            } else {
                banner.style.display = 'none';
            }

        } catch (error) {
            alert("Prediction failed");
        } finally {
            btn.innerText = originalText;
        }
    });
}
