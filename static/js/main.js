// =====================================================
// HOUSING AFFORDABILITY DASHBOARD — main.js
// =====================================================

// Cluster colour palette (mirrors backend /api/clusters)
const CLUSTER_COLORS = {
    0: "#38bdf8",   // Middle-Income Stable   — sky blue
    1: "#ef4444",   // Low-Income Burdened    — red
    2: "#22c55e",   // High-Income Secure     — green
    3: "#f59e0b"    // Extremely Low-Income   — amber
};

let dashboardData  = null;
let clusterDefs    = [];   // from /api/clusters

// =====================================================
// INIT
// =====================================================

// Handle 401 (session expired) gracefully
async function handleApiResponse(res) {
    if (res.status === 401) {
        window.location.href = '/login';
        return null;
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

document.addEventListener('DOMContentLoaded', () => {
    fetchClusterDefs().then(() => {
        fetchData();
    });
    setupForm();
});

// =====================================================
// CLUSTER DEFINITIONS
// =====================================================
async function fetchClusterDefs() {
    try {
        const res = await fetch('/api/clusters');
        const data = await handleApiResponse(res);
        if (!data) return;
        clusterDefs = data;
        renderClusterGuide(clusterDefs);
    } catch (e) {
        console.warn("Could not load cluster definitions:", e);
    }
}

function renderClusterGuide(defs) {
    const grid = document.getElementById('cluster-cards-grid');
    if (!grid) return;

    grid.innerHTML = defs.map(d => `
        <div class="cluster-info-card" style="
            background: rgba(10, 20, 40, 0.75);
            border-color: ${d.color}44;
            color: #e5e7eb;
        ">
            <div class="cluster-card-icon">${d.icon}</div>
            <div class="cluster-card-id">Cluster ${d.id}</div>
            <div class="cluster-card-label" style="color:${d.color}">${d.label}</div>
            <div class="cluster-card-policy">${d.policy}</div>
        </div>
    `).join('');
}

// =====================================================
// DASHBOARD DATA
// =====================================================
async function fetchData() {
    setMetricLoading(true);
    try {
        const res = await fetch('/api/data');
        const data = await handleApiResponse(res);
        if (!data) return;
        if (!data.stats || !data.scatter) throw new Error("Invalid data format");

        dashboardData = data;

        renderMetrics(data);

        if (typeof Plotly !== 'undefined') {
            renderScatter(data.scatter);
            renderBar(data.stats);
            renderDistribution(data.stats);
        } else {
            ['scatter-plot','bar-plot','distribution-plot'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.innerHTML = `<p style="color:#ef4444;text-align:center;padding:2rem">Charts unavailable — Plotly not loaded</p>`;
            });
        }

    } catch (err) {
        console.error("fetchData error:", err);
        setMetricError();
    } finally {
        setMetricLoading(false);
    }
}

function setMetricLoading(loading) {
    if (!loading) return;
    ['total-count','avg-burden','vulnerable-pct','cluster-count'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = '<span class="spinner">↻</span>';
    });
}

function setMetricError() {
    const retry = `<span style="color:#ef4444;font-size:0.85rem">Failed <button onclick="fetchData()" style="background:none;border:1px solid #ef4444;color:#ef4444;cursor:pointer;padding:2px 6px;border-radius:4px;margin-left:4px">Retry</button></span>`;
    ['total-count','avg-burden','vulnerable-pct','cluster-count'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = retry;
    });
}

// =====================================================
// METRICS
// =====================================================
function renderMetrics(data) {
    let total = 0, burdenedCount = 0, weightedBurden = 0;

    Object.values(data.stats).forEach(c => {
        total          += c.Count || 0;
        weightedBurden += (c.cost_burden_ratio || 0) * (c.Count || 0);
        if ((c.cost_burden_ratio || 0) > 0.3) burdenedCount += c.Count || 0;
    });

    const avgBurden = total > 0 ? (weightedBurden / total).toFixed(2) : "—";
    const vulnPct   = total > 0 ? ((burdenedCount / total) * 100).toFixed(1) + "%" : "—";
    const clusterN  = Object.keys(data.stats).length;

    animateNumber('total-count', total);
    document.getElementById('avg-burden').innerText   = avgBurden;
    document.getElementById('vulnerable-pct').innerText = vulnPct;
    document.getElementById('cluster-count').innerText  = clusterN;
}

function animateNumber(id, target) {
    const el = document.getElementById(id);
    if (!el) return;
    const duration = 1200;
    const start = performance.now();
    const step = (now) => {
        const t = Math.min((now - start) / duration, 1);
        const ease = 1 - Math.pow(1 - t, 3);
        el.innerText = Math.floor(ease * target).toLocaleString();
        if (t < 1) requestAnimationFrame(step);
        else el.innerText = target.toLocaleString();
    };
    requestAnimationFrame(step);
}

// =====================================================
// CHARTS
// =====================================================
const plotlyLayout = (extra = {}) => ({
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor:  'rgba(0,0,0,0)',
    font: { color: '#94a3b8', family: 'Inter, sans-serif', size: 12 },
    margin: { t: 20, r: 20, b: 50, l: 60 },
    ...extra
});

const plotlyConfig = { displayModeBar: false, responsive: true };

function clusterColor(id) {
    return CLUSTER_COLORS[id] ?? '#94a3b8';
}

function clusterName(id) {
    const d = clusterDefs.find(x => x.id === Number(id));
    return d ? d.label : `Cluster ${id}`;
}

// Scatter: Income vs. Burden
function renderScatter(scatterData) {
    if (!scatterData) return;
    const clusters = [...new Set(scatterData.map(d => d.Cluster_Label))].sort();

    const traces = clusters.map(cl => {
        const pts = scatterData.filter(d => d.Cluster_Label === cl);
        return {
            x: pts.map(d => d.ZINC2),
            y: pts.map(d => d.cost_burden_ratio),
            mode: 'markers',
            type: 'scatter',
            name: clusterName(cl),
            marker: {
                size: 5,
                opacity: 0.65,
                color: clusterColor(cl)
            }
        };
    });

    const layout = plotlyLayout({
        xaxis: {
            title: { text: 'Annual Income ($)', font: { color: '#94a3b8' } },
            gridcolor: 'rgba(255,255,255,0.06)',
            zeroline: false
        },
        yaxis: {
            title: { text: 'Cost Burden Ratio', font: { color: '#94a3b8' } },
            gridcolor: 'rgba(255,255,255,0.06)',
            range: [0, 2],
            zeroline: false
        },
        legend: { orientation: 'h', y: -0.2, x: 0 },
        shapes: [
            {
                type: 'line', xref: 'paper', yref: 'y',
                x0: 0, x1: 1, y0: 0.3, y1: 0.3,
                line: { color: '#f59e0b', width: 1.5, dash: 'dot' }
            },
            {
                type: 'line', xref: 'paper', yref: 'y',
                x0: 0, x1: 1, y0: 0.5, y1: 0.5,
                line: { color: '#ef4444', width: 1.5, dash: 'dot' }
            }
        ],
        annotations: [
            { xref: 'paper', yref: 'y', x: 1.01, y: 0.3, text: '30%', showarrow: false, font: { color: '#f59e0b', size: 10 }, xanchor: 'left' },
            { xref: 'paper', yref: 'y', x: 1.01, y: 0.5, text: '50%', showarrow: false, font: { color: '#ef4444', size: 10 }, xanchor: 'left' }
        ]
    });

    Plotly.newPlot('scatter-plot', traces, layout, plotlyConfig);
}

// Bar: Avg Income + Burden per Cluster
function renderBar(stats) {
    if (!stats) return;
    const ids     = Object.keys(stats).map(Number).sort();
    const labels  = ids.map(id => `Cluster ${id}`);
    const incomes = ids.map(id => stats[id].ZINC2 || 0);
    const burdens = ids.map(id => stats[id].cost_burden_ratio || 0);
    const colors  = ids.map(id => clusterColor(id));

    const t1 = {
        x: labels, y: incomes,
        name: 'Avg Income ($)',
        type: 'bar',
        marker: { color: colors, opacity: 0.85 },
        text: incomes.map(v => `$${(v/1000).toFixed(0)}k`),
        textposition: 'outside',
        textfont: { color: '#94a3b8', size: 11 }
    };

    const t2 = {
        x: labels, y: burdens,
        name: 'Burden Ratio',
        type: 'scatter',
        mode: 'lines+markers',
        yaxis: 'y2',
        line: { color: '#f59e0b', width: 2 },
        marker: { color: '#f59e0b', size: 8 }
    };

    const layout = plotlyLayout({
        yaxis:  { title: 'Income ($)', gridcolor: 'rgba(255,255,255,0.06)', zeroline: false },
        yaxis2: { title: 'Burden Ratio', overlaying: 'y', side: 'right', gridcolor: 'rgba(0,0,0,0)', zeroline: false },
        barmode: 'group',
        legend: { orientation: 'h', y: -0.2, x: 0 }
    });

    Plotly.newPlot('bar-plot', [t1, t2], layout, plotlyConfig);
}

// Distribution: Pie / Donut
function renderDistribution(stats) {
    const ids    = Object.keys(stats).map(Number).sort();
    const labels = ids.map(id => `Cluster ${id} — ${clusterName(id)}`);
    const values = ids.map(id => stats[id].Count || 0);
    const colors = ids.map(id => clusterColor(id));

    const trace = {
        labels, values,
        type: 'pie',
        hole: 0.55,
        marker: { colors, line: { color: '#050b14', width: 2 } },
        textinfo: 'label+percent',
        textfont: { color: '#e5e7eb', size: 12 },
        hovertemplate: '<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>'
    };

    const layout = plotlyLayout({ height: 320, margin: { t: 20, b: 20, l: 20, r: 20 } });
    Plotly.newPlot('distribution-plot', [trace], layout, plotlyConfig);
}

// =====================================================
// TABS
// =====================================================
window.openTab = function(tabName, btn) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById(tabName).classList.add('active');
    btn.classList.add('active');
    window.dispatchEvent(new Event('resize'));
};

// =====================================================
// PREDICTION FORM
// =====================================================
function setupForm() {
    const form = document.getElementById('predict-form');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const payload = {
            income:   document.getElementById('income').value,
            cost:     document.getElementById('cost').value,
            age:      document.getElementById('age').value,
            bedrooms: document.getElementById('bedrooms').value
        };

        // Button state
        const btnText   = document.getElementById('btn-text');
        const btnLoader = document.getElementById('btn-loader');
        btnText.style.display   = 'none';
        btnLoader.style.display = 'inline';

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await handleApiResponse(res);
            if (!result) return;

            renderResult(result);

        } catch (err) {
            console.error("Prediction failed:", err);
            alert("⚠️ Prediction failed. Check the server is running.");
        } finally {
            btnText.style.display   = 'inline';
            btnLoader.style.display = 'none';
        }
    });
}

function renderResult(result) {
    // Show the result card
    const card = document.getElementById('result-card');
    card.style.display = 'block';
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Cluster badge
    const clusterDef = clusterDefs.find(d => d.id === result.cluster) || {};
    const color  = clusterDef.color  || '#38bdf8';
    const icon   = clusterDef.icon   || '🔵';
    const label  = clusterDef.label  || `Cluster ${result.cluster}`;
    const badge  = document.getElementById('cluster-badge-wrap');
    badge.innerHTML = `
        <div class="cluster-badge" style="
            color: ${color};
            border-color: ${color}55;
            background: ${color}18;
        ">
            <span>${icon}</span>
            <span>${label}</span>
        </div>
    `;

    // Core values
    document.getElementById('pred-cluster').innerText = `Cluster ${result.cluster}`;
    document.getElementById('pred-cluster').style.color = color;

    const burdenPct = (result.cost_burden_ratio * 100).toFixed(1);
    document.getElementById('pred-burden').innerText =
        `${burdenPct}% (${result.cost_burden_label})`;

    document.getElementById('pred-rec').innerText = result.recommendation;

    // Burden meter
    const fill = document.getElementById('burden-meter-fill');
    const ratio = Math.min(result.cost_burden_ratio, 1.0);
    fill.style.width = (ratio * 100) + '%';
    if (ratio < 0.3)       fill.style.background = `linear-gradient(90deg, #22c55e, #4ade80)`;
    else if (ratio < 0.5)  fill.style.background = `linear-gradient(90deg, #f59e0b, #fbbf24)`;
    else                   fill.style.background = `linear-gradient(90deg, #ef4444, #f87171)`;

    // Anomaly banner
    const banner = document.getElementById('anomaly-banner');
    if (result.anomaly_flag && result.anomaly_reasons.length) {
        banner.style.display = 'block';
        banner.textContent = "⚠️ " + result.anomaly_reasons.join(" · ");
    } else {
        banner.style.display = 'none';
    }
}
