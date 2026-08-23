const API_BASE = window.location.origin;

async function fetchStats() {
  try {
    const res = await fetch(`${API_BASE}/ml/stats`);
    if (!res.ok) return;
    const data = await res.json();
    renderStats(data.stats);
    renderAnomaliesTable(data.recentAnomalies);
    renderClusters(data.clusters);
  } catch (err) {
    console.error("Failed to fetch ML stats", err);
  }
}

async function fetchModelHealth() {
  try {
    const res = await fetch(`${API_BASE}/ml/model/health`);
    if (!res.ok) return;
    const data = await res.json();
    renderModelHealth(data);
  } catch (err) {
    console.error("Failed to fetch model health", err);
  }
}

function renderStats(stats) {
  if (!stats) return;
  document.getElementById("kpi-critical").innerText = stats.criticalAlerts || 0;
  document.getElementById("kpi-high").innerText = stats.highRiskAlerts || 0;
  document.getElementById("kpi-medium").innerText = stats.mediumRiskAlerts || 0;
  document.getElementById("kpi-total-events").innerText = stats.totalEvents || 0;
}

function renderModelHealth(data) {
  const statusEl = document.getElementById("kpi-drift-status");
  const psiEl = document.getElementById("kpi-drift-psi");
  const recEl = document.getElementById("drift-recommendation");
  
  if (statusEl) {
    statusEl.innerText = data.status || "STABLE";
    statusEl.className = `badge ${data.status === 'DRIFT' ? 'CRITICAL' : (data.status === 'WARNING' ? 'MEDIUM' : 'LOW')}`;
  }
  if (psiEl) {
    psiEl.innerText = `PSI: ${data.psiScore ?? 0.00}`;
  }
  if (recEl) {
    recEl.innerText = data.recommendation || "Model parameters stable.";
  }
}

function renderAnomaliesTable(anomalies) {
  const tbody = document.getElementById("anomalies-tbody");
  if (!tbody) return;

  if (!anomalies || anomalies.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 24px;">No anomalies detected yet. Use the simulator above to test fraud scenarios.</td></tr>`;
    return;
  }

  tbody.innerHTML = anomalies.map(item => {
    const score = item.riskScore ?? item.risk_score ?? 0;
    const level = item.riskLevel ?? item.risk_level ?? 'LOW';
    const pHash = item.packHash ?? item.pack_hash ?? 'N/A';
    const bId = item.batchId ?? item.batch_id ?? 'N/A';
    const sId = item.shopId ?? item.shop_id ?? 'N/A';
    const anomList = item.anomalies || [];
    const ts = item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : 'Just now';

    const tagsHtml = anomList.map(a => `<span class="tag">${a}</span>`).join('');

    return `
      <tr>
        <td><strong>${ts}</strong></td>
        <td><span class="badge ${level}">${level} (${score}/100)</span></td>
        <td><code>${pHash.length > 18 ? pHash.substring(0, 16) + '...' : pHash}</code></td>
        <td>${bId}</td>
        <td>${sId}</td>
        <td><div class="tag-list">${tagsHtml || '<span class="tag">NORMAL</span>'}</div></td>
      </tr>
    `;
  }).join('');
}

function renderClusters(clusters) {
  const container = document.getElementById("clusters-container");
  if (!container) return;

  if (!clusters || clusters.length === 0) {
    container.innerHTML = `<div style="color: var(--text-muted); font-size: 13px; padding: 12px 0;">No active complaint hotspots detected.</div>`;
    return;
  }

  container.innerHTML = clusters.map(c => `
    <div class="cluster-card">
      <div class="cluster-header">
        <strong>📍 ${c.cluster_id}</strong>
        <span class="badge ${c.risk_level}">${c.risk_level} (${c.size} reports)</span>
      </div>
      <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 6px;">
        Centroid: ${c.centroid.latitude}, ${c.centroid.longitude} (Radius ~${c.radius_km} km)
      </div>
      <div style="font-size: 11px; color: var(--text-main);">
        Participating Batches: <strong>${c.participating_batches.join(', ') || 'N/A'}</strong>
      </div>
    </div>
  `).join('');
}

function logToConsole(text) {
  const consoleEl = document.getElementById("sim-console");
  if (!consoleEl) return;
  const time = new Date().toLocaleTimeString();
  consoleEl.textContent = `[${time}] ${text}\n` + consoleEl.textContent;
}

async function triggerSimulation(scenario) {
  logToConsole(`⚡ Triggering scenario '${scenario}'...`);
  try {
    const res = await fetch(`${API_BASE}/ml/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario })
    });
    const data = await res.json();
    logToConsole(`✅ Completed '${scenario}' (${data.resultsCount} events generated)`);
    
    if (data.timeline) {
      data.timeline.forEach(step => {
        if (step.result) {
          logToConsole(`  -> ${step.step} | Risk: ${step.result.riskLevel} (${step.result.riskScore}/100) | Flags: ${step.result.anomalies.join(', ') || 'None'}`);
        } else {
          logToConsole(`  -> ${step.step}`);
        }
      });
    }

    await fetchStats();
    await fetchModelHealth();
  } catch (err) {
    logToConsole(`❌ Simulation error: ${err}`);
  }
}

async function clearStore() {
  logToConsole(`🧹 Clearing event store...`);
  try {
    await fetch(`${API_BASE}/ml/clear`, { method: "POST" });
    logToConsole(`✅ Event store cleared.`);
    await fetchStats();
    await fetchModelHealth();
  } catch (err) {
    logToConsole(`❌ Clear error: ${err}`);
  }
}

// Polling initialization
document.addEventListener("DOMContentLoaded", () => {
  fetchStats();
  fetchModelHealth();
  setInterval(() => {
    fetchStats();
    fetchModelHealth();
  }, 4000);
});
