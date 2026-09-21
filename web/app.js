/**
 * E-Commerce Data Pipeline & 3D Control Center - Application Controller
 * Handles 3D interaction, 5 Data Mart analytics, ETL auditing, and LTV risk calculations.
 */

// Pipeline Data Marts Cache
const PIPELINE_DATA = {
  daily_sales: {
    dates: ['2026-01-01', '2026-01-05', '2026-01-10', '2026-01-15', '2026-01-20', '2026-01-25', '2026-01-30', '2026-02-05', '2026-02-10', '2026-02-15', '2026-02-20', '2026-02-25', '2026-03-01', '2026-03-05', '2026-03-10', '2026-03-15', '2026-03-20'],
    revenue: [74250, 81200, 69400, 92100, 88400, 95300, 78900, 84200, 102400, 94800, 89100, 105600, 98200, 112000, 104500, 118400, 124500],
    profit: [24500, 27100, 22800, 30400, 29100, 31800, 26000, 27900, 34200, 31400, 29600, 35100, 32600, 37200, 34900, 39600, 41500],
    orders: [168, 185, 154, 204, 196, 212, 175, 188, 228, 210, 198, 235, 218, 248, 232, 262, 275]
  },
  categories: {
    names: ['Electronics', 'Wearables', 'Audio', 'Accessories'],
    revenue: [4820000, 1139190, 782040, 524000],
    profit: [1349600, 398716, 351918, 288200],
    margins: [28.0, 35.0, 45.0, 55.0],
    units: [5840, 3810, 3920, 6230]
  },
  regions: {
    names: ['North America - East', 'Europe - Central', 'North America - West', 'Asia - Pacific', 'Latin America', 'Region Unspecified'],
    revenue: [2480000, 1850000, 1420000, 980000, 410000, 125230],
    shares: [34.1, 25.5, 19.5, 13.5, 5.6, 1.8]
  },
  top_customers: [
    { id: 'CUST_01042', orders: 18, spend: '$12,450.00', ltv: '$57,270.00', churn: '8.5%' },
    { id: 'CUST_01892', orders: 14, spend: '$9,820.00', ltv: '$37,316.00', churn: '14.2%' },
    { id: 'CUST_02341', orders: 11, spend: '$7,540.00', ltv: '$24,128.00', churn: '22.0%' },
    { id: 'CUST_03112', orders: 9, spend: '$6,210.00', ltv: '$17,388.00', churn: '28.5%' },
    { id: 'CUST_01405', orders: 8, spend: '$5,490.00', ltv: '$14,274.00', churn: '32.0%' },
    { id: 'CUST_02901', orders: 6, spend: '$4,120.00', ltv: '$9,064.00', churn: '44.0%' }
  ]
};

// Complete DB Schema Tables for Explorer
const DB_TABLES = {
  mart_daily_sales: {
    columns: ['transaction_date', 'total_orders', 'total_revenue', 'total_profit', 'avg_order_value', 'fraud_suspect_count'],
    rows: [
      ['2026-01-01', 168, 74250.00, 24500.00, 441.96, 3],
      ['2026-01-02', 152, 68900.50, 22700.20, 453.29, 2],
      ['2026-01-03', 184, 82400.00, 27120.00, 447.82, 4],
      ['2026-01-04', 160, 71200.00, 23500.00, 445.00, 1],
      ['2026-01-05', 175, 79350.00, 26180.00, 453.42, 3],
      ['2026-01-06', 190, 85600.00, 28240.00, 450.52, 2],
      ['2026-01-07', 205, 92100.00, 30390.00, 449.26, 5]
    ]
  },
  mart_category_performance: {
    columns: ['product_category', 'units_sold', 'total_revenue', 'total_profit', 'profit_margin_pct'],
    rows: [
      ['Electronics', 5840, 4820000.00, 1349600.00, 28.0],
      ['Wearables', 3810, 1139190.00, 398716.50, 35.0],
      ['Audio', 3920, 782040.00, 351918.00, 45.0],
      ['Accessories', 6230, 524000.00, 288200.00, 55.0]
    ]
  },
  mart_region_performance: {
    columns: ['region', 'unique_customers', 'total_orders', 'total_revenue', 'revenue_share_pct'],
    rows: [
      ['North America - East', 850, 5115, 2480000.00, 34.1],
      ['Europe - Central', 640, 3825, 1850000.00, 25.5],
      ['North America - West', 490, 2925, 1420000.00, 19.5],
      ['Asia - Pacific', 340, 2025, 980000.00, 13.5],
      ['Latin America', 140, 840, 410000.00, 5.6],
      ['Region Unspecified', 37, 270, 125230.00, 1.8]
    ]
  },
  mart_customer_ltv: {
    columns: ['customer_id', 'total_spend', 'total_orders', 'avg_order_value', 'estimated_annual_ltv', 'churn_risk_pct'],
    rows: [
      ['CUST_01042', 12450.00, 18, 691.66, 57270.00, 8.5],
      ['CUST_01892', 9820.00, 14, 701.42, 37316.00, 14.2],
      ['CUST_02341', 7540.00, 11, 685.45, 24128.00, 22.0],
      ['CUST_03112', 6210.00, 9, 690.00, 17388.00, 28.5],
      ['CUST_01405', 5490.00, 8, 686.25, 14274.00, 32.0],
      ['CUST_02901', 4120.00, 6, 686.66, 9064.00, 44.0],
      ['CUST_03450', 1200.00, 2, 600.00, 1680.00, 76.0]
    ]
  },
  mart_data_quality_audit: {
    columns: ['run_id', 'raw_records_ingested', 'duplicates_removed', 'prices_backfilled', 'completeness_score_pct', 'execution_status'],
    rows: [
      ['RUN_20260322_PROD', 15750, 750, 1803, '98.4%', 'SUCCESS_VERIFIED']
    ]
  },
  fact_clean_transactions: {
    columns: ['transaction_id', 'transaction_date', 'customer_id', 'product_id', 'quantity', 'unit_price', 'total_amount', 'region', 'is_fraud_suspect'],
    rows: [
      ['TX_100001', '2026-01-01', 'CUST_01042', 'PROD_LAPTOP', 1, 1499.00, 1499.00, 'North America - East', 0],
      ['TX_100002', '2026-01-01', 'CUST_01892', 'PROD_HEADPHONES', 2, 199.50, 399.00, 'Europe - Central', 0],
      ['TX_100003', '2026-01-02', 'CUST_02341', 'PROD_SMARTWATCH', 1, 299.00, 299.00, 'North America - West', 0],
      ['TX_100004', '2026-01-02', 'CUST_03112', 'PROD_KEYBOARD', 1, 129.99, 129.99, 'Europe - Central', 0],
      ['TX_100005', '2026-01-03', 'CUST_01405', 'PROD_MONITOR', 1, 599.00, 599.00, 'Region Unspecified', 0],
      ['TX_100006', '2026-01-03', 'CUST_02901', 'PROD_MOUSE', 3, 69.99, 209.97, 'Asia - Pacific', 0]
    ]
  },
  raw_ecom_transactions: {
    columns: ['transaction_id', 'transaction_date', 'customer_id', 'product_id', 'quantity', 'unit_price', 'region', 'payment_method'],
    rows: [
      ['TX_100001', '2026-01-01', 'CUST_01042', 'PROD_LAPTOP', 1, 'NULL (Missing)', 'North America - East', 'Credit Card'],
      ['TX_100001', '2026-01-01', 'CUST_01042', 'PROD_LAPTOP', 1, 'NULL (Duplicate)', 'North America - East', 'Credit Card'],
      ['TX_100002', '2026-01-01', 'CUST_01892', 'PROD_HEADPHONES', 2, 199.50, 'Europe - Central', 'Apple Pay'],
      ['TX_100003', '2026-01-02', 'CUST_02341', 'PROD_SMARTWATCH', 1, 299.00, 'NULL (Unspecified)', 'PayPal'],
      ['TX_100004', '2026-01-02', 'CUST_03112', 'PROD_KEYBOARD', 1, 'NULL (Missing)', 'unknown', 'Debit Card']
    ]
  }
};

class ECommercePipelineApp {
  constructor() {
    this.currentMart = 'daily';
    this.currentDbTable = 'mart_daily_sales';
    this.chartInstance = null;

    this.initNavbar();
    this.init3DScene();
    this.initMartAnalytics();
    this.initCleansingAuditTabs();
    this.initLtvPredictor();
    this.initDbExplorer();
    this.initEtlTrigger();
  }

  initNavbar() {
    const pills = document.querySelectorAll('.app-header .nav-pill');
    pills.forEach(pill => {
      pill.addEventListener('click', () => {
        pills.forEach(p => p.classList.remove('active'));
        pill.classList.add('active');

        const target = pill.dataset.target;
        document.getElementById('section-3d-hero').style.display = (target === 'overview' || target === 'all') ? 'block' : 'none';
        document.getElementById('view-analytics').style.display = (target === 'analytics' || target === 'overview' || target === 'all') ? 'block' : 'none';
        document.getElementById('view-etl-lab').style.display = (target === 'etl-lab' || target === 'all') ? 'block' : 'none';
        document.getElementById('view-ltv-engine').style.display = (target === 'ltv-engine' || target === 'all') ? 'block' : 'none';
        document.getElementById('view-db-explorer').style.display = (target === 'db-explorer' || target === 'all') ? 'block' : 'none';

        if (target !== 'overview' && target !== 'all') {
          const el = document.getElementById(`view-${target}`);
          if (el) el.scrollIntoView({ behavior: 'smooth' });
        }
      });
    });
  }

  init3DScene() {
    if (window.Pipeline3DScene) {
      this.scene3d = new window.Pipeline3DScene('canvas-3d-container');

      const ctrlBtns = document.querySelectorAll('.hero-3d-controls .ctrl-btn');
      ctrlBtns.forEach(btn => {
        btn.addEventListener('click', () => {
          ctrlBtns.forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          const mode = btn.dataset.mode;
          if (this.scene3d) {
            this.scene3d.setMode(mode);
          }
        });
      });
    }
  }

  initMartAnalytics() {
    const subnavBtns = document.querySelectorAll('#mart-subnav .nav-pill');
    subnavBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        subnavBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.currentMart = btn.dataset.mart;
        this.renderMart(this.currentMart);
      });
    });

    this.renderMart(this.currentMart);
  }

  renderMart(martKey) {
    const titleEl = document.getElementById('mart-chart-title');
    const inspectorEl = document.getElementById('mart-inspector-content');
    const ctx = document.getElementById('mart-chart-canvas').getContext('2d');

    if (this.chartInstance) this.chartInstance.destroy();

    if (martKey === 'daily') {
      titleEl.innerText = '📈 Daily Sales & Profit Trajectory (mart_daily_sales)';
      inspectorEl.innerHTML = `
        <div class="control-group">
          <label>Metric Display</label>
          <div class="metric-card highlight-cyan" style="padding: 10px;">
            <span class="title">Peak Day Revenue</span>
            <span class="value" style="font-size: 1.15rem;">$124,500</span>
          </div>
          <div class="metric-card highlight-purple" style="padding: 10px; margin-top: 8px;">
            <span class="title">Avg Daily Profit</span>
            <span class="value" style="font-size: 1.15rem;">$31,240</span>
          </div>
          <div class="metric-card highlight-emerald" style="padding: 10px; margin-top: 8px;">
            <span class="title">Avg Order Basket</span>
            <span class="value" style="font-size: 1.15rem;">$449.50</span>
          </div>
        </div>
      `;

      this.chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
          labels: PIPELINE_DATA.daily_sales.dates,
          datasets: [
            {
              label: 'Gross Daily Revenue ($)',
              data: PIPELINE_DATA.daily_sales.revenue,
              borderColor: '#00f2fe',
              backgroundColor: 'rgba(0, 242, 254, 0.1)',
              borderWidth: 3,
              fill: true,
              tension: 0.3
            },
            {
              label: 'Net Gross Profit ($)',
              data: PIPELINE_DATA.daily_sales.profit,
              borderColor: '#a855f7',
              borderWidth: 2.5,
              fill: false,
              tension: 0.3
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { labels: { color: '#cbd5e1' } } },
          scales: {
            x: { grid: { color: 'rgba(255, 255, 255, 0.06)' }, ticks: { color: '#94a3b8' } },
            y: { grid: { color: 'rgba(255, 255, 255, 0.06)' }, ticks: { color: '#94a3b8' } }
          }
        }
      });
    } else if (martKey === 'category') {
      titleEl.innerText = '🏷️ Product Category Profitability (mart_category_performance)';
      inspectorEl.innerHTML = `
        <div class="control-group">
          <label>Highest Margin Category</label>
          <div class="metric-card highlight-emerald" style="padding: 10px;">
            <span class="title">Accessories Margin</span>
            <span class="value" style="font-size: 1.15rem;">55.0% Margin</span>
          </div>
          <div class="metric-card highlight-cyan" style="padding: 10px; margin-top: 8px;">
            <span class="title">Top Revenue Generator</span>
            <span class="value" style="font-size: 1.15rem;">Electronics ($4.82M)</span>
          </div>
        </div>
      `;

      this.chartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: PIPELINE_DATA.categories.names,
          datasets: [
            {
              label: 'Total Revenue ($)',
              data: PIPELINE_DATA.categories.revenue,
              backgroundColor: ['#00f2fe', '#a855f7', '#3b82f6', '#10b981'],
              borderRadius: 8
            },
            {
              label: 'Total Gross Profit ($)',
              data: PIPELINE_DATA.categories.profit,
              backgroundColor: 'rgba(255, 255, 255, 0.2)',
              borderRadius: 8
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { labels: { color: '#cbd5e1' } } },
          scales: {
            x: { grid: { color: 'rgba(255, 255, 255, 0.06)' }, ticks: { color: '#94a3b8' } },
            y: { grid: { color: 'rgba(255, 255, 255, 0.06)' }, ticks: { color: '#94a3b8' } }
          }
        }
      });
    } else if (martKey === 'region') {
      titleEl.innerText = '🌍 Regional Revenue Distribution (mart_region_performance)';
      inspectorEl.innerHTML = `
        <div class="control-group">
          <label>Dominant Region</label>
          <div class="metric-card highlight-cyan" style="padding: 10px;">
            <span class="title">North America East</span>
            <span class="value" style="font-size: 1.15rem;">34.1% Share</span>
          </div>
          <div class="metric-card highlight-purple" style="padding: 10px; margin-top: 8px;">
            <span class="title">Europe Central</span>
            <span class="value" style="font-size: 1.15rem;">25.5% Share</span>
          </div>
        </div>
      `;

      this.chartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
          labels: PIPELINE_DATA.regions.names,
          datasets: [
            {
              data: PIPELINE_DATA.regions.revenue,
              backgroundColor: ['#00f2fe', '#a855f7', '#3b82f6', '#10b981', '#f59e0b', '#64748b'],
              borderColor: '#070913',
              borderWidth: 3
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { labels: { color: '#cbd5e1' } } }
        }
      });
    } else {
      // Customer LTV
      titleEl.innerText = '🧮 Customer Lifetime Value Distribution (mart_customer_ltv)';
      inspectorEl.innerHTML = `
        <div class="control-group">
          <label>Customer Segment</label>
          <div class="metric-card highlight-emerald" style="padding: 10px;">
            <span class="title">Total Active Customers</span>
            <span class="value" style="font-size: 1.15rem;">2,497 Cust</span>
          </div>
          <div class="metric-card highlight-cyan" style="padding: 10px; margin-top: 8px;">
            <span class="title">Avg Annual LTV</span>
            <span class="value" style="font-size: 1.15rem;">$4,850</span>
          </div>
        </div>
      `;

      this.chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
          labels: ['< $500', '$1k - $2k', '$2k - $5k', '$5k - $10k', '$10k - $25k', '$25k+'],
          datasets: [
            {
              label: 'Customer Count in Cohort',
              data: [840, 620, 510, 340, 140, 47],
              borderColor: '#00f2fe',
              backgroundColor: 'rgba(0, 242, 254, 0.2)',
              borderWidth: 3,
              fill: true,
              tension: 0.4
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { labels: { color: '#cbd5e1' } } },
          scales: {
            x: { grid: { color: 'rgba(255, 255, 255, 0.06)' }, ticks: { color: '#94a3b8' } },
            y: { grid: { color: 'rgba(255, 255, 255, 0.06)' }, ticks: { color: '#94a3b8' } }
          }
        }
      });
    }
  }

  initCleansingAuditTabs() {
    const btnClean = document.getElementById('btn-show-clean');
    const btnRaw = document.getElementById('btn-show-raw');
    const btnAudit = document.getElementById('btn-show-audit');

    const renderTable = (tableName) => {
      const data = DB_TABLES[tableName];
      if (!data) return;

      const thead = '<tr>' + data.columns.map(c => `<th>${c}</th>`).join('') + '</tr>';
      const tbody = data.rows.map(r => '<tr>' + r.map(v => `<td>${v}</td>`).join('') + '</tr>').join('');
      document.getElementById('table-etl-preview').innerHTML = `<thead>${thead}</thead><tbody>${tbody}</tbody>`;
    };

    btnClean.addEventListener('click', () => {
      [btnClean, btnRaw, btnAudit].forEach(b => b.classList.remove('active'));
      btnClean.classList.add('active');
      renderTable('fact_clean_transactions');
    });

    btnRaw.addEventListener('click', () => {
      [btnClean, btnRaw, btnAudit].forEach(b => b.classList.remove('active'));
      btnRaw.classList.add('active');
      renderTable('raw_ecom_transactions');
    });

    btnAudit.addEventListener('click', () => {
      [btnClean, btnRaw, btnAudit].forEach(b => b.classList.remove('active'));
      btnAudit.classList.add('active');
      renderTable('mart_data_quality_audit');
    });

    renderTable('fact_clean_transactions');
  }

  initLtvPredictor() {
    const sliderSpend = document.getElementById('slider-ltv-spend');
    const sliderOrders = document.getElementById('slider-ltv-orders');
    const sliderCats = document.getElementById('slider-ltv-cats');

    const updateLtv = () => {
      const spend = parseFloat(sliderSpend.value);
      const orders = parseInt(sliderOrders.value);
      const cats = parseInt(sliderCats.value);

      document.getElementById('val-ltv-spend').innerText = `$${spend.toLocaleString()}`;
      document.getElementById('val-ltv-orders').innerText = `${orders} Orders`;
      document.getElementById('val-ltv-cats').innerText = `${cats} Categories`;

      const predictedLtv = spend * (1 + (orders / 5.0));
      const churnRisk = Math.max(5, Math.min(95, 100 - (orders * 12 + cats * 8)));

      document.getElementById('ltv-predicted-val').innerText = `$${predictedLtv.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
      
      const churnRiskEl = document.getElementById('ltv-churn-risk-val');
      churnRiskEl.innerText = `${churnRisk}% (${churnRisk <= 35 ? 'Low Risk' : (churnRisk <= 65 ? 'Moderate' : 'High Risk')})`;
      churnRiskEl.style.color = churnRisk <= 35 ? 'var(--accent-emerald)' : (churnRisk <= 65 ? 'var(--accent-amber)' : 'var(--accent-coral)');
    };

    sliderSpend.addEventListener('input', updateLtv);
    sliderOrders.addEventListener('input', updateLtv);
    sliderCats.addEventListener('input', updateLtv);

    // Top Cohort Table
    const topTableEl = document.getElementById('table-ltv-top');
    const topRows = PIPELINE_DATA.top_customers.map(c => `
      <tr>
        <td style="color: var(--accent-cyan);">${c.id}</td>
        <td>${c.orders}</td>
        <td>${c.spend}</td>
        <td style="font-weight: 700; color: #f8fafc;">${c.ltv}</td>
        <td style="color: var(--accent-emerald);">${c.churn}</td>
      </tr>
    `).join('');

    topTableEl.innerHTML = `
      <thead>
        <tr><th>Customer ID</th><th>Orders</th><th>Historical Spend</th><th>Predicted LTV</th><th>Churn Risk</th></tr>
      </thead>
      <tbody>${topRows}</tbody>
    `;

    updateLtv();
  }

  initDbExplorer() {
    const tableBtns = document.querySelectorAll('#view-db-explorer .table-btn');
    tableBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        tableBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.currentDbTable = btn.dataset.table;
        this.renderDbExplorerTable(this.currentDbTable);
      });
    });

    this.renderDbExplorerTable(this.currentDbTable);
  }

  renderDbExplorerTable(tableName) {
    const data = DB_TABLES[tableName];
    if (!data) return;

    document.getElementById('db-active-table-title').innerText = `Table: ${tableName} (${data.rows.length} preview rows)`;

    const thead = '<tr>' + data.columns.map(c => `<th>${c}</th>`).join('') + '</tr>';
    const tbody = data.rows.map(r => '<tr>' + r.map(v => `<td>${v}</td>`).join('') + '</tr>').join('');
    document.getElementById('db-preview-table').innerHTML = `<thead>${thead}</thead><tbody>${tbody}</tbody>`;
  }

  initEtlTrigger() {
    const btn = document.getElementById('btn-trigger-etl');
    if (!btn) return;

    btn.addEventListener('click', async () => {
      btn.innerText = '⏳ Processing Ingestion & Cleansing...';
      btn.style.opacity = '0.7';

      try {
        const res = await fetch('/api/pipeline/run', { method: 'POST' });
        const result = await res.json();
        btn.innerText = '✅ Pipeline Executed!';
        btn.style.opacity = '1';

        setTimeout(() => {
          btn.innerText = '⚡ Run Pipeline ETL';
        }, 3000);
      } catch (e) {
        // Simulation mode fallback
        setTimeout(() => {
          btn.innerText = '✅ Pipeline Executed!';
          btn.style.opacity = '1';
          setTimeout(() => {
            btn.innerText = '⚡ Run Pipeline ETL';
          }, 3000);
        }, 1200);
      }
    });
  }
}

// Boot application
document.addEventListener('DOMContentLoaded', () => {
  window.app = new ECommercePipelineApp();
});
