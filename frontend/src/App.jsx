
import { useEffect, useState } from 'react'
import {
  Activity, AlertTriangle, ArrowDownRight, ArrowUpRight,
  BarChart3, Boxes, ChevronDown, ChevronLeft, ChevronRight,
  CircleHelp, Clock3, Command, Download, LayoutDashboard,
  Menu, Package, RefreshCw, Search, Settings2, ShieldCheck,
  SlidersHorizontal, Sparkles, Truck, X,
} from 'lucide-react'
import {
} from 'recharts'

const money = (n) => Number(n || 0).toLocaleString('en-IN')
const compact = (n) =>
  Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(n || 0)

const navItems = [
  { label: 'Overview', icon: LayoutDashboard },
  { label: 'Inventory', icon: Boxes },
  { label: 'Forecasting', icon: Activity },
  { label: 'Replenishment', icon: Truck },
]


function MetricCard({ title, value, subtitle, icon: Icon, tone = 'blue' }) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <span>{title}</span>
        <span className={`metric-icon ${tone}`}><Icon size={18} /></span>
      </div>
      <div className="metric-value">{value}</div>
      <div className="metric-subtitle">{subtitle}</div>
    </div>
  )
}

function StatusBadge({ status }) {
  const className = status === 'Reorder now'
    ? 'status status-red'
    : status === 'Monitor'
      ? 'status status-amber'
      : status === 'Review inactive product'
        ? 'status status-gray'
        : 'status status-green'

  return <span className={className}><span className="status-dot" />{status}</span>
}

function App() {
  const [summary, setSummary] = useState(null)
  const [items, setItems] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [recommendation, setRecommendation] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [activeNav, setActiveNav] = useState('Overview')
  const [selectedProduct, setSelectedProduct] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [modelEvaluation, setModelEvaluation] = useState(null)
  const [inventoryEvaluation, setInventoryEvaluation] = useState(null)
  const [testEvaluation, setTestEvaluation] = useState(null)

  const pageSize = 10

  async function loadSummary() {
    const response = await fetch('/api/summary')
    if (!response.ok) throw new Error('Could not load inventory summary.')
    setSummary(await response.json())
  }

  async function loadInventory() {
    setLoading(true)
    setError('')
    try {
      const params = new URLSearchParams({
        page: String(page),
        page_size: String(pageSize),
        sort_by: 'recommended_order_qty',
        order: 'desc',
      })
      if (search.trim()) params.set('search', search.trim())
      if (recommendation) params.set('recommendation', recommendation)

      const response = await fetch(`/api/inventory?${params}`)
      if (!response.ok) throw new Error('Could not load inventory recommendations.')
      const data = await response.json()
      setItems(data.items || [])
      setTotal(data.total || 0)
    } catch (err) {
      setError(`${err.message} Ensure FastAPI is running on port 8000.`)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadSummary().catch((err) => setError(err.message))
  }, [])

  useEffect(() => {
    async function loadEvaluations() {
      try {
        const [modelsResponse, validationResponse, testResponse] =
          await Promise.all([
            fetch('/api/evaluation/models'),
            fetch('/api/evaluation/inventory?period=validation'),
            fetch('/api/evaluation/inventory?period=test'),
          ])

        if (!modelsResponse.ok || !validationResponse.ok || !testResponse.ok) {
          throw new Error('Could not load evaluation reports.')
        }

        const [models, validation, test] = await Promise.all([
          modelsResponse.json(),
          validationResponse.json(),
          testResponse.json(),
        ])

        setModelEvaluation(models)
        setInventoryEvaluation(validation)
        setTestEvaluation(test)
      } catch (err) {
        setError(err.message)
      }
    }

    loadEvaluations()
  }, [])

  useEffect(() => {
    const timer = setTimeout(() => loadInventory(), 250)
    return () => clearTimeout(timer)
  }, [page, search, recommendation])

  const maxPages = Math.max(1, Math.ceil(total / pageSize))


  function exportCsv() {
    if (!items.length) return
    const columns = [
      'StockCode', 'average_daily_demand', 'safety_stock',
      'reorder_point', 'recommended_order_qty', 'recommendation',
    ]
    const escapeCsv = (value) =>
      `"${String(value ?? '').replaceAll('"', '""')}"`
    const rows = [columns.join(',')]
    items.forEach((item) => {
      rows.push(columns.map((column) => escapeCsv(item[column])).join(','))
    })
    const blob = new Blob([rows.join('\r\n')], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = 'stockpilot-inventory-page.csv'
    link.click()
    URL.revokeObjectURL(url)
  }

  async function openProduct(stockCode) {
    try {
      const response = await fetch(`/api/inventory/${encodeURIComponent(stockCode)}`)
      if (!response.ok) throw new Error('Product details unavailable.')
      setSelectedProduct(await response.json())
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? 'sidebar-open' : ''}`}>
        <div className="brand">
          <div className="brand-mark"><Command size={21} /></div>
          <div>
            <div className="brand-name">stockpilot<span>ml</span></div>
            <div className="brand-caption">INTELLIGENT INVENTORY</div>
          </div>
          <button className="icon-button sidebar-close" onClick={() => setSidebarOpen(false)} aria-label="Close sidebar"><X size={18} /></button>
        </div>

        <div className="workspace">
          <div className="workspace-avatar">S</div>
          <div className="workspace-copy">
            <strong>StockPilot Workspace</strong>
            <span>Retail operations</span>
          </div>
          <ChevronDown size={15} className="muted-icon" />
        </div>

        <div className="nav-label">WORKSPACE</div>
        <nav className="nav-list">
          {navItems.map(({ label, icon: Icon }) => (
            <button
              key={label}
              className={`nav-item ${activeNav === label ? 'nav-active' : ''}`}
              onClick={() => {
                setActiveNav(label)
                setSidebarOpen(false)
                setPage(1)
                setSearch('')
                setRecommendation('')
              }}
            >
              <Icon size={18} />
              <span>{label}</span>
              {label === 'Replenishment' && <span className="nav-count">{summary?.reorder_now ?? '—'}</span>}
            </button>
          ))}
        </nav>

        <div className="nav-label tools-label">TOOLS</div>
        <button className="nav-item" onClick={() => window.open('http://127.0.0.1:8000/docs', '_blank')}>
          <Settings2 size={18} /><span>API explorer</span>
        </button>
        <button className="nav-item" onClick={() => window.open('http://127.0.0.1:8000/health', '_blank')}>
          <ShieldCheck size={18} /><span>System health</span>
        </button>

        <div className="sidebar-bottom">
          <div className="help-card">
            <div className="help-icon"><CircleHelp size={18} /></div>
            <strong>Need a hand?</strong>
            <p>Explore the API docs to learn how StockPilot works.</p>
            <button onClick={() => window.open('http://127.0.0.1:8000/docs', '_blank')}>Explore documentation ↗</button>
          </div>
          <div className="user-profile">
            <div className="user-avatar">SP</div>
            <div className="user-copy"><strong>Project workspace</strong><span>Local development</span></div>
            <span className="online-dot" title="Local app" />
          </div>
        </div>
      </aside>

      {sidebarOpen && <button className="mobile-overlay" onClick={() => setSidebarOpen(false)} aria-label="Close menu" />}

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumbs">
            <button className="mobile-menu icon-button" onClick={() => setSidebarOpen(true)} aria-label="Open menu"><Menu size={20} /></button>
            <span>Workspace</span><span className="crumb-slash">/</span><strong>{activeNav}</strong>
          </div>
          <div className="topbar-actions">
            <div className="system-status"><span className="online-dot" /> API connected</div>
            <button className="icon-button" onClick={() => { loadSummary().catch(() => {}); loadInventory() }} title="Refresh data" aria-label="Refresh data"><RefreshCw size={17} /></button>
            <div className="top-avatar">SP</div>
          </div>
        </header>

        <div className="page-container">
          <section className="page-heading">
            <div>
              <div className="eyebrow"><Sparkles size={14} /> RETAIL INTELLIGENCE PLATFORM</div>
              <h1>{{
                Overview: 'Inventory overview',
                Inventory: 'Inventory management',
                Forecasting: 'Model evaluation',
                Replenishment: 'Inventory simulation',
              }[activeNav] || activeNav}</h1>
              <p>{{
                Overview: 'A concise view of inventory health and replenishment priorities.',
                Inventory: 'Search products, inspect stock policies, and review recommended quantities.',
                Forecasting: 'Compare evaluated forecasting approaches using held-out validation metrics.',
                Replenishment: 'Evaluate inventory policies against service, stockout, and cost trade-offs.',
              }[activeNav] || 'StockPilot ML retail intelligence.'}</p>
            </div>
            <div className="heading-actions">
              <div className="live-chip"><span className="online-dot" /> LIVE DATA</div>
              {activeNav === 'Inventory' && <button className="primary-button" onClick={exportCsv}><Download size={16} /> Export current page</button>}
            </div>
          </section>

          {error && <div className="error-banner"><AlertTriangle size={17} />{error}<button onClick={() => setError('')} aria-label="Dismiss error"><X size={16} /></button></div>}

          {activeNav === 'Overview' && (
          <section className="metric-grid">
            <MetricCard title="Products analyzed" value={summary ? money(summary.total_products) : '—'} subtitle="Training-cohort products" icon={Package} tone="blue" />
            <MetricCard title="Reorder recommended" value={summary ? money(summary.reorder_now) : '—'} subtitle="Prioritized for review" icon={Truck} tone="purple" />
            <MetricCard title="Units to replenish now" value={summary ? compact(summary.immediate_replenishment_units) : '—'} subtitle="Based on simulated stock inputs" icon={Boxes} tone="orange" />
            <MetricCard title="Products to monitor" value={summary ? money(summary.monitor) : '—'} subtitle="Review demand activity" icon={Clock3} tone="green" />
          </section>
          )}

          {activeNav === 'Overview' && (
            <section className="insight-grid overview-insights">
              <div className="panel priority-panel">
                <div className="panel-header">
                  <div><h2>Action center</h2><p>Current recommendation breakdown</p></div>
                  <span className="panel-icon orange"><AlertTriangle size={18} /></span>
                </div>
                <div className="priority-total">
                  <div><span>Immediate reorder candidates</span><strong>{summary ? money(summary.reorder_now) : '—'}</strong></div>
                  <span className="priority-symbol"><ArrowUpRight size={18} /></span>
                </div>
                {[
                  { label: 'Reorder now', count: summary?.reorder_now, color: 'bar-red' },
                  { label: 'Monitor', count: summary?.monitor, color: 'bar-amber' },
                  { label: 'Review inactive', count: summary?.review_inactive, color: 'bar-gray' },
                  { label: 'No order needed', count: summary?.no_order_needed, color: 'bar-green' },
                ].map((entry) => {
                  const count = entry.count ?? 0
                  const percent = summary?.total_products ? count / summary.total_products * 100 : 0
                  return (
                    <div className="priority-row" key={entry.label}>
                      <div className="priority-label"><span>{entry.label}</span><strong>{summary ? money(count) : '—'}</strong></div>
                      <div className="progress-track"><div className={`progress-fill ${entry.color}`} style={{ width: `${percent}%` }} /></div>
                    </div>
                  )
                })}
                <div className="assumption-note"><ShieldCheck size={16} /><span>Recommendations use simulated on-hand inventory and order quantities. Verify actual stock before acting.</span></div>
              </div>
              <div className="panel overview-note-panel">
                <div className="panel-header">
                  <div><h2>Project status</h2><p>What the current results mean</p></div>
                  <span className="panel-icon"><ShieldCheck size={18} /></span>
                </div>
                <div className="overview-status-item">
                  <span className="status status-amber"><span className="status-dot" />Forecasting needs improvement</span>
                  <p>The zero-demand baseline currently leads on overall validation MAE and WAPE. The model evaluation page shows the measured comparison.</p>
                </div>
                <div className="overview-status-item">
                  <span className="status status-green"><span className="status-dot" />Inventory simulation evaluated</span>
                  <p>The safety-stock policy reaches an 86.58% validation fill rate, but 78.41% on held-out test. Both results are simulated.</p>
                </div>
                <div className="assumption-note"><ShieldCheck size={16} /><span>Historical sales are an imperfect demand proxy; on-hand stock and costs are simulated assumptions.</span></div>
              </div>
            </section>
          )}

          {activeNav === 'Forecasting' && (
            <section className="evaluation-grid">
            <div className="panel evaluation-panel">
              <div className="panel-header">
                <div>
                  <h2>Forecasting model evaluation</h2>
                  <p>Measured validation results · 500 products · 55,500 rows</p>
                </div>
                <span className="panel-icon"><BarChart3 size={18} /></span>
              </div>

              {!modelEvaluation ? (
                <p className="evaluation-loading">Loading model metrics…</p>
              ) : (
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>MODEL</th>
                        <th>MAE ↓</th>
                        <th>RMSE ↓</th>
                        <th>WAPE ↓</th>
                        <th>POSITIVE-DAY MAE ↓</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(modelEvaluation.metrics || []).map((model, index) => (
                        <tr key={model.Model ?? model.model ?? index}>
                          <td><strong>{String(model.Model ?? model.model ?? 'Unknown').replaceAll('_', ' ')}</strong></td>
                          <td>{Number(model.MAE ?? model.mae ?? 0).toFixed(3)}</td>
                          <td>{Number(model.RMSE ?? model.rmse ?? 0).toFixed(3)}</td>
                          <td>{Number(model.WAPE_percent ?? model.wape_percent ?? 0).toFixed(2)}%</td>
                          <td>{Number(model.Positive_day_MAE ?? model.positive_day_mae ?? 0).toFixed(3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              <p className="evaluation-note">
                The zero baseline performs best on overall MAE and WAPE in this
                comparison. Lower positive-day MAE and RMSE tell a different story.
                These are saved validation results, not live forecasts.
              </p>
            </div>

            </section>
          )}

          {activeNav === 'Replenishment' && (
            <section className="evaluation-grid">
            <div className="panel evaluation-panel">
              <div className="panel-header">
                <div>
                  <h2>Inventory policy evaluation</h2>
                  <p>Validation compared with held-out test results</p>
                </div>
                <span className="panel-icon orange"><Boxes size={18} /></span>
              </div>

              {!inventoryEvaluation || !testEvaluation ? (
                <p className="evaluation-loading">Loading simulation results…</p>
              ) : (
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>POLICY</th>
                        <th>VALIDATION FILL</th>
                        <th>TEST FILL</th>
                        <th>TEST LOST UNITS</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(inventoryEvaluation.results || []).map((validation, index) => {
                        const policy = validation.policy ?? validation.Policy ?? validation.policy_name
                        const test = (testEvaluation.results || []).find(
                          (row) => (row.policy ?? row.Policy ?? row.policy_name) === policy
                        )
                        const validationFill = validation.unit_fill_rate_pct ?? validation.fill_rate_pct
                        const testFill = test?.unit_fill_rate_pct ?? test?.fill_rate_pct
                        const testLostUnits = test?.lost_units ?? test?.total_lost_units

                        return (
                          <tr key={policy ?? index}>
                            <td><strong>{String(policy ?? 'Unknown').replaceAll('_', ' ')}</strong></td>
                            <td>{validationFill == null ? '—' : `${Number(validationFill).toFixed(2)}%`}</td>
                            <td>{testFill == null ? '—' : `${Number(testFill).toFixed(2)}%`}</td>
                            <td>{testLostUnits == null ? '—' : Number(testLostUnits).toLocaleString('en-IN')}</td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              )}

              <p className="evaluation-note">
                The minimum fill-rate target is 85%. The safety-stock policy meets
                it on validation but not on the held-out test period. Inventory
                quantities are simulated, not live warehouse measurements.
              </p>
            </div>
            </section>
          )}

          {(activeNav === 'Inventory') && (
          <section className="panel inventory-panel">
            <div className="inventory-header">
              <div><h2>Inventory recommendations</h2><p>Prioritized product-level replenishment view</p></div>
              <div className="table-actions">
                <label className="search-box"><Search size={16} /><input value={search} onChange={(event) => { setSearch(event.target.value); setPage(1) }} placeholder="Search stock code..." /></label>
                <label className="filter-select"><SlidersHorizontal size={15} /><select value={recommendation} onChange={(event) => { setRecommendation(event.target.value); setPage(1) }}><option value="">All actions</option><option value="Reorder now">Reorder now</option><option value="Monitor">Monitor</option><option value="Review inactive product">Review inactive</option><option value="No order needed">No order needed</option></select></label>
              </div>
            </div>

            <div className="table-scroll">
              <table>
                <thead><tr><th>PRODUCT</th><th>AVG. DAILY DEMAND</th><th>SAFETY STOCK</th><th>REORDER POINT</th><th>ORDER QTY</th><th>RECOMMENDATION</th><th /></tr></thead>
                <tbody>
                  {loading ? (
                    <tr><td colSpan="7" className="table-message">Loading inventory recommendations…</td></tr>
                  ) : items.length === 0 ? (
                    <tr><td colSpan="7" className="table-message">No products match these filters.</td></tr>
                  ) : items.map((item) => (
                    <tr key={item.StockCode} onClick={() => openProduct(item.StockCode)} className="clickable-row">
                      <td><div className="product-cell"><div className="product-icon"><Package size={17} /></div><div><strong>{item.StockCode}</strong><span>Retail stock code</span></div></div></td>
                      <td>{Number(item.average_daily_demand || 0).toFixed(2)} <span className="unit-label">units/day</span></td>
                      <td>{money(item.safety_stock)}</td>
                      <td>{money(item.reorder_point)}</td>
                      <td><strong className="order-qty">{money(item.recommended_order_qty)}</strong></td>
                      <td><StatusBadge status={item.recommendation} /></td>
                      <td><ChevronRight size={16} className="row-chevron" /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="table-footer">
              <span>Showing <strong>{items.length ? (page - 1) * pageSize + 1 : 0}–{Math.min(page * pageSize, total)}</strong> of <strong>{money(total)}</strong> products</span>
              <div className="pagination">
                <button disabled={page <= 1} onClick={() => setPage((p) => p - 1)} aria-label="Previous page"><ChevronLeft size={17} /></button>
                <span>Page <strong>{page}</strong> of <strong>{maxPages}</strong></span>
                <button disabled={page >= maxPages} onClick={() => setPage((p) => p + 1)} aria-label="Next page"><ChevronRight size={17} /></button>
              </div>
            </div>
          </section>
          )}

          <footer className="page-footer">
            <span>© 2026 StockPilot ML · Retail intelligence prototype</span>
            <span><span className="online-dot" /> Connected to local API</span>
          </footer>
        </div>
      </main>

      {selectedProduct && (
        <div className="modal-backdrop" onClick={() => setSelectedProduct(null)}>
          <section className="product-modal" onClick={(event) => event.stopPropagation()}>
            <div className="modal-header"><div><span className="eyebrow">PRODUCT DETAIL</span><h2>Stock code {selectedProduct.StockCode}</h2></div><button className="icon-button" onClick={() => setSelectedProduct(null)} aria-label="Close product details"><X size={19} /></button></div>
            <StatusBadge status={selectedProduct.recommendation} />
            <div className="detail-grid">
              {[
                ['Average daily demand', `${Number(selectedProduct.average_daily_demand || 0).toFixed(2)} units`],
                ['Daily demand std.', money(selectedProduct.daily_demand_std)],
                ['Safety stock', money(selectedProduct.safety_stock)],
                ['Reorder point', money(selectedProduct.reorder_point)],
                ['Order-up-to level', money(selectedProduct.order_up_to_level)],
                ['Recommended quantity', money(selectedProduct.recommended_order_qty)],
                ['Inventory position', money(selectedProduct.inventory_position)],
                ['Days since last sale', selectedProduct.days_since_last_sale ?? 'N/A'],
              ].map(([label, value]) => <div className="detail-item" key={label}><span>{label}</span><strong>{value}</strong></div>)}
            </div>
            <div className="assumption-note"><ShieldCheck size={16} /><span>Planning uses simulated inventory inputs and net-adjusted recorded sales. It is not a live warehouse feed.</span></div>
          </section>
        </div>
      )}
    </div>
  )
}

export default App
