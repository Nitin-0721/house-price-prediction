import { useState } from 'react'
import './App.css'

const API_URL = 'https://estimateiq-backend.onrender.com/api/predict'

function App() {
  const [form, setForm] = useState({
    OverallQual: 5,
    GrLivArea: 1500,
    GarageCars: 2,
    GarageArea: 400,
    TotalBsmtSF: 800,
    FirstFlrSF: 1000,
    FullBath: 2,
    YearBuilt: 2000,
    YrSold: 2010,
    TotRmsAbvGrd: 6,
  })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: Number(e.target.value) })
  }

  const handleSubmit = async () => {
    setError('')
    setResult(null)
    setLoading(true)
    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Prediction failed')
      }
      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const formatUSD = (val) =>
    new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0,
    }).format(val)

  return (
    <div className="app">
      <nav className="navbar">
        <div className="nav-brand">🏡 EstimateIQ</div>
        <div className="nav-links">
          <a href="#">Home</a>
          <a href="https://estimateiq-backend.onrender.com/docs" target="_blank" className="docs">API Docs ↗</a>
        </div>
      </nav>

      <header className="hero">
        <p className="eyebrow">Ridge Regression · Ames Housing · FastAPI + React</p>
        <h1>House Price <span>Predictor</span></h1>
        <p className="hero-sub">Based on Ames Housing Dataset · R² = 0.9006 · 1166 houses trained</p>
      </header>

      <div className="stats-bar">
        <div className="stat">
          <div className="stat-val">90%</div>
          <div className="stat-label">Accuracy</div>
        </div>
        <div className="stat">
          <div className="stat-val">1,166</div>
          <div className="stat-label">Houses Trained</div>
        </div>
        <div className="stat">
          <div className="stat-val">0.9006</div>
          <div className="stat-label">R² Score</div>
        </div>
        <div className="stat">
          <div className="stat-val">10</div>
          <div className="stat-label">Features Used</div>
        </div>
      </div>

      <main className="container">
        <div className="layout">

          <div className="sidebar">
            <h2 className="card-title">🏠 Enter House Features</h2>
            <div className="divider"/>

            <div className="field">
              <label>Overall Quality <span className="val">{form.OverallQual}/10</span></label>
              <input type="range" name="OverallQual" min="1" max="10" step="1"
                value={form.OverallQual} onChange={handleChange}/>
              <div className="slider-labels"><span>1 · Poor</span><span>10 · Excellent</span></div>
            </div>

            <div className="field">
              <label>Living Area (sq ft)</label>
              <input type="number" className="text-input" name="GrLivArea"
                value={form.GrLivArea} onChange={handleChange} min="0"/>
            </div>

            <div className="field">
              <label>Garage Cars <span className="val">{form.GarageCars}</span></label>
              <input type="range" name="GarageCars" min="0" max="5" step="1"
                value={form.GarageCars} onChange={handleChange}/>
            </div>

            <div className="field">
              <label>Garage Area (sq ft)</label>
              <input type="number" className="text-input" name="GarageArea"
                value={form.GarageArea} onChange={handleChange} min="0"/>
            </div>

            <div className="field">
              <label>Basement Area (sq ft)</label>
              <input type="number" className="text-input" name="TotalBsmtSF"
                value={form.TotalBsmtSF} onChange={handleChange} min="0"/>
            </div>

            <div className="field">
              <label>1st Floor Area (sq ft)</label>
              <input type="number" className="text-input" name="FirstFlrSF"
                value={form.FirstFlrSF} onChange={handleChange} min="0"/>
            </div>

            <div className="field">
              <label>Full Bathrooms <span className="val">{form.FullBath}</span></label>
              <input type="range" name="FullBath" min="0" max="5" step="1"
                value={form.FullBath} onChange={handleChange}/>
            </div>

            <div className="field">
              <label>Year Built</label>
              <input type="number" className="text-input" name="YearBuilt"
                value={form.YearBuilt} onChange={handleChange} min="1800" max="2024"/>
            </div>

            <div className="field">
              <label>Year Sold</label>
              <input type="number" className="text-input" name="YrSold"
                value={form.YrSold} onChange={handleChange} min="2000" max="2024"/>
            </div>

            <div className="field">
              <label>Total Rooms <span className="val">{form.TotRmsAbvGrd}</span></label>
              <input type="range" name="TotRmsAbvGrd" min="1" max="20" step="1"
                value={form.TotRmsAbvGrd} onChange={handleChange}/>
            </div>

            {error && <div className="error">⚠️ {error}</div>}

            <button className="btn-predict" onClick={handleSubmit} disabled={loading}>
              {loading ? '⏳ Predicting...' : '✦ Predict Price'}
            </button>
          </div>

          <div className="main-content">
            {!result ? (
              <>
                <div className="prompt-box">
                  <div className="prompt-icon">🏡</div>
                  <h3>Ready to Estimate</h3>
                  <p>Adjust the house features in the sidebar and click Predict Price to get an instant ML-powered valuation.</p>
                </div>
                <div className="info-grid">
                  <div className="info-card">
                    <div className="ic-label">Model</div>
                    <div className="ic-val">Ridge Regression</div>
                  </div>
                  <div className="info-card">
                    <div className="ic-label">Training Data</div>
                    <div className="ic-val">1,166 Houses</div>
                  </div>
                  <div className="info-card">
                    <div className="ic-label">Accuracy</div>
                    <div className="ic-val">90% · R² = 0.9006</div>
                  </div>
                  <div className="info-card">
                    <div className="ic-label">Dataset</div>
                    <div className="ic-val">Ames Housing</div>
                  </div>
                </div>
              </>
            ) : (
              <div className="result-card">
                <p className="result-label">Estimated Market Value</p>
                <h2 className="result-price">{formatUSD(result.predicted_price)}</h2>
                <p className="result-range">
                  Confidence range: {formatUSD(result.confidence_low)} – {formatUSD(result.confidence_high)}
                </p>
                <div className="result-meta">
                  <div><strong>{form.GrLivArea.toLocaleString()}</strong><span>sq ft living</span></div>
                  <div><strong>{form.OverallQual}/10</strong><span>quality</span></div>
                  <div><strong>{form.FullBath}</strong><span>bathrooms</span></div>
                  <div><strong>{form.YearBuilt}</strong><span>year built</span></div>
                </div>
                <button className="btn-reset" onClick={() => setResult(null)}>← Try Another</button>
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="footer">
        Built with FastAPI + React · Ridge Regression · Ames Housing Dataset · by <span>Nitin</span>
      </footer>
    </div>
  )
}

export default App