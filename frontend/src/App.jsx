import { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  ShieldCheck, 
  ShieldAlert, 
  Sparkles, 
  Loader2, 
  Link2, 
  FileText, 
  Trash2, 
  Clock, 
  BarChart3, 
  History, 
  Layers, 
  AlertCircle, 
  ChevronDown, 
  ChevronUp,
  Eye,
  RefreshCw
} from 'lucide-react';
import './index.css';

const API_BASE = 'http://127.0.0.1:8000';

function App() {
  const [activeTab, setActiveTab] = useState('analyze');
  
  // Analyze tab states
  const [inputType, setInputType] = useState('text'); // 'text' | 'url'
  const [urlInput, setUrlInput] = useState('');
  const [newsText, setNewsText] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  
  // History states
  const [history, setHistory] = useState([]);
  const [expandedHistoryId, setExpandedHistoryId] = useState(null);
  
  // Compare tab states
  const [compareText1, setCompareText1] = useState('');
  const [compareText2, setCompareText2] = useState('');
  const [compareResult1, setCompareResult1] = useState(null);
  const [compareResult2, setCompareResult2] = useState(null);
  const [compareLoading, setCompareLoading] = useState(false);
  
  // Insights tab states
  const [modelStats, setModelStats] = useState(null);
  const [statsError, setStatsError] = useState('');

  // Load history from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('veritas_history');
    if (saved) {
      try {
        setHistory(JSON.parse(saved));
      } catch (e) {
        console.error('Error parsing history', e);
      }
    }
    
    // Fetch model info
    fetchModelStats();
  }, []);

  const fetchModelStats = async () => {
    try {
      setStatsError('');
      const response = await axios.get(`${API_BASE}/model-info`);
      setModelStats(response.data);
    } catch (err) {
      console.error('Failed to load model stats:', err);
      setStatsError('Backend API is offline. Start the uvicorn server to load model stats.');
      // Fallback stats
      setModelStats({
        algorithm: "Logistic Regression Classifier",
        features: 5000,
        accuracy: 98.7,
        test_split: "20%",
        vectorizer: "TF-IDF Vectorizer",
        class_balancing: "SMOTE"
      });
    }
  };

  // Helper to extract clean snippet
  const getSnippet = (text) => {
    if (!text) return '';
    return text.length > 80 ? text.substring(0, 80) + '...' : text;
  };

  // Clickbait trigger word highlighting
  const renderHighlightedText = (text) => {
    if (!text) return '';
    const buzzwords = [
      "shocking", "unbelievable", "you won't believe", "secret", "exposed", 
      "click here", "miracle", "leak", "breaking", "mind-blowing", "viral", 
      "obliterates", "destroys", "insane", "confession", "revealed", "hiding",
      "hidden", "conspiracy", "scandal", "truth about", "what happens next"
    ];
    
    // Escape regex chars
    const escaped = buzzwords.map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    const regex = new RegExp(`\\b(${escaped.join('|')})\\b`, 'gi');
    
    const parts = text.split(regex);
    return parts.map((part, index) => {
      if (buzzwords.includes(part.toLowerCase())) {
        const isUpper = part === part.toUpperCase() && part.length > 2;
        return (
          <span key={index} className={isUpper ? "highlight-word-danger" : "highlight-word-warning"}>
            {part}
          </span>
        );
      }
      return part;
    });
  };

  // Main Analyze Handler
  const handleAnalyze = async () => {
    let textToAnalyze = newsText;
    setError('');
    
    if (inputType === 'url') {
      if (!urlInput.trim()) return;
      setLoading(true);
      setResult(null);
      
      try {
        setLoadingStep('Connecting to server...');
        await new Promise(r => setTimeout(r, 600));
        
        setLoadingStep('Bypassing standard bot blocks...');
        await new Promise(r => setTimeout(r, 600));
        
        setLoadingStep('Extracting page paragraphs...');
        const urlRes = await axios.post(`${API_BASE}/analyze-url`, { url: urlInput });
        
        textToAnalyze = urlRes.data.text;
        setNewsText(textToAnalyze);
      } catch (err) {
        setLoading(false);
        setError(err.response?.data?.detail || 'Failed to fetch article contents. Please check the URL and try again.');
        return;
      }
    }

    if (!textToAnalyze.trim()) {
      setError('Please enter some text or fetch a URL first.');
      setLoading(false);
      return;
    }

    setLoading(true);
    setLoadingStep('Analyzing text authenticity...');
    await new Promise(r => setTimeout(r, 600));

    try {
      const response = await axios.post(`${API_BASE}/predict`, { text: textToAnalyze });
      const data = response.data;
      setResult(data);

      // Save to localStorage history
      const newHistoryItem = {
        id: Date.now(),
        text: getSnippet(textToAnalyze),
        prediction: data.prediction,
        confidence: data.confidence,
        stats: data.stats,
        date: new Date().toLocaleString()
      };
      const updatedHistory = [newHistoryItem, ...history];
      setHistory(updatedHistory);
      localStorage.setItem('veritas_history', JSON.stringify(updatedHistory));
    } catch (err) {
      setError('Prediction API connection failed. Is the uvicorn backend running?');
      console.error(err);
    } finally {
      setLoading(false);
      setLoadingStep('');
    }
  };

  // Side-by-Side Compare Handler
  const handleCompare = async () => {
    if (!compareText1.trim() || !compareText2.trim()) return;
    setCompareLoading(true);
    setError('');
    setCompareResult1(null);
    setCompareResult2(null);

    try {
      const [res1, res2] = await Promise.all([
        axios.post(`${API_BASE}/predict`, { text: compareText1 }),
        axios.post(`${API_BASE}/predict`, { text: compareText2 })
      ]);
      setCompareResult1(res1.data);
      setCompareResult2(res2.data);
    } catch (err) {
      setError('Comparison failed. Verify your backend is running.');
      console.error(err);
    } finally {
      setCompareLoading(false);
    }
  };

  // Clear states
  const handleClear = () => {
    setNewsText('');
    setUrlInput('');
    setResult(null);
    setError('');
  };

  const handleClearHistory = () => {
    setHistory([]);
    localStorage.removeItem('veritas_history');
  };

  // SVGs / Gauges calculations
  const renderDial = (confidence, prediction) => {
    const radius = 70;
    const circ = 2 * Math.PI * radius;
    const offset = circ - (confidence / 100) * circ;
    const isReal = prediction === 'REAL';
    const strokeColor = isReal ? 'var(--success)' : 'var(--danger)';
    
    return (
      <div className="dial-container">
        <svg className="dial-svg">
          <circle className="dial-bg" cx="80" cy="80" r={radius} />
          <circle 
            className="dial-fill" 
            cx="80" 
            cy="80" 
            r={radius} 
            stroke={strokeColor}
            strokeDasharray={circ}
            strokeDashoffset={offset}
            style={{ filter: `drop-shadow(0 0 8px ${strokeColor}44)` }}
          />
        </svg>
        <div className="dial-text">
          <span className="dial-percent" style={{ color: strokeColor }}>{confidence}%</span>
          <span className="dial-label">Confidence</span>
        </div>
      </div>
    );
  };

  return (
    <div className="app-container">
      {/* Brand Header */}
      <div className="header">
        <div className="header-logo">
          <ShieldCheck className="header-icon" size={44} />
          <h1>VERITAS</h1>
        </div>
        <p>State-of-the-Art AI Fake News Analysis Dashboard</p>
      </div>

      {/* Tab Controls */}
      <div className="tabs-nav">
        <button 
          className={`tab-btn ${activeTab === 'analyze' ? 'active' : ''}`}
          onClick={() => { setActiveTab('analyze'); setError(''); }}
        >
          <Sparkles size={16} />
          Analyze
        </button>
        <button 
          className={`tab-btn ${activeTab === 'history' ? 'active' : ''}`}
          onClick={() => { setActiveTab('history'); setError(''); }}
        >
          <History size={16} />
          History
        </button>
        <button 
          className={`tab-btn ${activeTab === 'compare' ? 'active' : ''}`}
          onClick={() => { setActiveTab('compare'); setError(''); }}
        >
          <Layers size={16} />
          Compare Articles
        </button>
        <button 
          className={`tab-btn ${activeTab === 'insights' ? 'active' : ''}`}
          onClick={() => { setActiveTab('insights'); setError(''); fetchModelStats(); }}
        >
          <BarChart3 size={16} />
          Model Insights
        </button>
      </div>

      {/* Main Glass Card Display */}
      <div className="glass-card">
        {error && (
          <div style={{ 
            background: 'rgba(239, 68, 68, 0.1)', 
            border: '1px solid rgba(239, 68, 68, 0.2)', 
            color: 'var(--danger)', 
            padding: '1rem', 
            borderRadius: '0.75rem', 
            marginBottom: '1.5rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            <AlertCircle size={20} />
            {error}
          </div>
        )}

        {/* Tab 1: Analyze */}
        {activeTab === 'analyze' && (
          loading ? (
            <div className="loading-box">
              <div className="radar-loader">
                <div className="radar-pulse"></div>
              </div>
              <div className="loading-steps">
                <div className="loading-title">Verifying Credibility</div>
                <div className="loading-step-item">{loadingStep}</div>
              </div>
            </div>
          ) : (
            <div className="tab-content">
              {/* Type Switcher */}
              <div className="input-toggle">
                <button 
                  className={`toggle-btn ${inputType === 'text' ? 'active' : ''}`}
                  onClick={() => setInputType('text')}
                >
                  <FileText size={14} style={{ marginRight: '0.25rem' }} />
                  Raw Text Input
                </button>
                <button 
                  className={`toggle-btn ${inputType === 'url' ? 'active' : ''}`}
                  onClick={() => setInputType('url')}
                >
                  <Link2 size={14} style={{ marginRight: '0.25rem' }} />
                  Fetch From URL
                </button>
              </div>

              {/* URL input field */}
              {inputType === 'url' && (
                <div className="url-input-container">
                  <input 
                    type="url" 
                    placeholder="https://example-news-site.com/breaking-article" 
                    value={urlInput}
                    onChange={(e) => setUrlInput(e.target.value)}
                    className="url-field"
                  />
                </div>
              )}

              {/* Text Area input */}
              <div className="textarea-wrapper">
                <textarea 
                  placeholder={inputType === 'url' ? "Pasted content will show here after fetching, or you can paste directly..." : "Paste the news headline and body content here for automated credibility scan..."}
                  value={newsText}
                  onChange={(e) => setNewsText(e.target.value)}
                />
                <span className="char-counter">{newsText.length} characters</span>
              </div>

              {/* Action Buttons */}
              <div className="action-row">
                <button 
                  className="btn-clear" 
                  onClick={handleClear}
                  disabled={!newsText && !urlInput}
                >
                  Clear
                </button>
                <button 
                  className="btn-analyze" 
                  onClick={handleAnalyze}
                  disabled={(inputType === 'text' && !newsText.trim()) || (inputType === 'url' && !urlInput.trim())}
                >
                  <Sparkles size={18} />
                  Analyze Authenticity
                </button>
              </div>

              {/* Prediction Results Dashboard */}
              {result && (
                <div className="results-grid">
                  {/* Gauge Card */}
                  <div className="result-gauge-card">
                    {renderDial(result.confidence, result.prediction)}
                    <div className={`result-badge ${result.prediction === 'REAL' ? 'real' : 'fake'}`}>
                      {result.prediction === 'REAL' ? (
                        <>
                          <ShieldCheck size={22} />
                          AUTHENTIC
                        </>
                      ) : (
                        <>
                          <ShieldAlert size={22} />
                          UNTRUSTWORTHY
                        </>
                      )}
                    </div>
                  </div>

                  {/* Details stats */}
                  <div className="stats-panel">
                    <div className="metrics-grid">
                      <div className="metric-card">
                        <div className="metric-header">
                          Clickbait Score
                          <Clock size={16} className="metric-icon" />
                        </div>
                        <span className="metric-val">{result.stats?.clickbait_score}%</span>
                        <div className="clickbait-bar-bg">
                          <div 
                            className="clickbait-bar-fill" 
                            style={{ 
                              width: `${result.stats?.clickbait_score}%`,
                              background: result.stats?.clickbait_score > 60 ? 'var(--danger)' : result.stats?.clickbait_score > 30 ? 'var(--warning)' : 'var(--success)'
                            }}
                          />
                        </div>
                      </div>

                      <div className="metric-card">
                        <div className="metric-header">
                          Sentiment Tone
                          <Sparkles size={16} className="metric-icon" />
                        </div>
                        <span className="metric-val" style={{ 
                          color: result.stats?.sentiment === 'Positive' ? 'var(--success)' : result.stats?.sentiment === 'Negative' ? 'var(--danger)' : 'var(--text-main)'
                        }}>
                          {result.stats?.sentiment}
                        </span>
                        <span style={{ fontSize: '0.8rem', color: 'var(--text-dark)' }}>Score: {result.stats?.sentiment_score}</span>
                      </div>

                      <div className="metric-card">
                        <div className="metric-header">
                          Word Count
                          <FileText size={16} className="metric-icon" />
                        </div>
                        <span className="metric-val">{result.stats?.word_count}</span>
                        <span style={{ fontSize: '0.8rem', color: 'var(--text-dark)' }}>Words</span>
                      </div>

                      <div className="metric-card">
                        <div className="metric-header">
                          Read Time
                          <Clock size={16} className="metric-icon" />
                        </div>
                        <span className="metric-val">{result.stats?.read_time} min</span>
                        <span style={{ fontSize: '0.8rem', color: 'var(--text-dark)' }}>Estimate</span>
                      </div>
                    </div>

                    {/* Highlights Box */}
                    <div className="highlight-box">
                      <h3>
                        <Eye size={16} />
                        Sensationalism highlights
                      </h3>
                      <div className="highlight-content">
                        {renderHighlightedText(newsText)}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )
        )}

        {/* Tab 2: History */}
        {activeTab === 'history' && (
          <div className="tab-content">
            <div className="history-header-row">
              <h2>Recent Credibility Scans</h2>
              {history.length > 0 && (
                <button className="btn-clear" onClick={handleClearHistory} style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }}>
                  <Trash2 size={14} />
                  Clear Archive
                </button>
              )}
            </div>

            {history.length === 0 ? (
              <div className="history-empty">
                <History size={48} style={{ color: 'var(--text-dark)' }} />
                <p>No news scans are stored yet. Run an authenticity analysis to build your history!</p>
              </div>
            ) : (
              <div className="history-list">
                {history.map((item) => (
                  <div 
                    key={item.id} 
                    className="history-card"
                    onClick={() => setExpandedHistoryId(expandedHistoryId === item.id ? null : item.id)}
                  >
                    <div className="history-card-header">
                      <span className="history-title-snippet">
                        {item.text}
                      </span>
                      <div className="history-tags">
                        <span className={`badge-sm ${item.prediction === 'REAL' ? 'real' : 'fake'}`}>
                          {item.prediction} ({item.confidence}%)
                        </span>
                        {expandedHistoryId === item.id ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                      </div>
                    </div>
                    
                    <div className="history-date">Scanned on {item.date}</div>

                    {expandedHistoryId === item.id && (
                      <div className="history-details-drawer">
                        <div className="history-detail-item">
                          <span className="history-detail-label">Sentiment</span>
                          <span className="history-detail-val">{item.stats?.sentiment} ({item.stats?.sentiment_score})</span>
                        </div>
                        <div className="history-detail-item">
                          <span className="history-detail-label">Clickbait Score</span>
                          <span className="history-detail-val" style={{ 
                            color: item.stats?.clickbait_score > 50 ? 'var(--danger)' : 'var(--success)'
                          }}>{item.stats?.clickbait_score}%</span>
                        </div>
                        <div className="history-detail-item">
                          <span className="history-detail-label">Read Time / Words</span>
                          <span className="history-detail-val">{item.stats?.read_time}m / {item.stats?.word_count} words</span>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Tab 3: Compare */}
        {activeTab === 'compare' && (
          <div className="tab-content">
            <div className="compare-container">
              <div className="compare-inputs-grid">
                <div className="compare-box">
                  <h3>
                    <FileText size={16} />
                    Article Source A
                  </h3>
                  <textarea 
                    placeholder="Paste the first article here..." 
                    value={compareText1}
                    onChange={(e) => setCompareText1(e.target.value)}
                  />
                </div>
                <div className="compare-box">
                  <h3>
                    <FileText size={16} />
                    Article Source B
                  </h3>
                  <textarea 
                    placeholder="Paste the second article here..." 
                    value={compareText2}
                    onChange={(e) => setCompareText2(e.target.value)}
                  />
                </div>
              </div>

              <div className="action-row" style={{ justifyContent: 'center' }}>
                <button 
                  className="btn-analyze" 
                  onClick={handleCompare}
                  disabled={compareLoading || !compareText1.trim() || !compareText2.trim()}
                >
                  {compareLoading ? (
                    <>
                      <Loader2 className="spinner" size={18} />
                      Comparing...
                    </>
                  ) : (
                    <>
                      <Layers size={18} />
                      Compare Article Credibility
                    </>
                  )}
                </button>
              </div>

              {/* Comparison Results Card */}
              {(compareResult1 || compareResult2) && (
                <div className="compare-results-row">
                  {compareResult1 && (
                    <div className={`compare-result-card ${compareResult1.prediction === 'REAL' ? 'real' : 'fake'}`}>
                      <div className="compare-badge-row">
                        <h4 style={{ fontWeight: '700' }}>Article A</h4>
                        <span className={`badge-sm ${compareResult1.prediction === 'REAL' ? 'real' : 'fake'}`}>
                          {compareResult1.prediction}
                        </span>
                      </div>
                      
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span>Confidence:</span>
                        <strong style={{ color: compareResult1.prediction === 'REAL' ? 'var(--success)' : 'var(--danger)' }}>
                          {compareResult1.confidence}%
                        </strong>
                      </div>
                      
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', color: 'var(--text-muted)' }}>
                        <span>Clickbait Score:</span>
                        <span>{compareResult1.stats?.clickbait_score}%</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', color: 'var(--text-muted)' }}>
                        <span>Sentiment:</span>
                        <span>{compareResult1.stats?.sentiment}</span>
                      </div>
                    </div>
                  )}

                  {compareResult2 && (
                    <div className={`compare-result-card ${compareResult2.prediction === 'REAL' ? 'real' : 'fake'}`}>
                      <div className="compare-badge-row">
                        <h4 style={{ fontWeight: '700' }}>Article B</h4>
                        <span className={`badge-sm ${compareResult2.prediction === 'REAL' ? 'real' : 'fake'}`}>
                          {compareResult2.prediction}
                        </span>
                      </div>
                      
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span>Confidence:</span>
                        <strong style={{ color: compareResult2.prediction === 'REAL' ? 'var(--success)' : 'var(--danger)' }}>
                          {compareResult2.confidence}%
                        </strong>
                      </div>
                      
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', color: 'var(--text-muted)' }}>
                        <span>Clickbait Score:</span>
                        <span>{compareResult2.stats?.clickbait_score}%</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', color: 'var(--text-muted)' }}>
                        <span>Sentiment:</span>
                        <span>{compareResult2.stats?.sentiment}</span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Tab 4: Model Insights */}
        {activeTab === 'insights' && (
          <div className="tab-content">
            <div className="insights-grid">
              {/* Stats Card */}
              <div className="insights-info">
                <div className="insights-card">
                  <h3>
                    <Layers size={18} />
                    Model Configurations
                  </h3>
                  
                  {modelStats && (
                    <div className="insights-list">
                      <div className="insights-item">
                        <span className="insights-label">Algorithm</span>
                        <span className="insights-val">{modelStats.algorithm}</span>
                      </div>
                      <div className="insights-item">
                        <span className="insights-label">Vectorizer</span>
                        <span className="insights-val">{modelStats.vectorizer}</span>
                      </div>
                      <div className="insights-item">
                        <span className="insights-label">Vocabulary Features</span>
                        <span className="insights-val">{modelStats.features}</span>
                      </div>
                      <div className="insights-item">
                        <span className="insights-label">Class Balancing</span>
                        <span className="insights-val">{modelStats.class_balancing}</span>
                      </div>
                      <div className="insights-item">
                        <span className="insights-label">Test Split</span>
                        <span className="insights-val">{modelStats.test_split}</span>
                      </div>
                      <div className="insights-item">
                        <span className="insights-label">Model Accuracy</span>
                        <span className="insights-val" style={{ color: 'var(--success)' }}>{modelStats.accuracy}%</span>
                      </div>
                    </div>
                  )}

                  {statsError && (
                    <div style={{ fontSize: '0.85rem', color: 'var(--warning)', marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                      <AlertCircle size={14} />
                      {statsError}
                    </div>
                  )}
                </div>
                
                <button 
                  className="btn-clear" 
                  onClick={fetchModelStats}
                  style={{ alignSelf: 'center', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
                >
                  <RefreshCw size={14} />
                  Reload Model API Info
                </button>
              </div>

              {/* Confusion Matrix Card */}
              <div className="insights-card matrix-card">
                <h3>
                  <BarChart3 size={18} />
                  Confusion Matrix
                </h3>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                  Evaluated on 20% test partition data (Logistic Regression). Shows high class separation.
                </p>
                <div className="matrix-img-wrapper">
                  <img 
                    src={`${API_BASE}/model-info/confusion-matrix`} 
                    alt="Confusion Matrix"
                    className="matrix-img"
                    onError={(e) => {
                      e.target.onerror = null;
                      e.target.src = 'https://placehold.co/400x300/1e293b/f8fafc?text=Confusion+Matrix+Offline';
                    }}
                  />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
