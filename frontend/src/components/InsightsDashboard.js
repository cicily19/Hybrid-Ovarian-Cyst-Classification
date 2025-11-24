import React, { useState, useEffect } from 'react';
import './InsightsDashboard.css';

const InsightsDashboard = () => {
  const [insights, setInsights] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchInsights();
  }, []);

  const fetchInsights = async () => {
    try {
      const response = await fetch('http://localhost:8000/insights');
      if (response.ok) {
        const data = await response.json();
        setInsights(data);
      } else {
        setError('Failed to load insights');
      }
    } catch (err) {
      console.error('Error fetching insights:', err);
      setError('Error connecting to server');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="insights-container">
        <div className="loading-state">Loading insights...</div>
      </div>
    );
  }

  if (error || !insights) {
    return (
      <div className="insights-container">
        <div className="error-state">{error || 'Failed to load insights'}</div>
      </div>
    );
  }

  const { metrics, dataset, model_info, explainability, verification, clinical, charts } = insights;

  const formatNumber = (num) => {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toString();
  };

  return (
    <div className="insights-container">
      <div className="insights-content">
        <div className="insights-header">
          <h1>Model Performance & Insights</h1>
          <p className="subtitle">Comprehensive analysis of the ConvNeXt-Tiny ovarian cyst classification model</p>
        </div>

        {/* Performance Metrics */}
        <section className="metrics-section">
          <h2>Performance Metrics</h2>
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-label">Accuracy</div>
              <div className="metric-value">{(metrics.accuracy * 100).toFixed(2)}%</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">ROC-AUC</div>
              <div className="metric-value">{(metrics.roc_auc * 100).toFixed(2)}%</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Precision</div>
              <div className="metric-value">{(metrics.precision_macro * 100).toFixed(2)}%</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Recall</div>
              <div className="metric-value">{(metrics.recall_macro * 100).toFixed(2)}%</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">F1-Score</div>
              <div className="metric-value">{(metrics.f1_macro * 100).toFixed(2)}%</div>
            </div>
          </div>

          {/* Per-Class Metrics */}
          <div className="per-class-metrics">
            <h3>Per-Class Performance</h3>
            <div className="per-class-grid">
              <div className="per-class-card">
                <h4>Complex Cyst</h4>
                <div className="per-class-stats">
                  <div><strong>Precision:</strong> {(metrics.per_class.Complex.precision * 100).toFixed(0)}%</div>
                  <div><strong>Recall:</strong> {(metrics.per_class.Complex.recall * 100).toFixed(0)}%</div>
                  <div><strong>F1-Score:</strong> {(metrics.per_class.Complex.f1 * 100).toFixed(0)}%</div>
                  <div><strong>Support:</strong> {metrics.per_class.Complex.support}</div>
                </div>
              </div>
              <div className="per-class-card">
                <h4>Simple Cyst</h4>
                <div className="per-class-stats">
                  <div><strong>Precision:</strong> {(metrics.per_class.Simple.precision * 100).toFixed(0)}%</div>
                  <div><strong>Recall:</strong> {(metrics.per_class.Simple.recall * 100).toFixed(0)}%</div>
                  <div><strong>F1-Score:</strong> {(metrics.per_class.Simple.f1 * 100).toFixed(0)}%</div>
                  <div><strong>Support:</strong> {metrics.per_class.Simple.support}</div>
                </div>
              </div>
            </div>
            <div className="test-summary">
              <p><strong>Test Set:</strong> {metrics.test_size} images | <strong>Misclassified:</strong> {metrics.misclassified} ({((metrics.misclassified / metrics.test_size) * 100).toFixed(1)}%)</p>
            </div>
          </div>
        </section>

        {/* Dataset Information */}
        <section className="dataset-section">
          <h2>Dataset Information</h2>
          <div className="dataset-grid">
            <div className="dataset-card">
              <h3>Training Set</h3>
              <div className="dataset-stats">
                <div><strong>Complex:</strong> {dataset.train_complex} images</div>
                <div><strong>Simple:</strong> {dataset.train_simple} images</div>
                <div><strong>Total:</strong> {dataset.train_total} images</div>
              </div>
            </div>
            <div className="dataset-card">
              <h3>Test Set</h3>
              <div className="dataset-stats">
                <div><strong>Complex:</strong> {dataset.test_complex} images</div>
                <div><strong>Simple:</strong> {dataset.test_simple} images</div>
                <div><strong>Total:</strong> {dataset.test_total} images</div>
              </div>
            </div>
          </div>
        </section>

        {/* Model Information */}
        <section className="model-info-section">
          <h2>Model Architecture & Configuration</h2>
          <div className="model-info-grid">
            <div className="info-card">
              <h3>Architecture</h3>
              <p>{model_info.architecture}</p>
              <div className="info-details">
                <div><strong>Input Shape:</strong> {model_info.input_shape.join(' × ')}</div>
                <div><strong>Total Parameters:</strong> {formatNumber(model_info.total_params)}</div>
                <div><strong>Trainable:</strong> {formatNumber(model_info.trainable_params)}</div>
                <div><strong>Non-trainable:</strong> {formatNumber(model_info.non_trainable_params)}</div>
              </div>
            </div>
            <div className="info-card">
              <h3>Training Configuration</h3>
              <div className="info-details">
                <div><strong>Loss Function:</strong> {model_info.loss}</div>
                <div><strong>Optimizer:</strong> {model_info.optimizer}</div>
                <div><strong>Metrics:</strong> {model_info.metrics.join(', ')}</div>
              </div>
              <div className="preprocessing-info">
                <strong>Preprocessing:</strong> {model_info.preprocessing}
              </div>
            </div>
            <div className="info-card">
              <h3>Data Augmentation</h3>
              <ul className="augmentation-list">
                {model_info.data_augmentation.map((aug, idx) => (
                  <li key={idx}>{aug}</li>
                ))}
              </ul>
            </div>
          </div>
        </section>

        {/* Explainability */}
        <section className="explainability-section">
          <h2>Explainability Methods</h2>
          <div className="explainability-content">
            <div className="method-card">
              <h3>Occlusion-Based Sensitivity Maps</h3>
              <p><strong>Status:</strong> Deployed in production</p>
              <p>{explainability.occlusion.description}</p>
              <p><strong>Patch Size:</strong> {explainability.occlusion.patch_size}</p>
            </div>
            <div className="method-card">
              <h3>SHAP (GradientExplainer)</h3>
              <p><strong>Status:</strong> Offline research</p>
              <p>{explainability.shap.description}</p>
              <p><strong>Usage:</strong> {explainability.shap.usage}</p>
            </div>
          </div>
        </section>

        {/* Verification */}
        <section className="verification-section">
          <h2>Ultrasound Verification</h2>
          <div className="verification-card">
            <p><strong>Type:</strong> {verification.type}</p>
            <p><strong>Score Range:</strong> {verification.score_range[0]} - {verification.score_range[1]}</p>
            <p><strong>Threshold:</strong> ≥ {verification.threshold}</p>
            <p><strong>Description:</strong> {verification.description}</p>
          </div>
        </section>

        {/* Visual Diagnostics */}
        <section className="charts-section">
          <h2>Visual Diagnostics</h2>
          <div className="charts-grid">
            <div className="chart-item">
              <img 
                src={`http://localhost:8000${charts.confusion_matrix}`} 
                alt="Confusion Matrix"
                onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'block'; }}
              />
              <div className="chart-placeholder" style={{display: 'none'}}>Confusion Matrix (image not available)</div>
              <p className="chart-caption">Confusion Matrix</p>
            </div>
            <div className="chart-item">
              <img 
                src={`http://localhost:8000${charts.roc_curve}`} 
                alt="ROC Curve"
                onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'block'; }}
              />
              <div className="chart-placeholder" style={{display: 'none'}}>ROC Curve (image not available)</div>
              <p className="chart-caption">ROC Curve (AUC = {(metrics.roc_auc * 100).toFixed(2)}%)</p>
            </div>
            <div className="chart-item">
              <img 
                src={`http://localhost:8000${charts.pr_curve}`} 
                alt="Precision-Recall Curve"
                onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'block'; }}
              />
              <div className="chart-placeholder" style={{display: 'none'}}>Precision-Recall Curve (image not available)</div>
              <p className="chart-caption">Precision-Recall Curve</p>
            </div>
            <div className="chart-item">
              <img 
                src={`http://localhost:8000${charts.train_val_loss}`} 
                alt="Training vs Validation Loss"
                onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'block'; }}
              />
              <div className="chart-placeholder" style={{display: 'none'}}>Training Loss Curve (image not available)</div>
              <p className="chart-caption">Training vs Validation Loss</p>
            </div>
            <div className="chart-item">
              <img 
                src={`http://localhost:8000${charts.train_val_roc_auc}`} 
                alt="Training vs Validation ROC-AUC"
                onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'block'; }}
              />
              <div className="chart-placeholder" style={{display: 'none'}}>ROC-AUC Curve (image not available)</div>
              <p className="chart-caption">Training vs Validation ROC-AUC</p>
            </div>
            <div className="chart-item">
              <img 
                src={`http://localhost:8000${charts.hist_simple}`} 
                alt="Pixel Intensity Histogram - Simple"
                onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'block'; }}
              />
              <div className="chart-placeholder" style={{display: 'none'}}>Simple Histogram (image not available)</div>
              <p className="chart-caption">Pixel Intensity Histogram — Simple</p>
            </div>
            <div className="chart-item">
              <img 
                src={`http://localhost:8000${charts.hist_complex}`} 
                alt="Pixel Intensity Histogram - Complex"
                onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'block'; }}
              />
              <div className="chart-placeholder" style={{display: 'none'}}>Complex Histogram (image not available)</div>
              <p className="chart-caption">Pixel Intensity Histogram — Complex</p>
            </div>
          </div>
        </section>

        {/* Clinical Disclaimer */}
        <section className="clinical-section">
          <div className="clinical-warning">
            <h3>⚠️ Important Clinical Disclaimer</h3>
            <p><strong>Intended Use:</strong> {clinical.intended_use}</p>
            <p><strong>Limitations:</strong> {clinical.limitations}</p>
            <p className="disclaimer-text">{clinical.disclaimer}</p>
          </div>
        </section>
      </div>
    </div>
  );
};

export default InsightsDashboard;
