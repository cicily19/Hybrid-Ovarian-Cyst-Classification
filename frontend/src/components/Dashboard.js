import React, { useEffect, useState } from 'react';
import { useNavigate, Link, useLocation } from 'react-router-dom';
import './Dashboard.css';
import ReportForm from './ReportForm';
import StatCard from './StatCard';
import StatsGrid from './StatsGrid';
import EnhancedCasesTable from './EnhancedCasesTable';
import SkeletonLoader from './SkeletonLoader';
import EmptyState from './EmptyState';
import {
  FaFileMedical,
  FaCalendar,
  FaChartLine,
  FaClock,
  FaCheckCircle,
  FaExclamationTriangle,
  FaHome,
  FaHistory,
  FaUser,
  FaBars,
  FaTimes,
  FaUpload,
  FaLayerGroup
} from 'react-icons/fa';

const Dashboard = () => {
  const [cases, setCases] = useState([]);
  const [profile, setProfile] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });
  const [searchTerm, setSearchTerm] = useState('');
  const [filterClass, setFilterClass] = useState('all');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    const token = localStorage.getItem('access_token');
    if (!token) {
      navigate('/login');
      return;
    }
    fetchData(token);
  }, [navigate]);

  const fetchData = async (token) => {
    try {
      // Fetch profile
      const profileRes = await fetch('http://localhost:8000/profile', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (profileRes.ok) {
        const profileData = await profileRes.json();
        setProfile(profileData);
      }

      // Fetch cases
      const casesRes = await fetch('http://localhost:8000/cases?limit=10', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (casesRes.ok) {
        const casesData = await casesRes.json();
        setCases(casesData.cases || []);
      }

      // Fetch stats
      await fetchStats(token);
    } catch (err) {
      console.error('Error fetching data:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchStats = async (token) => {
    try {
      const res = await fetch('http://localhost:8000/dashboard/stats', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) {
        const statsData = await res.json();
        setStats(statsData);
      }
    } catch (err) {
      console.error('Error fetching stats:', err);
    }
  };

  // Filter and search cases
  const filteredAndSortedCases = cases.filter(c => {
    const matchesSearch = !searchTerm || 
      (c.patient_name && c.patient_name.toLowerCase().includes(searchTerm.toLowerCase()));
    
    const matchesFilter = filterClass === 'all' ||
      (filterClass === 'simple' && (c.prediction_label === 0 || c.predicted_class === 'Simple')) ||
      (filterClass === 'complex' && (c.prediction_label === 1 || c.predicted_class === 'Complex')) ||
      (filterClass === 'pending' && !c.predicted_class && c.prediction_label === null);
    
    return matchesSearch && matchesFilter;
  });

  if (loading) {
    return (
      <div className="dashboard-container">
        <nav className="sidebar">
          <div className="sidebar-title">Ovarian Cyst Classifier</div>
          <ul className="sidebar-links">
            <li><Link to="/dashboard" style={{color: '#fff', textDecoration: 'none'}}>Dashboard</Link></li>
            <li><Link to="/history" style={{color: '#fff', textDecoration: 'none'}}>Case History</Link></li>
            <li><Link to="/insights" style={{color: '#fff', textDecoration: 'none'}}>Insights</Link></li>
            <li><Link to="/profile" style={{color: '#fff', textDecoration: 'none'}}>Profile</Link></li>
          </ul>
        </nav>
        <main className="main-content">
          <SkeletonLoader type="grid" rows={4} />
          <SkeletonLoader type="table" rows={5} columns={7} />
        </main>
      </div>
    );
  }

  const isActive = (path) => location.pathname === path;

  const sidebarLinks = [
    { path: '/dashboard', label: 'Dashboard', icon: FaHome },
    { path: '/history', label: 'Case History', icon: FaHistory },
    { path: '/batches', label: 'Batch Processing', icon: FaLayerGroup },
    { path: '/insights', label: 'Insights', icon: FaChartLine },
    { path: '/profile', label: 'Profile', icon: FaUser },
  ];

  const handleSidebarLinkClick = () => {
    // Close sidebar on mobile after navigation
    if (window.innerWidth < 900) {
      setSidebarOpen(false);
    }
  };

  return (
    <div className="dashboard-container">
      {/* Sidebar Navigation */}
      <nav className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <div className="sidebar-brand">
            <FaChartLine className="sidebar-logo-icon" />
            <span className="sidebar-title">Ovarian Cyst AI</span>
          </div>
          <button 
            className="sidebar-toggle"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            aria-label="Toggle sidebar"
          >
            {sidebarOpen ? <FaTimes /> : <FaBars />}
          </button>
        </div>
        <ul className="sidebar-links">
          {sidebarLinks.map((link) => {
            const Icon = link.icon;
            return (
              <li key={link.path}>
                <Link 
                  to={link.path} 
                  className={`sidebar-link ${isActive(link.path) ? 'active' : ''}`}
                  onClick={() => handleSidebarLinkClick(link)}
                >
                  <Icon className="sidebar-link-icon" />
                  <span className="sidebar-link-text">{link.label}</span>
                </Link>
              </li>
            );
          })}
          <li>
            <a
              href="#upload"
              className="sidebar-link"
              onClick={(e) => {
                e.preventDefault();
                handleSidebarLinkClick();
                const uploadSection = document.getElementById('upload');
                if (uploadSection) {
                  uploadSection.scrollIntoView({ behavior: 'smooth' });
                }
              }}
            >
              <FaUpload className="sidebar-link-icon" />
              <span className="sidebar-link-text">Upload Scan</span>
            </a>
          </li>
        </ul>
      </nav>

      {/* Sidebar Overlay for Mobile */}
      {sidebarOpen && (
        <div 
          className="sidebar-overlay"
          onClick={() => setSidebarOpen(false)}
          aria-label="Close sidebar"
        />
      )}

      {/* Mobile Menu Button */}
      <button 
        className="mobile-sidebar-toggle"
        onClick={() => setSidebarOpen(!sidebarOpen)}
        aria-label="Toggle sidebar"
      >
        <FaBars />
      </button>

      {/* Main Content */}
      <main className="main-content">
        {/* Welcome Section */}
        <section className="welcome-section">
          <div className="welcome-header">
            <h2>Welcome back, {profile?.name || 'User'}!</h2>
            <p className="welcome-subtitle">Your ovarian cyst classification dashboard</p>
            {stats?.last_case_date && (
              <p className="welcome-meta">
                Last active: {new Date(stats.last_case_date).toLocaleDateString('en-US', { 
                  month: 'short', 
                  day: 'numeric', 
                  year: 'numeric' 
                })}
              </p>
            )}
          </div>
          
          {/* Quick Actions */}
          <div className="quick-actions">
            <button
              onClick={() => document.getElementById('upload')?.scrollIntoView({ behavior: 'smooth' })}
              className="btn-primary"
            >
              📤 Upload New Scan
            </button>
            <Link to="/history" className="btn-secondary">View All Cases</Link>
            <Link to="/insights" className="btn-secondary">View Analytics</Link>
          </div>
        </section>

        {/* Statistics Cards */}
        {stats && (
          <StatsGrid>
            <StatCard
              icon={FaFileMedical}
              label="Total Cases"
              value={stats.total_cases}
              color="blue"
            />
            <StatCard
              icon={FaCalendar}
              label="This Month"
              value={stats.cases_this_month}
              color="green"
            />
            <StatCard
              icon={FaChartLine}
              label="Avg. Confidence"
              value={`${stats.avg_confidence}%`}
              color="yellow"
              showProgress
            />
            {stats.pending_cases > 0 && (
              <StatCard
                icon={FaClock}
                label="Pending"
                value={stats.pending_cases}
                color="orange"
              />
            )}
          </StatsGrid>
        )}

        {/* Classification Breakdown */}
        {stats && (
          <section className="classification-breakdown">
            <h3>Classification Overview</h3>
            <div className="breakdown-cards">
              <div className="breakdown-card simple">
                <FaCheckCircle className="breakdown-icon" />
                <div className="breakdown-content">
                  <div className="breakdown-label">Simple Cases</div>
                  <div className="breakdown-value">{stats.simple_cases}</div>
                  <div className="breakdown-percentage">
                    {stats.total_cases > 0
                      ? ((stats.simple_cases / stats.total_cases) * 100).toFixed(1)
                      : 0}%
                  </div>
                </div>
              </div>
              <div className="breakdown-card complex">
                <FaExclamationTriangle className="breakdown-icon" />
                <div className="breakdown-content">
                  <div className="breakdown-label">Complex Cases</div>
                  <div className="breakdown-value">{stats.complex_cases}</div>
                  <div className="breakdown-percentage">
                    {stats.total_cases > 0
                      ? ((stats.complex_cases / stats.total_cases) * 100).toFixed(1)
                      : 0}%
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* Model Performance Panel */}
        <section className="performance-panel">
          <h3>Model Performance</h3>
          <div className="metrics">
            <div className="metric">
              <span className="metric-label">Accuracy</span>
              <span className="metric-value">0.95</span>
            </div>
            <div className="metric">
              <span className="metric-label">ROC AUC</span>
              <span className="metric-value">0.99</span>
            </div>
          </div>
        </section>

        {/* Recent Cases Table */}
        <section className="recent-cases">
          <div className="section-header">
            <h3>Recent Cases</h3>
            <div className="table-controls">
              <input
                type="text"
                placeholder="Search by patient name..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="search-input"
              />
              <select
                value={filterClass}
                onChange={(e) => setFilterClass(e.target.value)}
                className="filter-select"
              >
                <option value="all">All Types</option>
                <option value="simple">Simple</option>
                <option value="complex">Complex</option>
                <option value="pending">Pending</option>
              </select>
            </div>
          </div>

          {loading ? (
            <SkeletonLoader type="table" rows={5} columns={7} />
          ) : filteredAndSortedCases.length === 0 ? (
            <EmptyState
              icon={FaFileMedical}
              title="No cases yet"
              message="Upload your first case to get started with ovarian cyst classification."
              actionLabel="Upload New Case"
              onAction={() => document.getElementById('upload')?.scrollIntoView({ behavior: 'smooth' })}
            />
          ) : (
            <>
              <EnhancedCasesTable
                cases={filteredAndSortedCases}
                onRowClick={(caseId) => navigate(`/case/${caseId}`)}
                sortConfig={sortConfig}
                onSort={setSortConfig}
              />
              {cases.length >= 10 && (
                <div className="view-all-link">
                  <Link to="/history">View All Cases →</Link>
                </div>
              )}
            </>
          )}
        </section>

        {/* Upload & Prediction Section */}
        <section className="upload-section" id="upload">
          <ReportForm />
        </section>
      </main>
    </div>
  );
};

export default Dashboard;
