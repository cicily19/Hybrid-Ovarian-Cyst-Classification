import React, { useEffect, useState, useMemo } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { 
  FaSearch, 
  FaFilter, 
  FaSort, 
  FaSortUp, 
  FaSortDown,
  FaCheckCircle,
  FaExclamationTriangle,
  FaClock,
  FaEye,
  FaFileAlt,
  FaCalendarAlt,
  FaUser,
  FaTimes
} from 'react-icons/fa';
import EnhancedCasesTable from './EnhancedCasesTable';
import SkeletonLoader from './SkeletonLoader';
import EmptyState from './EmptyState';
import './CaseHistory.css';

const CaseHistory = () => {
  const [cases, setCases] = useState([]);
  const [allCases, setAllCases] = useState([]); // Store all cases for client-side filtering
  const [loading, setLoading] = useState(true);
  const [total, setTotal] = useState(0);
  const [skip, setSkip] = useState(0);
  const [limit, setLimit] = useState(10); // Default 10 entries per page, user can adjust
  
  // Search and filter states
  const [searchTerm, setSearchTerm] = useState('');
  const [filterClass, setFilterClass] = useState('all');
  const [filterVerification, setFilterVerification] = useState('all');
  const [sortConfig, setSortConfig] = useState({ key: 'created_at', direction: 'desc' });
  const [showFilters, setShowFilters] = useState(false);
  
  const navigate = useNavigate();

  useEffect(() => {
    const token = localStorage.getItem('access_token');
    if (!token) {
      navigate('/login');
      return;
    }
    fetchAllCases(token);
  }, [navigate]);

  const fetchAllCases = async (token) => {
    try {
      setLoading(true);
      // Fetch all cases (or a large number) for client-side search/filter
      const response = await fetch(`http://localhost:8000/cases?skip=0&limit=1000`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) {
        if (response.status === 401) {
          localStorage.removeItem('access_token');
          navigate('/login');
          return;
        }
        throw new Error('Failed to fetch cases');
      }

      const data = await response.json();
      setAllCases(data.cases || []);
      setTotal(data.total || 0);
    } catch (err) {
      console.error('Error fetching cases:', err);
      setAllCases([]);
    } finally {
      setLoading(false);
    }
  };

  // Filter and sort cases
  const filteredAndSortedCases = useMemo(() => {
    let filtered = [...allCases];

    // Search filter
    if (searchTerm) {
      const searchLower = searchTerm.toLowerCase();
      filtered = filtered.filter(c => 
        (c.patient_name && c.patient_name.toLowerCase().includes(searchLower)) ||
        (c.patient_id && c.patient_id.toLowerCase().includes(searchLower)) ||
        (c.date_of_scan && c.date_of_scan.toLowerCase().includes(searchLower))
      );
    }

    // Classification filter
    if (filterClass !== 'all') {
      filtered = filtered.filter(c => {
        if (filterClass === 'simple') {
          return c.prediction_label === 0 || c.predicted_class === 'Simple';
        } else if (filterClass === 'complex') {
          return c.prediction_label === 1 || c.predicted_class === 'Complex';
        } else if (filterClass === 'pending') {
          return !c.predicted_class && c.prediction_label === null;
        }
        return true;
      });
    }

    // Verification filter
    if (filterVerification !== 'all') {
      filtered = filtered.filter(c => {
        if (filterVerification === 'passed') {
          return c.verification_score !== null && c.verification_score >= 0.65;
        } else if (filterVerification === 'failed') {
          return c.verification_score !== null && c.verification_score < 0.65;
        } else if (filterVerification === 'none') {
          return c.verification_score === null;
        }
        return true;
      });
    }

    // Sorting
    filtered.sort((a, b) => {
      let aVal, bVal;
      switch (sortConfig.key) {
        case 'patient_name':
          aVal = a.patient_name || '';
          bVal = b.patient_name || '';
          break;
        case 'age':
          aVal = a.age || 0;
          bVal = b.age || 0;
          break;
        case 'date_of_scan':
          aVal = new Date(a.date_of_scan || a.created_at || 0).getTime();
          bVal = new Date(b.date_of_scan || b.created_at || 0).getTime();
          break;
        case 'classification':
          aVal = a.predicted_class || 'Pending';
          bVal = b.predicted_class || 'Pending';
          break;
        case 'confidence':
          const aConf = a.prob_simple !== null && a.prob_complex !== null 
            ? Math.max(a.prob_simple || 0, a.prob_complex || 0) 
            : 0;
          const bConf = b.prob_simple !== null && b.prob_complex !== null 
            ? Math.max(b.prob_simple || 0, b.prob_complex || 0) 
            : 0;
          aVal = aConf;
          bVal = bConf;
          break;
        case 'verification_score':
          aVal = a.verification_score || 0;
          bVal = b.verification_score || 0;
          break;
        case 'created_at':
          aVal = new Date(a.created_at || 0).getTime();
          bVal = new Date(b.created_at || 0).getTime();
          break;
        default:
          return 0;
      }

      if (typeof aVal === 'string') {
        return sortConfig.direction === 'asc' 
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal);
      }
      return sortConfig.direction === 'asc' ? aVal - bVal : bVal - aVal;
    });

    return filtered;
  }, [allCases, searchTerm, filterClass, filterVerification, sortConfig]);

  // Pagination
  const paginatedCases = useMemo(() => {
    const start = skip;
    const end = skip + limit;
    return filteredAndSortedCases.slice(start, end);
  }, [filteredAndSortedCases, skip, limit]);

  const totalFiltered = filteredAndSortedCases.length;
  const totalPages = Math.ceil(totalFiltered / limit);
  const currentPage = Math.floor(skip / limit) + 1;

  const handleSort = (key) => {
    let direction = 'asc';
    if (sortConfig && sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
    setSkip(0); // Reset to first page when sorting
  };

  const handleLimitChange = (e) => {
    const newLimit = parseInt(e.target.value, 10);
    setLimit(newLimit);
    setSkip(0); // Reset to first page when changing limit
  };

  const clearFilters = () => {
    setSearchTerm('');
    setFilterClass('all');
    setFilterVerification('all');
    setSkip(0);
  };

  const hasActiveFilters = searchTerm || filterClass !== 'all' || filterVerification !== 'all';

  if (loading) {
    return (
      <div className="case-history-container">
        <div className="case-history-content">
          <SkeletonLoader type="table" rows={10} columns={8} />
        </div>
      </div>
    );
  }

  return (
    <div className="case-history-container">
      <div className="case-history-content">
        {/* Header */}
        <div className="case-history-header">
          <div>
            <h1 className="page-title">Case History</h1>
            <p className="page-subtitle">
              {totalFiltered === total 
                ? `Total: ${total} cases` 
                : `Showing ${totalFiltered} of ${total} cases`}
            </p>
          </div>
          <Link to="/dashboard" className="btn-primary">
            <FaFileAlt /> Upload New Case
          </Link>
        </div>

        {/* Search and Filter Bar */}
        <div className="search-filter-bar">
          <div className="search-container">
            <FaSearch className="search-icon" />
            <input
              type="text"
              placeholder="Search by patient name, ID, or date..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setSkip(0);
              }}
              className="search-input"
            />
            {searchTerm && (
              <button 
                className="clear-search-btn"
                onClick={() => setSearchTerm('')}
                aria-label="Clear search"
              >
                <FaTimes />
              </button>
            )}
          </div>

          <button
            className={`filter-toggle-btn ${showFilters ? 'active' : ''}`}
            onClick={() => setShowFilters(!showFilters)}
          >
            <FaFilter /> Filters
            {hasActiveFilters && <span className="filter-badge">{[
              searchTerm && '1',
              filterClass !== 'all' && '1',
              filterVerification !== 'all' && '1'
            ].filter(Boolean).length}</span>}
          </button>
        </div>

        {/* Filter Panel */}
        {showFilters && (
          <div className="filter-panel">
            <div className="filter-group">
              <label className="filter-label">Classification</label>
              <select
                value={filterClass}
                onChange={(e) => {
                  setFilterClass(e.target.value);
                  setSkip(0);
                }}
                className="filter-select"
              >
                <option value="all">All Types</option>
                <option value="simple">Simple</option>
                <option value="complex">Complex</option>
                <option value="pending">Pending</option>
              </select>
            </div>

            <div className="filter-group">
              <label className="filter-label">Verification Status</label>
              <select
                value={filterVerification}
                onChange={(e) => {
                  setFilterVerification(e.target.value);
                  setSkip(0);
                }}
                className="filter-select"
              >
                <option value="all">All</option>
                <option value="passed">Passed (≥0.65)</option>
                <option value="failed">Failed (&lt;0.65)</option>
                <option value="none">Not Verified</option>
              </select>
            </div>

            {hasActiveFilters && (
              <button className="clear-filters-btn" onClick={clearFilters}>
                <FaTimes /> Clear All Filters
              </button>
            )}
          </div>
        )}

        {/* Cases Table */}
        {paginatedCases.length === 0 ? (
          <EmptyState
            icon={FaFileAlt}
            title={hasActiveFilters ? "No cases match your filters" : "No cases yet"}
            message={
              hasActiveFilters
                ? "Try adjusting your search or filter criteria."
                : "Upload your first case to get started with ovarian cyst classification."
            }
            actionLabel={hasActiveFilters ? "Clear Filters" : "Upload New Case"}
            onAction={hasActiveFilters ? clearFilters : () => navigate('/dashboard')}
          />
        ) : (
          <>
            <div className="table-container">
              <EnhancedCasesTable
                cases={paginatedCases}
                onRowClick={(caseId) => navigate(`/case/${caseId}`)}
                sortConfig={sortConfig}
                onSort={handleSort}
              />
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="pagination">
                <button
                  onClick={() => setSkip(Math.max(0, skip - limit))}
                  disabled={skip === 0}
                  className="pagination-btn"
                >
                  Previous
                </button>
                
                <div className="pagination-info">
                  Page {currentPage} of {totalPages}
                  <span className="pagination-count">
                    ({skip + 1} - {Math.min(skip + limit, totalFiltered)} of {totalFiltered})
                  </span>
                </div>

                <button
                  onClick={() => setSkip(skip + limit)}
                  disabled={skip + limit >= totalFiltered}
                  className="pagination-btn"
                >
                  Next
                </button>
              </div>
            )}

            {/* Entries per page selector */}
            <div className="entries-per-page">
              <label htmlFor="entries-per-page-select" className="entries-label">
                Show:
              </label>
              <select
                id="entries-per-page-select"
                value={limit}
                onChange={handleLimitChange}
                className="entries-select"
              >
                <option value={5}>5</option>
                <option value={10}>10</option>
                <option value={20}>20</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
              </select>
              <span className="entries-text">entries per page</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default CaseHistory;
