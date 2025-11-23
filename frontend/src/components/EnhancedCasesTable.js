import React from 'react';
import { Link } from 'react-router-dom';
import { FaCheckCircle, FaClock, FaExclamationTriangle, FaSort, FaSortUp, FaSortDown } from 'react-icons/fa';
import './EnhancedCasesTable.css';

const EnhancedCasesTable = ({ cases, onRowClick, sortConfig, onSort }) => {
  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    } catch {
      return dateString;
    }
  };

  const getConfidenceScore = (caseItem) => {
    if (caseItem.prob_simple !== null && caseItem.prob_complex !== null) {
      return Math.max(caseItem.prob_simple || 0, caseItem.prob_complex || 0) * 100;
    }
    return null;
  };

  const getClassificationBadge = (caseItem) => {
    if (!caseItem.predicted_class && caseItem.prediction_label === null) {
      return { label: 'Pending', class: 'badge-pending', icon: FaClock };
    }
    if (caseItem.prediction_label === 0 || caseItem.predicted_class === 'Simple') {
      return { label: 'Simple', class: 'badge-simple', icon: FaCheckCircle };
    }
    if (caseItem.prediction_label === 1 || caseItem.predicted_class === 'Complex') {
      return { label: 'Complex', class: 'badge-complex', icon: FaExclamationTriangle };
    }
    return { label: 'Pending', class: 'badge-pending', icon: FaClock };
  };

  const handleSort = (key) => {
    let direction = 'asc';
    if (sortConfig && sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    onSort({ key, direction });
  };

  const getSortIcon = (columnKey) => {
    if (!sortConfig || sortConfig.key !== columnKey) {
      return <FaSort className="sort-icon inactive" />;
    }
    return sortConfig.direction === 'asc' 
      ? <FaSortUp className="sort-icon active" />
      : <FaSortDown className="sort-icon active" />;
  };

  const sortedCases = [...cases].sort((a, b) => {
    if (!sortConfig || !sortConfig.key) return 0;

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
      case 'date':
        aVal = new Date(a.date_of_scan || a.created_at || 0).getTime();
        bVal = new Date(b.date_of_scan || b.created_at || 0).getTime();
        break;
      case 'classification':
        aVal = a.predicted_class || 'Pending';
        bVal = b.predicted_class || 'Pending';
        break;
      case 'confidence':
        aVal = getConfidenceScore(a) || 0;
        bVal = getConfidenceScore(b) || 0;
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

  return (
    <div className="enhanced-table-container">
      <table className="enhanced-cases-table">
        <thead>
          <tr>
            <th onClick={() => handleSort('patient_name')} className="sortable">
              <div className="th-content">
                Patient Name
                {getSortIcon('patient_name')}
              </div>
            </th>
            <th onClick={() => handleSort('age')} className="sortable">
              <div className="th-content">
                Age
                {getSortIcon('age')}
              </div>
            </th>
            <th onClick={() => handleSort('date')} className="sortable">
              <div className="th-content">
                Date of Scan
                {getSortIcon('date')}
              </div>
            </th>
            <th onClick={() => handleSort('classification')} className="sortable">
              <div className="th-content">
                Classification
                {getSortIcon('classification')}
              </div>
            </th>
            <th onClick={() => handleSort('confidence')} className="sortable">
              <div className="th-content">
                Confidence
                {getSortIcon('confidence')}
              </div>
            </th>
            <th>Status</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {sortedCases.length === 0 ? (
            <tr>
              <td colSpan={7} className="empty-table-message">
                No cases found
              </td>
            </tr>
          ) : (
            sortedCases.map((caseItem) => {
              const badge = getClassificationBadge(caseItem);
              const confidence = getConfidenceScore(caseItem);
              const BadgeIcon = badge.icon;

              return (
                <tr 
                  key={caseItem.id} 
                  onClick={() => onRowClick && onRowClick(caseItem.id)}
                  className="table-row-clickable"
                >
                  <td className="patient-name">{caseItem.patient_name || 'Unknown'}</td>
                  <td>{caseItem.age || 'N/A'}</td>
                  <td>{formatDate(caseItem.date_of_scan || caseItem.created_at)}</td>
                  <td>
                    <span className={`classification-badge ${badge.class}`}>
                      <BadgeIcon className="badge-icon" />
                      {badge.label}
                    </span>
                  </td>
                  <td>
                    {confidence !== null ? (
                      <div className="confidence-cell">
                        <div className="confidence-value">{confidence.toFixed(1)}%</div>
                        <div className="confidence-bar">
                          <div 
                            className={`confidence-bar-fill ${
                              confidence >= 85 ? 'high' : confidence >= 70 ? 'medium' : 'low'
                            }`}
                            style={{ width: `${confidence}%` }}
                          />
                        </div>
                      </div>
                    ) : (
                      <span className="confidence-na">N/A</span>
                    )}
                  </td>
                  <td>
                    {caseItem.predicted_class ? (
                      <span className="status-completed">✓ Completed</span>
                    ) : (
                      <span className="status-pending">⏳ Pending</span>
                    )}
                  </td>
                  <td onClick={(e) => e.stopPropagation()}>
                    <Link 
                      to={`/case/${caseItem.id}`} 
                      className="view-link"
                    >
                      View
                    </Link>
                  </td>
                </tr>
              );
            })
          )}
        </tbody>
      </table>
    </div>
  );
};

export default EnhancedCasesTable;


