import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const Profile = () => {
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    fetchProfile();
  }, []);

  const getAuthToken = () => {
    return localStorage.getItem('access_token');
  };

  const fetchProfile = async () => {
    const token = getAuthToken();
    if (!token) {
      navigate('/login');
      return;
    }

    try {
      const response = await fetch('http://localhost:8000/profile', {
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
        throw new Error('Failed to fetch profile');
      }

      const data = await response.json();
      setProfile(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('access_token');
    navigate('/login');
  };

  if (loading) {
    return (
      <div style={{padding: '2rem', textAlign: 'center'}}>
        <div>Loading profile...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{padding: '2rem', background: '#fee2e2', color: '#991b1b', borderRadius: '0.5rem', margin: '2rem'}}>
        Error: {error}
      </div>
    );
  }

  if (!profile) {
    return null;
  }

  return (
    <div style={{maxWidth: '800px', margin: '2rem auto', padding: '0 1rem'}}>
      <div style={{background: '#fff', borderRadius: '0.5rem', boxShadow: '0 2px 8px rgba(0,0,0,0.04)', padding: '2rem'}}>
        <h2 style={{color: '#047857', marginBottom: '2rem'}}>User Profile</h2>
        
        <div style={{display: 'flex', gap: '2rem', marginBottom: '2rem', alignItems: 'flex-start'}}>
          {profile.profile_pic ? (
            <img 
              src={profile.profile_pic} 
              alt="Profile" 
              style={{
                width: '120px', 
                height: '120px', 
                borderRadius: '50%', 
                objectFit: 'cover',
                border: '3px solid #047857'
              }} 
            />
          ) : (
            <div style={{
              width: '120px', 
              height: '120px', 
              borderRadius: '50%', 
              background: '#047857',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#fff',
              fontSize: '2rem',
              fontWeight: 'bold'
            }}>
              {profile.name ? profile.name.charAt(0).toUpperCase() : 'U'}
            </div>
          )}
          
          <div style={{flex: 1}}>
            <h3 style={{color: '#374151', marginBottom: '0.5rem'}}>{profile.name || 'No name'}</h3>
            <div style={{color: '#6b7280', marginBottom: '1rem'}}>{profile.email}</div>
          </div>
        </div>

        <div style={{display: 'grid', gap: '1.5rem', marginBottom: '2rem'}}>
          <div style={{padding: '1rem', background: '#f3f4f6', borderRadius: '0.5rem'}}>
            <div style={{fontSize: '0.875rem', color: '#6b7280', marginBottom: '0.25rem'}}>Cases Analyzed</div>
            <div style={{fontSize: '2rem', fontWeight: 'bold', color: '#047857'}}>{profile.cases_analyzed || 0}</div>
          </div>

          <div style={{padding: '1rem', background: '#f3f4f6', borderRadius: '0.5rem'}}>
            <div style={{fontSize: '0.875rem', color: '#6b7280', marginBottom: '0.25rem'}}>Account Created</div>
            <div style={{color: '#374151'}}>
              {profile.created_at ? new Date(profile.created_at).toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'long',
                day: 'numeric'
              }) : 'N/A'}
            </div>
          </div>

          <div style={{padding: '1rem', background: '#f3f4f6', borderRadius: '0.5rem'}}>
            <div style={{fontSize: '0.875rem', color: '#6b7280', marginBottom: '0.25rem'}}>Last Login</div>
            <div style={{color: '#374151'}}>
              {profile.last_login ? new Date(profile.last_login).toLocaleString('en-US', {
                year: 'numeric',
                month: 'long',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
              }) : 'Never'}
            </div>
          </div>
        </div>

        <div style={{display: 'flex', gap: '1rem', marginTop: '2rem'}}>
          <button 
            onClick={() => navigate('/history')}
            style={{
              background: '#047857', 
              color: '#fff', 
              border: 'none', 
              borderRadius: '0.3rem', 
              padding: '0.75rem 1.5rem', 
              cursor: 'pointer',
              fontSize: '1rem'
            }}
          >
            View Case History
          </button>
          <button 
            onClick={handleLogout}
            style={{
              background: '#dc2626', 
              color: '#fff', 
              border: 'none', 
              borderRadius: '0.3rem', 
              padding: '0.75rem 1.5rem', 
              cursor: 'pointer',
              fontSize: '1rem'
            }}
          >
            Logout
          </button>
        </div>
      </div>
    </div>
  );
};

export default Profile;


