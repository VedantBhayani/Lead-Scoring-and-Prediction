
import React, { useState } from 'react';

const Settings = () => {
  const [darkMode, setDarkMode] = useState(false);
  const [emailNotifications, setEmailNotifications] = useState(true);
  const [leadScoreThreshold, setLeadScoreThreshold] = useState(70);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState(5);

  const handleThresholdChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value);
    if (!isNaN(value) && value >= 0 && value <= 100) {
      setLeadScoreThreshold(value);
    }
  };

  const handleIntervalChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setRefreshInterval(parseInt(e.target.value));
  };

  const handleSaveSettings = () => {
    // In a real app, you would save these settings to a backend or localStorage
    console.log({
      darkMode,
      emailNotifications,
      leadScoreThreshold,
      autoRefresh,
      refreshInterval
    });
    
    alert('Settings saved successfully!');
  };

  return (
    <div className="main-content">
      <div className="page-header">
        <h1>Settings</h1>
      </div>

      {/* Appearance Settings */}
      <div className="card mb-6">
        <div className="card-header">
          <h2 className="card-title">Appearance</h2>
        </div>
        <div className="form-group">
          <div className="flex items-center justify-between">
            <label htmlFor="dark-mode" className="form-label mb-0">Dark Mode</label>
            <label className="switch">
              <input 
                type="checkbox" 
                id="dark-mode" 
                checked={darkMode} 
                onChange={() => setDarkMode(!darkMode)} 
              />
              <span className="slider"></span>
            </label>
          </div>
          <p className="text-sm text-gray-500 mt-1">
            Switch between light and dark color themes
          </p>
        </div>
      </div>

      {/* Notification Settings */}
      <div className="card mb-6">
        <div className="card-header">
          <h2 className="card-title">Notifications</h2>
        </div>
        <div className="form-group">
          <div className="flex items-center justify-between">
            <label htmlFor="email-notifications" className="form-label mb-0">Email Notifications</label>
            <label className="switch">
              <input 
                type="checkbox" 
                id="email-notifications" 
                checked={emailNotifications} 
                onChange={() => setEmailNotifications(!emailNotifications)} 
              />
              <span className="slider"></span>
            </label>
          </div>
          <p className="text-sm text-gray-500 mt-1">
            Receive email notifications for new high-potential leads
          </p>
        </div>
      </div>

      {/* Lead Scoring Settings */}
      <div className="card mb-6">
        <div className="card-header">
          <h2 className="card-title">Lead Scoring</h2>
        </div>
        <div className="form-group">
          <label htmlFor="threshold" className="form-label">High Potential Lead Threshold</label>
          <div className="flex items-center">
            <input 
              type="range" 
              id="threshold" 
              min="0" 
              max="100" 
              value={leadScoreThreshold} 
              onChange={handleThresholdChange}
              className="w-full mr-4"
            />
            <div className="w-12 text-center">{leadScoreThreshold}</div>
          </div>
          <p className="text-sm text-gray-500 mt-1">
            Leads scored above this threshold will be marked as high potential
          </p>
        </div>
      </div>

      {/* Dashboard Settings */}
      <div className="card mb-6">
        <div className="card-header">
          <h2 className="card-title">Dashboard</h2>
        </div>
        <div className="form-group">
          <div className="flex items-center justify-between">
            <label htmlFor="auto-refresh" className="form-label mb-0">Auto-refresh Dashboard</label>
            <label className="switch">
              <input 
                type="checkbox" 
                id="auto-refresh" 
                checked={autoRefresh} 
                onChange={() => setAutoRefresh(!autoRefresh)} 
              />
              <span className="slider"></span>
            </label>
          </div>
          
          {autoRefresh && (
            <div className="mt-3">
              <label className="form-label">Refresh Interval</label>
              <select 
                className="form-select" 
                value={refreshInterval} 
                onChange={handleIntervalChange}
              >
                <option value="1">Every 1 minute</option>
                <option value="5">Every 5 minutes</option>
                <option value="15">Every 15 minutes</option>
                <option value="30">Every 30 minutes</option>
                <option value="60">Every hour</option>
              </select>
            </div>
          )}
        </div>
      </div>

      {/* Save Button */}
      <div className="flex justify-end">
        <button 
          className="btn btn-primary" 
          onClick={handleSaveSettings}
        >
          Save Settings
        </button>
      </div>
    </div>
  );
};

export default Settings;
