
import React, { useState } from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

const PredictionResults = () => {
  // Sample prediction data
  const [predictionData, setPredictionData] = useState([
    { id: 1, name: 'John Doe', email: 'john@example.com', score: 87, probability: 0.87, factors: ['Website Visit', 'Downloaded Whitepaper', 'Industry Match'] },
    { id: 2, name: 'Jane Smith', email: 'jane@example.com', score: 65, probability: 0.65, factors: ['Email Open', 'Form Submission', 'Company Size'] },
    { id: 3, name: 'Mike Johnson', email: 'mike@example.com', score: 42, probability: 0.42, factors: ['Social Media Click', 'Blog Visit'] },
    { id: 4, name: 'Sarah Williams', email: 'sarah@example.com', score: 91, probability: 0.91, factors: ['Multiple Page Views', 'Pricing Page Visit', 'Webinar Attendance', 'Industry Match'] },
    { id: 5, name: 'Alex Brown', email: 'alex@example.com', score: 78, probability: 0.78, factors: ['Case Study Download', 'Multiple Page Views'] },
  ]);

  // Chart data
  const scoreDistributionData = [
    { range: '0-20', count: 12 },
    { range: '21-40', count: 18 },
    { range: '41-60', count: 24 },
    { range: '61-80', count: 30 },
    { range: '81-100', count: 16 },
  ];

  // Function to determine status based on score
  const getStatus = (score: number) => {
    if (score >= 80) return 'high';
    if (score >= 50) return 'medium';
    return 'low';
  };

  return (
    <div className="main-content">
      <div className="page-header">
        <h1>Prediction Results</h1>
      </div>

      {/* Score Distribution Chart */}
      <div className="card mb-6">
        <div className="card-header">
          <h2 className="card-title">Lead Score Distribution</h2>
        </div>
        <div className="chart-container">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={scoreDistributionData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="range" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#2563eb" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Prediction Results Table */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Lead Score Predictions</h2>
        </div>
        <div className="table-container">
          <table className="table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Email</th>
                <th>Score</th>
                <th>Probability</th>
                <th>Key Factors</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {predictionData.map(lead => (
                <tr key={lead.id}>
                  <td>{lead.name}</td>
                  <td>{lead.email}</td>
                  <td>{lead.score}/100</td>
                  <td>{(lead.probability * 100).toFixed(0)}%</td>
                  <td>
                    <ul className="list-disc list-inside">
                      {lead.factors.map((factor, index) => (
                        <li key={index} className="text-sm">{factor}</li>
                      ))}
                    </ul>
                  </td>
                  <td>
                    <span className={`status-pill ${getStatus(lead.score)}`}>
                      {getStatus(lead.score) === 'high' ? 'High' : getStatus(lead.score) === 'medium' ? 'Medium' : 'Low'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Score Explanation */}
      <div className="card mt-6">
        <div className="card-header">
          <h2 className="card-title">Understanding Lead Scores</h2>
        </div>
        <div className="p-4">
          <p className="mb-4">Our lead scoring algorithm analyzes multiple factors to predict conversion probability:</p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <h3 className="lead-score-high font-semibold mb-2">High Potential (80-100)</h3>
              <p className="text-sm">Leads with high engagement and multiple matching criteria. These leads should be prioritized for immediate follow-up.</p>
            </div>
            <div>
              <h3 className="lead-score-medium font-semibold mb-2">Medium Potential (50-79)</h3>
              <p className="text-sm">Leads showing moderate interest but missing key criteria. These leads should be nurtured with targeted content.</p>
            </div>
            <div>
              <h3 className="lead-score-low font-semibold mb-2">Low Potential (0-49)</h3>
              <p className="text-sm">Leads with minimal engagement. These leads may need more time or different approaches to develop interest.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionResults;
