import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { Link } from 'react-router-dom';
import { 
  Table,
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { fetchPredictions } from '@/lib/api';

const Dashboard = () => {
  // State for prediction data
  const [predictions, setPredictions] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch predictions on component mount
  useEffect(() => {
    const loadPredictions = async () => {
      try {
        setIsLoading(true);
        const data = await fetchPredictions();
        setPredictions(data);
      } catch (err) {
        console.error('Error loading predictions:', err);
        setError('Failed to load prediction data');
      } finally {
        setIsLoading(false);
      }
    };

    loadPredictions();
  }, []);

  // Calculate prediction statistics
  const calculatePredictionStats = () => {
    if (!predictions.length) return { high: 0, medium: 0, low: 0 };
    
    return predictions.reduce((stats, pred) => {
      if (pred.lead_score >= 80) stats.high += 1;
      else if (pred.lead_score >= 50) stats.medium += 1;
      else stats.low += 1;
      return stats;
    }, { high: 0, medium: 0, low: 0 });
  };

  const predictionStats = calculatePredictionStats();
  const totalPredictions = predictions.length;

  // Sample data
  const leadStatsData = [
    { name: 'New Leads', count: totalPredictions, change: '+5%', changeType: 'positive' },
    { name: 'Qualified Leads', count: predictionStats.high + predictionStats.medium, change: '+12%', changeType: 'positive' },
    { name: 'Conversion Rate', count: totalPredictions ? `${Math.round((predictionStats.high / totalPredictions) * 100)}%` : '0%', change: '-3%', changeType: 'negative' },
    { name: 'Avg. Score', count: totalPredictions ? `${Math.round(predictions.reduce((sum, p) => sum + p.lead_score, 0) / totalPredictions)}/100` : '0/100', change: '+2%', changeType: 'positive' }
  ];
  
  const performanceData = [
    { month: 'Jan', leads: 40 },
    { month: 'Feb', leads: 30 },
    { month: 'Mar', leads: 45 },
    { month: 'Apr', leads: 60 },
    { month: 'May', leads: 75 },
    { month: 'Jun', leads: 65 },
  ];

  // Prepare source data from predictions
  const calculateSourceData = () => {
    if (!predictions.length) return [
      { name: 'Website', value: 45 },
      { name: 'Social', value: 25 },
      { name: 'Email', value: 15 },
      { name: 'Referral', value: 15 },
    ];
    
    const sources = {};
    predictions.forEach(pred => {
      const source = pred.lead_source || 'Other';
      sources[source] = (sources[source] || 0) + 1;
    });
    
    return Object.entries(sources).map(([name, value]) => ({ name, value }));
  };

  const sourceData = calculateSourceData();
  const COLORS = ['#2563eb', '#10b981', '#f59e0b', '#ef4444'];

  // Transform predictions to recent leads data
  const recentLeadsData = predictions.slice(0, 5).map((pred, index) => ({
    id: index + 1,
    name: `Lead ${index + 1}`,
    email: `lead${index + 1}@example.com`,
    phone: `(555) ${100 + index}-${1000 + index}`,
    source: pred.lead_source || 'Unknown',
    score: pred.lead_score,
    status: pred.lead_score >= 80 ? 'high' : pred.lead_score >= 50 ? 'medium' : 'low'
  }));

  return (
    <div className="main-content max-h-screen overflow-y-auto p-6">
      <div className="page-header flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Dashboard</h1>
        <Link to="/add-lead">
          <Button className="bg-primary hover:bg-primary/90">+ New Lead</Button>
        </Link>
      </div>

      {isLoading ? (
        <div className="text-center p-10">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary mx-auto"></div>
          <p className="mt-4">Loading prediction data...</p>
        </div>
      ) : error ? (
        <div className="bg-red-100 text-red-700 p-4 rounded-lg mb-6">
          {error}
        </div>
      ) : (
        <>
          {/* Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
            {leadStatsData.map((stat, index) => (
              <div className="bg-white rounded-lg shadow p-4" key={index}>
                <div className="text-gray-500 font-medium">{stat.name}</div>
                <div className="text-2xl font-bold mt-1">{stat.count}</div>
                <div className={`text-sm mt-2 ${stat.changeType === 'positive' ? 'text-green-500' : 'text-red-500'}`}>
                  {stat.change} from last month
                </div>
              </div>
            ))}
          </div>

          {/* Performance Charts */}
          <div className="bg-white rounded-lg shadow mb-6">
            <div className="p-4 border-b">
              <h2 className="font-bold">Lead Performance</h2>
            </div>
            <div className="h-72 p-4">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="leads" fill="#2563eb" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Charts Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            {/* Lead Sources Chart */}
            <div className="bg-white rounded-lg shadow">
              <div className="p-4 border-b">
                <h2 className="font-bold">Lead Sources</h2>
              </div>
              <div className="h-64 p-4">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={sourceData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    >
                      {sourceData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Predictions Overview */}
            <div className="bg-white rounded-lg shadow">
              <div className="p-4 border-b">
                <h2 className="font-bold">Predictions Overview</h2>
              </div>
              <div className="p-4">
                <div className="mb-4">
                  <div className="flex justify-between mb-2">
                    <span>High Potential (80-100)</span>
                    <span className="font-semibold text-green-600">{predictionStats.high} leads</span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full">
                    <div 
                      className="h-2 bg-green-600 rounded-full" 
                      style={{ width: `${totalPredictions ? (predictionStats.high / totalPredictions) * 100 : 0}%` }}
                    ></div>
                  </div>
                </div>
                <div className="mb-4">
                  <div className="flex justify-between mb-2">
                    <span>Medium Potential (50-79)</span>
                    <span className="font-semibold text-yellow-600">{predictionStats.medium} leads</span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full">
                    <div 
                      className="h-2 bg-yellow-500 rounded-full" 
                      style={{ width: `${totalPredictions ? (predictionStats.medium / totalPredictions) * 100 : 0}%` }}
                    ></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between mb-2">
                    <span>Low Potential (0-49)</span>
                    <span className="font-semibold text-red-600">{predictionStats.low} leads</span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full">
                    <div 
                      className="h-2 bg-red-600 rounded-full" 
                      style={{ width: `${totalPredictions ? (predictionStats.low / totalPredictions) * 100 : 0}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Recent Leads Table */}
          <div className="bg-white rounded-lg shadow">
            <div className="p-4 border-b flex justify-between items-center">
              <h2 className="font-bold">Recent Predictions</h2>
              <Link to="/leads">
                <Button variant="outline" size="sm">View All</Button>
              </Link>
            </div>
            <div className="overflow-x-auto max-h-80">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>ID</TableHead>
                    <TableHead>Source</TableHead>
                    <TableHead>Company Size</TableHead>
                    <TableHead>Industry</TableHead>
                    <TableHead>Score</TableHead>
                    <TableHead>Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {predictions.slice(0, 5).map((pred, index) => (
                    <TableRow key={index}>
                      <TableCell className="font-medium">Lead {index + 1}</TableCell>
                      <TableCell>{pred.lead_source || 'Unknown'}</TableCell>
                      <TableCell>{pred.company_size || 'Unknown'}</TableCell>
                      <TableCell>{pred.industry || 'Unknown'}</TableCell>
                      <TableCell>{pred.lead_score}/100</TableCell>
                      <TableCell>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          pred.lead_score >= 80 ? 'bg-green-100 text-green-800' : 
                          pred.lead_score >= 50 ? 'bg-yellow-100 text-yellow-800' : 
                          'bg-red-100 text-red-800'
                        }`}>
                          {pred.lead_score >= 80 ? 'High' : pred.lead_score >= 50 ? 'Medium' : 'Low'}
                        </span>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default Dashboard;
