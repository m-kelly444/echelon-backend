'use client';

import React, { useState, useEffect } from 'react';
import Header from '../../components/Header';

export default function IntelPage() {
  const [vulnerabilities, setVulnerabilities] = useState([]);
  const [activeFilter, setActiveFilter] = useState('all');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    // Fetch real data from the API
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await fetch('http://localhost:8080/cves');
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        const data = await response.json();
        setVulnerabilities(data.cves || []);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching data:', error);
        setError('Failed to load vulnerability data. Please try again later.');
        setLoading(false);
      }
    };
    
    fetchData();
  }, []);
  
  const filteredVulns = activeFilter === 'all' 
    ? vulnerabilities
    : vulnerabilities.filter(v => v.severity && v.severity.toLowerCase() === activeFilter.toLowerCase());
  
  return (
    <main className="min-h-screen">
      <Header />
      
      <div className="container mx-auto px-4 py-8">
        <h2 className="text-2xl font-bold mb-4 glitch-text" data-text="INTELLIGENCE">
          INTELLIGENCE
          <span className="terminal-cursor"></span>
        </h2>
        
        <div className="mb-6">
          <div className="flex space-x-4">
            <button
              className={`px-4 py-2 ${activeFilter === 'all' ? 'bg-green-900 border-green-500' : 'bg-gray-800 border-gray-700'} border rounded-sm transition-colors`}
              onClick={() => setActiveFilter('all')}
            >
              All
            </button>
            <button
              className={`px-4 py-2 ${activeFilter === 'critical' ? 'bg-red-900 border-red-500' : 'bg-gray-800 border-gray-700'} border rounded-sm transition-colors`}
              onClick={() => setActiveFilter('critical')}
            >
              Critical
            </button>
            <button
              className={`px-4 py-2 ${activeFilter === 'high' ? 'bg-yellow-900 border-yellow-500' : 'bg-gray-800 border-gray-700'} border rounded-sm transition-colors`}
              onClick={() => setActiveFilter('high')}
            >
              High
            </button>
          </div>
        </div>
        
        {loading ? (
          <div className="text-center py-8">
            <p>Loading vulnerability data...</p>
          </div>
        ) : error ? (
          <div className="bg-red-900/50 border border-red-500/50 p-4 rounded-sm">
            <p>{error}</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="bg-gray-900 border-b border-green-500/30">
                  <th className="px-4 py-3 text-left">CVE ID</th>
                  <th className="px-4 py-3 text-left">Description</th>
                  <th className="px-4 py-3 text-left">Published</th>
                  <th className="px-4 py-3 text-left">Severity</th>
                  <th className="px-4 py-3 text-left">CVSS</th>
                  <th className="px-4 py-3 text-left">Status</th>
                </tr>
              </thead>
              <tbody>
                {filteredVulns.length > 0 ? (
                  filteredVulns.map((vuln) => (
                    <tr key={vuln.id || vuln.cve_id} className="border-b border-gray-800 hover:bg-gray-900/50">
                      <td className="px-4 py-3 font-mono text-sm text-blue-400">{vuln.id || vuln.cve_id}</td>
                      <td className="px-4 py-3 text-sm">{vuln.description || vuln.name}</td>
                      <td className="px-4 py-3 text-sm text-gray-400">{vuln.published || vuln.date_added}</td>
                      <td className="px-4 py-3">
                        <span className={`px-2 py-1 text-xs rounded-sm ${
                          (vuln.severity === 'CRITICAL' || vuln.severity === 'HIGH')
                            ? 'bg-red-900/50 text-red-300 border border-red-500/30' 
                            : 'bg-yellow-900/50 text-yellow-300 border border-yellow-500/30'
                        }`}>
                          {vuln.severity || "MEDIUM"}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center">
                          <span className="mr-2">{(vuln.base_score || 5).toFixed(1)}</span>
                          <div className="w-16 bg-gray-700 rounded-full h-1.5">
                            <div 
                              className="h-1.5 rounded-full bg-gradient-to-r from-green-500 to-red-500" 
                              style={{ width: `${((vuln.base_score || 5) / 10) * 100}%` }}
                            ></div>
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        {vuln.exploited || vuln.source === "CISA KEV" ? (
                          <span className="bg-red-900/30 text-red-400 px-2 py-1 text-xs rounded-sm border border-red-500/30">
                            Actively Exploited
                          </span>
                        ) : null}
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={6} className="px-4 py-8 text-center">
                      No vulnerabilities found. Try a different filter or check API connection.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
        
        <div className="mt-8 bg-gray-900 bg-opacity-50 border border-green-500/30 p-4 rounded-sm">
          <h3 className="terminal-header text-lg mb-3">DATA SOURCES</h3>
          <div className="text-sm space-y-2">
            <p>
              All vulnerabilities data is sourced from:
            </p>
            <ul className="list-disc list-inside pl-4 space-y-1">
              <li>CISA Known Exploited Vulnerabilities (KEV) Catalog</li>
              <li>National Vulnerability Database (NVD)</li>
              <li>MITRE CVE Database</li>
              <li>Live API Integrations</li>
            </ul>
            <p className="mt-4 text-xs text-gray-400">
              All data shown represents real-world vulnerabilities from authoritative sources.
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
