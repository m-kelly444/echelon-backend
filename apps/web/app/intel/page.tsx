'use client';

import React, { useState } from 'react';
import Header from '../../components/Header';

// Real CVE data for display
const vulnerabilities = [
  {
    id: 'CVE-2023-3519',
    description: 'Buffer Overflow vulnerability in Citrix Gateway enables remote code execution',
    published: '2023-07-19',
    severity: 'CRITICAL',
    cvss: 9.8,
    exploited: true,
    products: ['Citrix ADC', 'Citrix Gateway']
  },
  {
    id: 'CVE-2023-20198',
    description: 'Privilege escalation vulnerability in Cisco IOS XE Software web UI',
    published: '2023-10-16',
    severity: 'CRITICAL',
    cvss: 10.0,
    exploited: true,
    products: ['Cisco IOS XE Software']
  },
  {
    id: 'CVE-2023-36884',
    description: 'Microsoft Office and Windows HTML Remote Code Execution Vulnerability',
    published: '2023-07-11',
    severity: 'HIGH',
    cvss: 8.3,
    exploited: true,
    products: ['Microsoft Office', 'Windows']
  },
  {
    id: 'CVE-2023-4966',
    description: 'NetScaler Citrix vulnerability (Citrix Bleed) allowing credential theft',
    published: '2023-10-10',
    severity: 'HIGH',
    cvss: 9.4,
    exploited: true,
    products: ['Citrix ADC', 'Citrix Gateway']
  },
  {
    id: 'CVE-2023-23397',
    description: 'Microsoft Outlook privilege escalation vulnerability',
    published: '2023-03-14',
    severity: 'HIGH',
    cvss: 9.8,
    exploited: true,
    products: ['Microsoft Outlook']
  },
  {
    id: 'CVE-2023-27350',
    description: 'PaperCut NG/MF authentication vulnerability',
    published: '2023-04-17',
    severity: 'CRITICAL',
    cvss: 8.7,
    exploited: true,
    products: ['PaperCut NG', 'PaperCut MF']
  }
];

export default function IntelPage() {
  const [activeFilter, setActiveFilter] = useState('all');
  
  const filteredVulns = activeFilter === 'all' 
    ? vulnerabilities
    : vulnerabilities.filter(v => v.severity.toLowerCase() === activeFilter.toLowerCase());
  
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
              {filteredVulns.map((vuln) => (
                <tr key={vuln.id} className="border-b border-gray-800 hover:bg-gray-900/50">
                  <td className="px-4 py-3 font-mono text-sm text-blue-400">{vuln.id}</td>
                  <td className="px-4 py-3 text-sm">{vuln.description}</td>
                  <td className="px-4 py-3 text-sm text-gray-400">{vuln.published}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-1 text-xs rounded-sm ${
                      vuln.severity === 'CRITICAL' 
                        ? 'bg-red-900/50 text-red-300 border border-red-500/30' 
                        : 'bg-yellow-900/50 text-yellow-300 border border-yellow-500/30'
                    }`}>
                      {vuln.severity}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center">
                      <span className="mr-2">{vuln.cvss.toFixed(1)}</span>
                      <div className="w-16 bg-gray-700 rounded-full h-1.5">
                        <div 
                          className="h-1.5 rounded-full bg-gradient-to-r from-green-500 to-red-500" 
                          style={{ width: `${(vuln.cvss / 10) * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    {vuln.exploited && (
                      <span className="bg-red-900/30 text-red-400 px-2 py-1 text-xs rounded-sm border border-red-500/30">
                        Actively Exploited
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
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
              <li>Vendor Security Advisories</li>
            </ul>
            <p className="mt-4 text-xs text-gray-400">
              All data shown represents real-world vulnerabilities that have been actively exploited in the wild.
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
