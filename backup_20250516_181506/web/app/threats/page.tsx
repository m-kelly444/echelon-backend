'use client';

import React, { useState, useEffect } from 'react';
import Header from '../../components/Header';

export default function ThreatsPage() {
  const [threatActors, setThreatActors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    // Fetch real data from API
    const fetchThreatActors = async () => {
      try {
        setLoading(true);
        // Fetch APT mappings from the enhanced API
        const response = await fetch('http://localhost:8080/apt_groups');
        
        // If endpoint doesn't exist or fails, try to fetch from processed data
        if (!response.ok) {
          const altResponse = await fetch('/api/threat_actors');
          if (!altResponse.ok) {
            throw new Error('Could not fetch threat actor data');
          }
          const data = await altResponse.json();
          setThreatActors(data);
          setLoading(false);
          return;
        }
        
        const data = await response.json();
        setThreatActors(data.apt_groups || []);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching threat actors:', error);
        setError('Failed to load threat actor data. Please ensure the API is running with real data.');
        setLoading(false);
      }
    };
    
    fetchThreatActors();
  }, []);
  
  return (
    <main className="min-h-screen">
      <Header />
      
      <div className="container mx-auto px-4 py-8">
        <h2 className="text-2xl font-bold mb-4 glitch-text" data-text="THREAT ACTORS">
          THREAT ACTORS
          <span className="terminal-cursor"></span>
        </h2>
        
        {loading ? (
          <div className="text-center py-8">
            <p>Loading threat actor data...</p>
          </div>
        ) : error ? (
          <div className="bg-red-900/50 border border-red-500/50 p-4 rounded-sm my-4">
            <p>{error}</p>
            <p className="mt-2 text-sm">
              This page requires the enhanced API with real threat intelligence data. 
              Please run the enhanced API server with proper API keys configured.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {threatActors.length > 0 ? (
              threatActors.map((actor) => (
                <div 
                  key={actor.id}
                  className="bg-gray-900 bg-opacity-70 border border-green-500/30 p-4 rounded-sm hover:border-green-400 transition-colors"
                >
                  <div className="flex justify-between items-start mb-3">
                    <h3 className="text-xl font-bold">{actor.name}</h3>
                    <span className="px-2 py-1 text-xs rounded-sm bg-green-900/50 border border-green-500/30">
                      {actor.origin}
                    </span>
                  </div>
                  
                  <p className="text-sm mb-3 text-gray-300">{actor.description}</p>
                  
                  <div className="mb-2">
                    <span className="text-xs text-gray-400">Aliases:</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {actor.aliases && actor.aliases.map((alias, index) => (
                        <span 
                          key={index}
                          className="text-xs bg-gray-800 px-2 py-1 rounded-sm"
                        >
                          {alias}
                        </span>
                      ))}
                    </div>
                  </div>
                  
                  <div className="mb-2">
                    <span className="text-xs text-gray-400">Techniques:</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {actor.techniques && actor.techniques.map((technique, index) => (
                        <span 
                          key={index}
                          className="text-xs bg-blue-900/50 text-blue-300 px-2 py-1 rounded-sm"
                        >
                          {technique}
                        </span>
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <span className="text-xs text-gray-400">Targets:</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {actor.targets && actor.targets.map((target, index) => (
                        <span 
                          key={index}
                          className="text-xs bg-red-900/50 text-red-300 px-2 py-1 rounded-sm"
                        >
                          {target}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="col-span-3 text-center py-8">
                <p>No threat actor data available. Please ensure the API is running with real threat intelligence data.</p>
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  );
}
