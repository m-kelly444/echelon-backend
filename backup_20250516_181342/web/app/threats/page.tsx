'use client';

import React from 'react';
import Header from '../../components/Header';

// Real threat actor data based on known APT groups
const threatActors = [
  {
    id: 'apt28',
    name: 'APT28',
    aliases: ['Fancy Bear', 'Sofacy', 'Sednit'],
    origin: 'Russia',
    description: 'State-sponsored threat actor known for targeting government, military, and security organizations.',
    techniques: ['Spear-phishing', 'Zero-day exploitation', 'Credential theft'],
    targets: ['Government', 'Military', 'NATO']
  },
  {
    id: 'apt29',
    name: 'APT29',
    aliases: ['Cozy Bear', 'The Dukes'],
    origin: 'Russia',
    description: 'Sophisticated threat actor focusing on intelligence collection and theft of sensitive data.',
    techniques: ['Supply chain attacks', 'Social engineering', 'Custom malware'],
    targets: ['Government', 'Think tanks', 'Healthcare']
  },
  {
    id: 'lazarus',
    name: 'Lazarus Group',
    aliases: ['Hidden Cobra', 'Guardians of Peace'],
    origin: 'North Korea',
    description: 'Responsible for destructive attacks and financial theft to fund state operations.',
    techniques: ['Watering hole attacks', 'Ransomware', 'SWIFT banking fraud'],
    targets: ['Financial', 'Cryptocurrency', 'Media']
  },
  {
    id: 'apt41',
    name: 'APT41',
    aliases: ['Winnti', 'Barium'],
    origin: 'China',
    description: 'Conducts state-sponsored espionage and financially-motivated attacks.',
    techniques: ['Supply chain compromises', 'Spear-phishing', 'Rootkits'],
    targets: ['Healthcare', 'Technology', 'Gaming']
  },
  {
    id: 'sandworm',
    name: 'Sandworm Team',
    aliases: ['BlackEnergy', 'Voodoo Bear'],
    origin: 'Russia',
    description: 'Known for destructive attacks against critical infrastructure and NotPetya ransomware.',
    techniques: ['BlackEnergy malware', 'Destructive wiper attacks', 'ICS targeting'],
    targets: ['Energy', 'Industrial systems', 'Ukraine']
  },
  {
    id: 'muddy',
    name: 'MuddyWater',
    aliases: ['Earth Vetala', 'TEMP.Zagros'],
    origin: 'Iran',
    description: 'Conducts espionage operations primarily in the Middle East.',
    techniques: ['Macro-enabled documents', 'PowerShell scripts', 'Custom backdoors'],
    targets: ['Government', 'Telecommunications', 'Defense']
  }
];

export default function ThreatsPage() {
  return (
    <main className="min-h-screen">
      <Header />
      
      <div className="container mx-auto px-4 py-8">
        <h2 className="text-2xl font-bold mb-4 glitch-text" data-text="THREAT ACTORS">
          THREAT ACTORS
          <span className="terminal-cursor"></span>
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {threatActors.map((actor) => (
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
                  {actor.aliases.map((alias, index) => (
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
                  {actor.techniques.map((technique, index) => (
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
                  {actor.targets.map((target, index) => (
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
          ))}
        </div>
      </div>
    </main>
  );
}
