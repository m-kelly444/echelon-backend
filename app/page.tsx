import Link from "next/link"
import { Button } from "@/components/ui/button"
import { EchelonLogo } from "@/components/echelon-logo"
import { Terminal, Shield, Zap, Cpu, Globe } from "lucide-react"

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8 bg-terminal-black text-neon-green relative overflow-hidden">
      <div className="absolute inset-0 bg-grid-pattern opacity-10"></div>
      <div className="absolute inset-0 bg-radial-gradient"></div>

      <div className="relative z-10 flex flex-col items-center max-w-5xl w-full">
        <div className="mb-8 relative">
          <EchelonLogo className="w-32 h-32 animate-pulse-slow" />
          <div className="absolute -inset-4 bg-neon-green opacity-20 blur-xl rounded-full animate-pulse-slow"></div>
        </div>

        <h1 className="text-5xl md:text-7xl font-orbitron font-bold mb-4 text-glow-xl tracking-wider text-center">
          ECHELON
        </h1>

        <div className="glitch-container mb-2">
          <p className="glitch-text text-xl md:text-2xl font-mono tracking-wide">NEURAL THREAT INTELLIGENCE</p>
        </div>

        <p className="terminal-text text-sm md:text-base mb-12 text-center max-w-2xl typing-effect">
          QUANTUM-SECURED NETWORK • ML-BASED ATTACK PREDICTION • REAL-TIME THREAT ANALYSIS
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12 w-full max-w-2xl">
          <div className="terminal-container p-4 border border-neon-green/30 bg-black/70 rounded-md shadow-glow">
            <div className="flex items-center mb-3">
              <Shield className="w-5 h-5 mr-2 text-neon-green" />
              <h3 className="font-orbitron text-sm">THREAT DETECTION</h3>
            </div>
            <p className="text-xs text-neon-green/80">
              Advanced neural networks identify emerging threats before they materialize
            </p>
          </div>

          <div className="terminal-container p-4 border border-neon-green/30 bg-black/70 rounded-md shadow-glow">
            <div className="flex items-center mb-3">
              <Zap className="w-5 h-5 mr-2 text-neon-green" />
              <h3 className="font-orbitron text-sm">REAL-TIME ANALYSIS</h3>
            </div>
            <p className="text-xs text-neon-green/80">
              Continuous monitoring of global threat landscape with millisecond precision
            </p>
          </div>

          <div className="terminal-container p-4 border border-neon-green/30 bg-black/70 rounded-md shadow-glow">
            <div className="flex items-center mb-3">
              <Cpu className="w-5 h-5 mr-2 text-neon-green" />
              <h3 className="font-orbitron text-sm">QUANTUM ALGORITHMS</h3>
            </div>
            <p className="text-xs text-neon-green/80">
              Leveraging quantum computing principles for unparalleled prediction accuracy
            </p>
          </div>

          <div className="terminal-container p-4 border border-neon-green/30 bg-black/70 rounded-md shadow-glow">
            <div className="flex items-center mb-3">
              <Globe className="w-5 h-5 mr-2 text-neon-green" />
              <h3 className="font-orbitron text-sm">GLOBAL COVERAGE</h3>
            </div>
            <p className="text-xs text-neon-green/80">
              Monitoring attack vectors across 195 countries and 17 digital jurisdictions
            </p>
          </div>
        </div>

        <div className="flex flex-col md:flex-row space-y-4 md:space-y-0 md:space-x-6">
          <Link href="/login">
            <Button
              variant="outline"
              className="terminal-button py-6 px-8 text-center border-neon-green text-neon-green hover:bg-neon-green/20 transition-all group"
            >
              <Terminal className="mr-2 h-4 w-4 group-hover:animate-pulse" />
              ACCESS SECURE TERMINAL
            </Button>
          </Link>

          <Link href="/register">
            <Button
              variant="ghost"
              className="py-6 px-8 text-center text-neon-green hover:bg-neon-green/10 transition-all"
            >
              CREATE CLEARANCE CREDENTIALS
            </Button>
          </Link>
        </div>

        <div className="text-xs text-center text-neon-green/50 mt-12 font-mono">
          <div className="flex items-center justify-center mb-1">
            <div className="pulse-dot mr-2"></div>
            SECURE QUANTUM ENCRYPTION ACTIVE
          </div>
          <div>SYSTEM VERSION 4.7.2 • CLEARANCE LEVEL REQUIRED</div>
        </div>
      </div>
    </main>
  )
}
