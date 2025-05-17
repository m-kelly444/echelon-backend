"use client"

import { useState, useEffect } from "react"
import { EchelonLogo } from "@/components/echelon-logo"
import { motion } from "framer-motion"

interface LoadingScreenProps {
  message?: string
}

export default function LoadingScreen({ message = "LOADING DATA" }: LoadingScreenProps) {
  const [progress, setProgress] = useState(0)
  const [statusMessages, setStatusMessages] = useState<string[]>([])

  useEffect(() => {
    // Simulate loading progress
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval)
          return 100
        }
        return prev + Math.random() * 10
      })
    }, 300)

    // Simulate status messages
    const messages = [
      "Establishing quantum-secured connection...",
      "Initializing neural network nodes...",
      "Syncing with global threat database...",
      "Calibrating prediction algorithms...",
      "Loading threat intelligence data...",
      "Verifying data integrity...",
      "Preparing visualization modules...",
      "Analyzing recent attack patterns...",
      "Calculating threat probabilities...",
      "Rendering interface components...",
    ]

    let index = 0
    const messageInterval = setInterval(() => {
      if (index < messages.length) {
        setStatusMessages((prev) => [...prev, messages[index]])
        index++
      } else {
        clearInterval(messageInterval)
      }
    }, 800)

    return () => {
      clearInterval(interval)
      clearInterval(messageInterval)
    }
  }, [])

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-8 bg-terminal-black text-neon-green relative">
      <div className="absolute inset-0 bg-grid-pattern opacity-10"></div>
      <div className="absolute inset-0 bg-radial-gradient"></div>

      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        className="terminal-container max-w-md w-full p-8 border border-neon-green/30 bg-black/80 rounded-md shadow-glow relative z-10"
      >
        <div className="flex flex-col items-center mb-8">
          <EchelonLogo className="w-24 h-24 mb-4 animate-pulse-slow" />
          <h2 className="text-2xl font-orbitron font-bold text-center text-glow">{message}</h2>
        </div>

        <div className="terminal-output mb-6 bg-black/60 border border-neon-green/20 p-3 h-48 overflow-y-auto font-mono text-xs">
          {statusMessages.map((msg, i) => (
            <div key={i} className="mb-1">
              <span className="text-neon-green/70">[SYSTEM]</span> {msg}
            </div>
          ))}
          <span className="typing-cursor">_</span>
        </div>

        <div className="mb-2 flex justify-between text-xs">
          <span>INITIALIZING NEURAL NETWORK</span>
          <span>{Math.min(100, Math.floor(progress))}%</span>
        </div>

        <div className="w-full h-2 bg-black/60 rounded-full overflow-hidden border border-neon-green/30">
          <motion.div
            className="h-full bg-neon-green"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.3 }}
          ></motion.div>
        </div>

        <div className="text-xs text-center text-neon-green/50 mt-4 font-mono">
          <div className="flex items-center justify-center">
            <div className="pulse-dot mr-2"></div>
            QUANTUM ENCRYPTION ACTIVE
          </div>
        </div>
      </motion.div>
    </div>
  )
}
