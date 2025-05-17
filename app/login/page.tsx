"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { EchelonLogo } from "@/components/echelon-logo"
import { useToast } from "@/hooks/use-toast"
import { Eye, EyeOff, Lock, Mail, Shield } from "lucide-react"
import { signIn } from "next-auth/react"
import { motion } from "framer-motion"

export default function Login() {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [loading, setLoading] = useState(false)
  const [showPassword, setShowPassword] = useState(false)
  const [authMessages, setAuthMessages] = useState<string[]>([])
  const router = useRouter()
  const searchParams = useSearchParams()
  const { toast } = useToast()

  // Check for error in URL
  const error = searchParams.get("error")

  useEffect(() => {
    if (error) {
      toast({
        variant: "destructive",
        title: "Authentication Error",
        description: "Invalid credentials. Access denied.",
      })
      setAuthMessages((prev) => [...prev, "Authentication failed. Access denied."])
    }
  }, [error, toast])

  // Simulated terminal messages
  useEffect(() => {
    const messages = [
      "Initializing secure connection...",
      "Verifying encryption protocols...",
      "Establishing quantum-secured channel...",
      "Connection established. Awaiting credentials...",
    ]

    let index = 0
    const interval = setInterval(() => {
      if (index < messages.length) {
        setAuthMessages((prev) => [...prev, messages[index]])
        index++
      } else {
        clearInterval(interval)
      }
    }, 800)

    return () => clearInterval(interval)
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setAuthMessages((prev) => [...prev, "Authenticating credentials..."])

    try {
      const result = await signIn("credentials", {
        redirect: false,
        email,
        password,
      })

      if (result?.error) {
        setAuthMessages((prev) => [...prev, "Authentication failed. Access denied."])
        toast({
          variant: "destructive",
          title: "Authentication Failed",
          description: "Invalid credentials. Access denied.",
        })
      } else {
        setAuthMessages((prev) => [...prev, "Authentication successful. Granting access..."])
        toast({
          title: "Access Granted",
          description: "Welcome to Echelon Neural Network.",
        })
        router.push("/dashboard")
      }
    } catch (error) {
      console.error("Login error:", error)
      setAuthMessages((prev) => [...prev, "System error. Authentication protocol failure."])
      toast({
        variant: "destructive",
        title: "System Error",
        description: "Authentication protocol failure. Please try again.",
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center p-8 bg-terminal-black text-neon-green">
      <div className="absolute inset-0 bg-grid-pattern opacity-10"></div>
      <div className="absolute inset-0 bg-radial-gradient"></div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="terminal-container max-w-md w-full p-8 border border-neon-green/30 bg-black/80 rounded-md shadow-glow relative z-10"
      >
        <div className="flex flex-col items-center mb-8">
          <EchelonLogo className="w-20 h-20 mb-4" />
          <h2 className="text-2xl font-orbitron font-bold text-center text-glow">ECHELON TERMINAL</h2>
          <p className="text-xs text-neon-green/70 mt-1 font-mono">NEURAL THREAT INTELLIGENCE SYSTEM</p>
        </div>

        <div className="terminal-output mb-6 bg-black/60 border border-neon-green/20 p-3 h-32 overflow-y-auto font-mono text-xs">
          {authMessages.map((msg, i) => (
            <div key={i} className="mb-1">
              <span className="text-neon-green/70">[SYSTEM]</span> {msg}
            </div>
          ))}
          {loading && <span className="typing-cursor">_</span>}
        </div>

        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label className="block text-xs mb-2 font-mono">CLEARANCE ID</label>
            <div className="relative">
              <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-neon-green/50" />
              <Input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full bg-black/60 border border-neon-green/50 pl-10 py-2 font-mono text-neon-green focus:border-neon-green focus:ring-1 focus:ring-neon-green"
                required
                placeholder="Enter your clearance ID"
              />
            </div>
          </div>

          <div className="mb-6">
            <label className="block text-xs mb-2 font-mono">ACCESS CODE</label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-neon-green/50" />
              <Input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-black/60 border border-neon-green/50 pl-10 pr-10 py-2 font-mono text-neon-green focus:border-neon-green focus:ring-1 focus:ring-neon-green"
                required
                placeholder="Enter your access code"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-neon-green/50 hover:text-neon-green"
              >
                {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
          </div>

          <Button
            type="submit"
            disabled={loading}
            variant="outline"
            className="w-full border border-neon-green py-2 px-4 hover:bg-neon-green/20 transition-all text-neon-green font-mono"
          >
            {loading ? (
              <div className="flex items-center">
                <Shield className="animate-pulse mr-2 h-4 w-4" />
                AUTHENTICATING...
              </div>
            ) : (
              <div className="flex items-center">
                <Shield className="mr-2 h-4 w-4" />
                AUTHENTICATE
              </div>
            )}
          </Button>
        </form>

        <div className="mt-6 text-center text-sm font-mono">
          <p className="text-neon-green/70">No clearance credentials?</p>
          <Link href="/register" className="text-neon-green hover:text-neon-blue transition-colors mt-2 inline-block">
            REQUEST ACCESS CREDENTIALS
          </Link>
        </div>
      </motion.div>
    </div>
  )
}
