"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { EchelonLogo } from "@/components/echelon-logo"
import { useToast } from "@/hooks/use-toast"
import { Eye, EyeOff, User, Mail, Lock, Shield, CheckCircle, XCircle } from "lucide-react"
import { motion } from "framer-motion"

export default function Register() {
  const [name, setName] = useState("")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [showPassword, setShowPassword] = useState(false)
  const [loading, setLoading] = useState(false)
  const [systemMessages, setSystemMessages] = useState<string[]>([])
  const [passwordStrength, setPasswordStrength] = useState(0)
  const router = useRouter()
  const { toast } = useToast()

  // Password strength checker
  useEffect(() => {
    if (!password) {
      setPasswordStrength(0)
      return
    }

    let strength = 0
    if (password.length >= 8) strength += 1
    if (/[A-Z]/.test(password)) strength += 1
    if (/[0-9]/.test(password)) strength += 1
    if (/[^A-Za-z0-9]/.test(password)) strength += 1

    setPasswordStrength(strength)
  }, [password])

  // Simulated terminal messages
  useEffect(() => {
    const messages = [
      "Initializing secure registration protocol...",
      "Preparing quantum-encrypted channel...",
      "Generating secure credential vault...",
      "Ready to create new security clearance...",
    ]

    let index = 0
    const interval = setInterval(() => {
      if (index < messages.length) {
        setSystemMessages((prev) => [...prev, messages[index]])
        index++
      } else {
        clearInterval(interval)
      }
    }, 800)

    return () => clearInterval(interval)
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (password !== confirmPassword) {
      setSystemMessages((prev) => [...prev, "ERROR: Access code verification failed. Codes do not match."])
      toast({
        variant: "destructive",
        title: "Verification Error",
        description: "Access codes do not match. Please verify and try again.",
      })
      return
    }

    if (passwordStrength < 3) {
      setSystemMessages((prev) => [...prev, "ERROR: Access code strength insufficient. Enhance security parameters."])
      toast({
        variant: "destructive",
        title: "Security Error",
        description: "Access code does not meet security requirements.",
      })
      return
    }

    setLoading(true)
    setSystemMessages((prev) => [...prev, "Creating new security clearance..."])

    try {
      const response = await fetch("/api/auth/register", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name,
          email,
          password,
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.message || "Registration failed")
      }

      setSystemMessages((prev) => [
        ...prev,
        "Security clearance created successfully. Redirecting to authentication...",
      ])
      toast({
        title: "Registration Successful",
        description: "Your security clearance has been created. Please authenticate to continue.",
      })

      setTimeout(() => {
        router.push("/login")
      }, 2000)
    } catch (error) {
      console.error("Registration error:", error)
      setSystemMessages((prev) => [
        ...prev,
        `ERROR: Registration protocol failure. ${error instanceof Error ? error.message : "Unknown error"}`,
      ])
      toast({
        variant: "destructive",
        title: "Registration Error",
        description: error instanceof Error ? error.message : "An error occurred during registration",
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
          <h2 className="text-2xl font-orbitron font-bold text-center text-glow">CREATE CLEARANCE</h2>
          <p className="text-xs text-neon-green/70 mt-1 font-mono">ECHELON NEURAL NETWORK ACCESS</p>
        </div>

        <div className="terminal-output mb-6 bg-black/60 border border-neon-green/20 p-3 h-32 overflow-y-auto font-mono text-xs">
          {systemMessages.map((msg, i) => (
            <div key={i} className="mb-1">
              <span className="text-neon-green/70">[SYSTEM]</span> {msg}
            </div>
          ))}
          {loading && <span className="typing-cursor">_</span>}
        </div>

        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label className="block text-xs mb-2 font-mono">OPERATIVE DESIGNATION</label>
            <div className="relative">
              <User className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-neon-green/50" />
              <Input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full bg-black/60 border border-neon-green/50 pl-10 py-2 font-mono text-neon-green focus:border-neon-green focus:ring-1 focus:ring-neon-green"
                required
                placeholder="Enter your operative name"
              />
            </div>
          </div>

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
                placeholder="Enter your email"
              />
            </div>
          </div>

          <div className="mb-4">
            <label className="block text-xs mb-2 font-mono">ACCESS CODE</label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-neon-green/50" />
              <Input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-black/60 border border-neon-green/50 pl-10 pr-10 py-2 font-mono text-neon-green focus:border-neon-green focus:ring-1 focus:ring-neon-green"
                required
                placeholder="Create access code"
                minLength={8}
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-neon-green/50 hover:text-neon-green"
              >
                {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>

            <div className="mt-2">
              <div className="flex justify-between mb-1">
                <span className="text-xs">Security Strength:</span>
                <span className="text-xs">
                  {passwordStrength === 4
                    ? "Maximum"
                    : passwordStrength === 3
                      ? "Strong"
                      : passwordStrength === 2
                        ? "Medium"
                        : passwordStrength === 1
                          ? "Weak"
                          : "None"}
                </span>
              </div>
              <div className="w-full h-1 bg-black/60 rounded-full overflow-hidden">
                <div
                  className={`h-full ${
                    passwordStrength === 4
                      ? "bg-green-500"
                      : passwordStrength === 3
                        ? "bg-neon-green"
                        : passwordStrength === 2
                          ? "bg-yellow-500"
                          : passwordStrength === 1
                            ? "bg-red-500"
                            : "bg-gray-700"
                  }`}
                  style={{ width: `${passwordStrength * 25}%` }}
                ></div>
              </div>

              <div className="grid grid-cols-2 gap-2 mt-2 text-xs">
                <div className="flex items-center">
                  {password.length >= 8 ? (
                    <CheckCircle className="h-3 w-3 text-green-500 mr-1" />
                  ) : (
                    <XCircle className="h-3 w-3 text-red-500 mr-1" />
                  )}
                  <span>8+ characters</span>
                </div>
                <div className="flex items-center">
                  {/[A-Z]/.test(password) ? (
                    <CheckCircle className="h-3 w-3 text-green-500 mr-1" />
                  ) : (
                    <XCircle className="h-3 w-3 text-red-500 mr-1" />
                  )}
                  <span>Uppercase</span>
                </div>
                <div className="flex items-center">
                  {/[0-9]/.test(password) ? (
                    <CheckCircle className="h-3 w-3 text-green-500 mr-1" />
                  ) : (
                    <XCircle className="h-3 w-3 text-red-500 mr-1" />
                  )}
                  <span>Number</span>
                </div>
                <div className="flex items-center">
                  {/[^A-Za-z0-9]/.test(password) ? (
                    <CheckCircle className="h-3 w-3 text-green-500 mr-1" />
                  ) : (
                    <XCircle className="h-3 w-3 text-red-500 mr-1" />
                  )}
                  <span>Special char</span>
                </div>
              </div>
            </div>
          </div>

          <div className="mb-6">
            <label className="block text-xs mb-2 font-mono">VERIFY ACCESS CODE</label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-neon-green/50" />
              <Input
                type={showPassword ? "text" : "password"}
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                className={`w-full bg-black/60 border pl-10 py-2 font-mono text-neon-green focus:ring-1 ${
                  confirmPassword && password !== confirmPassword
                    ? "border-red-500 focus:border-red-500 focus:ring-red-500"
                    : "border-neon-green/50 focus:border-neon-green focus:ring-neon-green"
                }`}
                required
                placeholder="Confirm access code"
                minLength={8}
              />
              {confirmPassword && password !== confirmPassword && (
                <div className="text-red-500 text-xs mt-1 flex items-center">
                  <XCircle className="h-3 w-3 mr-1" />
                  Access codes do not match
                </div>
              )}
            </div>
          </div>

          <Button
            type="submit"
            disabled={loading || password !== confirmPassword || passwordStrength < 3}
            variant="outline"
            className="w-full border border-neon-green py-2 px-4 hover:bg-neon-green/20 transition-all text-neon-green font-mono disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <div className="flex items-center">
                <Shield className="animate-pulse mr-2 h-4 w-4" />
                CREATING CLEARANCE...
              </div>
            ) : (
              <div className="flex items-center">
                <Shield className="mr-2 h-4 w-4" />
                CREATE SECURITY CLEARANCE
              </div>
            )}
          </Button>
        </form>

        <div className="mt-6 text-center text-sm font-mono">
          <p className="text-neon-green/70">Already have clearance?</p>
          <Link href="/login" className="text-neon-green hover:text-neon-blue transition-colors mt-2 inline-block">
            ACCESS SECURE TERMINAL
          </Link>
        </div>
      </motion.div>
    </div>
  )
}
