"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { EchelonLogo } from "@/components/echelon-logo"
import { signOut } from "next-auth/react"
import { Bell, LogOut, User, Settings, ChevronDown, AlertTriangle } from "lucide-react"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { motion } from "framer-motion"

interface DashboardHeaderProps {
  user?: {
    name?: string | null
    email?: string | null
    image?: string | null
  } | null
}

export default function DashboardHeader({ user }: DashboardHeaderProps) {
  const [currentTime, setCurrentTime] = useState("")
  const [alerts, setAlerts] = useState<string[]>([])
  const router = useRouter()

  useEffect(() => {
    // Update time every second
    const timer = setInterval(() => {
      const now = new Date()
      setCurrentTime(now.toISOString().replace("T", " ").substring(0, 19))
    }, 1000)

    // Simulate incoming alerts
    const alertTimer = setInterval(() => {
      if (Math.random() > 0.7) {
        const newAlert = getRandomAlert()
        setAlerts((prev) => [newAlert, ...prev].slice(0, 5))
      }
    }, 30000)

    // Initial alerts
    setAlerts([
      "Anomalous activity detected in financial sector",
      "New zero-day vulnerability identified: CVE-2025-1337",
    ])

    return () => {
      clearInterval(timer)
      clearInterval(alertTimer)
    }
  }, [])

  const getRandomAlert = () => {
    const alerts = [
      "Suspicious traffic detected from Eastern European IPs",
      "Potential data exfiltration attempt blocked",
      "New threat actor identified: QUANTUM SPHINX",
      "Brute force attack detected on authentication systems",
      "Unusual API call patterns detected in US-WEST region",
      "Potential supply chain compromise identified",
      "Ransomware signature detected in network traffic",
      "Critical infrastructure targeting increased by 27%",
    ]
    return alerts[Math.floor(Math.random() * alerts.length)]
  }

  const handleLogout = async () => {
    await signOut({ redirect: true, callbackUrl: "/login" })
  }

  return (
    <motion.header
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="flex justify-between items-center p-4 border-b border-neon-green/20 bg-black/60 backdrop-blur-sm relative z-20"
    >
      <div className="flex items-center">
        <EchelonLogo className="w-10 h-10 mr-3" />
        <div>
          <h1 className="text-3xl font-orbitron font-bold text-glow-lg">ECHELON</h1>
          <p className="text-xs text-neon-green/70">NEURAL THREAT INTELLIGENCE • QUANTUM-SECURED</p>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" className="relative">
              <Bell className="h-5 w-5 text-neon-green" />
              {alerts.length > 0 && (
                <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center">
                  {alerts.length}
                </span>
              )}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-80 bg-black border border-neon-green/30">
            <DropdownMenuLabel className="font-mono text-neon-green">
              <div className="flex items-center">
                <AlertTriangle className="h-4 w-4 mr-2 text-red-500" />
                Active Threat Alerts
              </div>
            </DropdownMenuLabel>
            <DropdownMenuSeparator className="bg-neon-green/20" />
            {alerts.length > 0 ? (
              alerts.map((alert, index) => (
                <DropdownMenuItem key={index} className="py-3 font-mono text-xs cursor-pointer hover:bg-neon-green/10">
                  <div>
                    <div className="flex items-center">
                      <span className="pulse-dot mr-2 bg-red-500"></span>
                      <span className="text-neon-green/70">{new Date().toISOString().substring(0, 10)}</span>
                    </div>
                    <div className="mt-1">{alert}</div>
                  </div>
                </DropdownMenuItem>
              ))
            ) : (
              <div className="p-3 text-center text-neon-green/50 text-sm">No active alerts</div>
            )}
          </DropdownMenuContent>
        </DropdownMenu>

        <div className="text-right hidden md:block">
          <div className="text-md font-mono text-neon-green">{currentTime}</div>
          <div className="flex items-center justify-end text-xs text-neon-green/70">
            <span className="pulse-dot mr-2"></span>
            NEURAL NETWORK ACTIVE
          </div>
        </div>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="flex items-center gap-2 hover:bg-neon-green/10">
              <div className="h-8 w-8 rounded-full bg-neon-green/20 flex items-center justify-center">
                <User className="h-4 w-4 text-neon-green" />
              </div>
              <div className="hidden md:block text-left">
                <div className="text-sm">{user?.name || "Operative"}</div>
                <div className="text-xs text-neon-green/70">{user?.email || "Unknown"}</div>
              </div>
              <ChevronDown className="h-4 w-4 text-neon-green/70" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="bg-black border border-neon-green/30">
            <DropdownMenuLabel className="font-mono text-neon-green">User Profile</DropdownMenuLabel>
            <DropdownMenuSeparator className="bg-neon-green/20" />
            <DropdownMenuItem className="font-mono cursor-pointer hover:bg-neon-green/10">
              <User className="mr-2 h-4 w-4" />
              <span>Profile</span>
            </DropdownMenuItem>
            <DropdownMenuItem className="font-mono cursor-pointer hover:bg-neon-green/10">
              <Settings className="mr-2 h-4 w-4" />
              <span>Settings</span>
            </DropdownMenuItem>
            <DropdownMenuSeparator className="bg-neon-green/20" />
            <DropdownMenuItem className="font-mono cursor-pointer hover:bg-neon-green/10" onClick={handleLogout}>
              <LogOut className="mr-2 h-4 w-4" />
              <span>Logout</span>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </motion.header>
  )
}
