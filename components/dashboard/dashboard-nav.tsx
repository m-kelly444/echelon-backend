"use client"

import { useState } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { motion } from "framer-motion"
import {
  Home,
  Shield,
  Brain,
  LineChart,
  Settings,
  Globe,
  Menu,
  X,
  Terminal,
  Database,
  FileText,
  AlertTriangle,
} from "lucide-react"

export default function DashboardNav() {
  const pathname = usePathname()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const navItems = [
    { name: "DASHBOARD", icon: Home, href: "/dashboard" },
    { name: "THREAT ACTORS", icon: Shield, href: "/dashboard?tab=threats" },
    { name: "NEURAL INSIGHTS", icon: Brain, href: "/dashboard?tab=intelligence" },
    { name: "PREDICTIONS", icon: LineChart, href: "/dashboard?tab=predictions" },
    { name: "GLOBAL THREATS", icon: Globe, href: "/dashboard?tab=worldmap" },
    { name: "TERMINAL", icon: Terminal, href: "/dashboard/terminal" },
    { name: "REPORTS", icon: FileText, href: "/dashboard/reports" },
    { name: "VULNERABILITIES", icon: AlertTriangle, href: "/dashboard/vulnerabilities" },
    { name: "DATA SOURCES", icon: Database, href: "/dashboard/data-sources" },
    { name: "SETTINGS", icon: Settings, href: "/dashboard/settings" },
  ]

  return (
    <>
      {/* Mobile menu button */}
      <div className="md:hidden fixed top-20 left-4 z-30">
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="p-2 rounded-md bg-black/80 border border-neon-green/30"
        >
          {mobileMenuOpen ? <X className="h-6 w-6 text-neon-green" /> : <Menu className="h-6 w-6 text-neon-green" />}
        </button>
      </div>

      {/* Mobile menu */}
      {mobileMenuOpen && (
        <motion.div
          initial={{ opacity: 0, x: -100 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -100 }}
          className="fixed inset-0 z-20 bg-black/95 md:hidden"
        >
          <div className="pt-20 p-4">
            <nav className="space-y-2">
              {navItems.map((item) => (
                <Link
                  key={item.name}
                  href={item.href}
                  className="flex items-center gap-3 px-3 py-3 rounded hover:bg-neon-green/10 transition-all"
                  onClick={() => setMobileMenuOpen(false)}
                >
                  <item.icon className="h-5 w-5 text-neon-green" />
                  <span>{item.name}</span>
                </Link>
              ))}
            </nav>
          </div>
        </motion.div>
      )}

      {/* Desktop sidebar */}
      <motion.nav
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="hidden md:block w-64 min-h-[calc(100vh-64px)] border-r border-neon-green/20 p-4 bg-black/60 backdrop-blur-sm relative z-10"
      >
        <div className="space-y-1">
          {navItems.map((item) => (
            <Link
              key={item.name}
              href={item.href}
              className={`flex items-center gap-3 px-3 py-2 rounded transition-all ${
                pathname === item.href ||
                (pathname === "/dashboard" && item.href === "/dashboard") ||
                (pathname.includes(item.href) && item.href !== "/dashboard")
                  ? "bg-neon-green/20 text-neon-green"
                  : "hover:bg-neon-green/10 text-neon-green/80"
              }`}
            >
              <item.icon className="h-4 w-4" />
              <span className="text-sm">{item.name}</span>
            </Link>
          ))}
        </div>

        <div className="absolute bottom-4 left-4 right-4">
          <div className="border border-neon-green/20 rounded p-3 bg-black/40">
            <div className="text-xs text-neon-green/70 mb-2">SYSTEM STATUS</div>
            <div className="flex items-center justify-between text-xs mb-1">
              <span>Neural Network</span>
              <span className="flex items-center text-green-500">
                <span className="pulse-dot mr-1 bg-green-500"></span>
                Online
              </span>
            </div>
            <div className="flex items-center justify-between text-xs mb-1">
              <span>Quantum Encryption</span>
              <span className="flex items-center text-green-500">
                <span className="pulse-dot mr-1 bg-green-500"></span>
                Active
              </span>
            </div>
            <div className="flex items-center justify-between text-xs">
              <span>Threat Database</span>
              <span className="flex items-center text-green-500">
                <span className="pulse-dot mr-1 bg-green-500"></span>
                Synced
              </span>
            </div>
          </div>
        </div>
      </motion.nav>
    </>
  )
}
