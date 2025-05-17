"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { useSession } from "next-auth/react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import DashboardHeader from "@/components/dashboard/dashboard-header"
import DashboardNav from "@/components/dashboard/dashboard-nav"
import OverviewTab from "@/components/dashboard/overview-tab"
import ThreatActorsTab from "@/components/dashboard/threat-actors-tab"
import IntelligenceTab from "@/components/dashboard/intelligence-tab"
import PredictionsTab from "@/components/dashboard/predictions-tab"
import WorldMapTab from "@/components/dashboard/world-map-tab"
import { fetchDashboardOverview } from "@/lib/api"
import { useToast } from "@/hooks/use-toast"
import LoadingScreen from "@/components/ui/loading-screen"
import { motion } from "framer-motion"

export default function Dashboard() {
  const { data: session, status } = useSession()
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [overviewData, setOverviewData] = useState(null)
  const router = useRouter()
  const { toast } = useToast()

  useEffect(() => {
    if (status === "unauthenticated") {
      router.push("/login")
      return
    }

    if (status === "authenticated") {
      const fetchData = async () => {
        try {
          const data = await fetchDashboardOverview()
          setOverviewData(data)
          setLoading(false)
        } catch (err) {
          console.error("Error fetching data:", err)
          setError("Failed to connect to Echelon Neural Network. Check connection parameters.")
          toast({
            variant: "destructive",
            title: "Connection Error",
            description: "Failed to establish secure connection to Echelon API.",
          })
          setLoading(false)
        }
      }

      fetchData()
    }
  }, [status, router, toast])

  if (status === "loading" || loading) {
    return <LoadingScreen message="INITIALIZING ECHELON NEURAL NETWORK" />
  }

  if (error) {
    return (
      <div className="min-h-screen bg-terminal-black flex items-center justify-center p-4">
        <div className="terminal-container p-8 max-w-md text-center border border-terminal-red border-opacity-50 bg-black bg-opacity-70 rounded shadow-glow">
          <h2 className="text-2xl mb-4 text-terminal-red font-orbitron">CONNECTION ERROR</h2>
          <p className="mb-4 text-neon-green font-mono">{error}</p>
          <div className="text-xs text-terminal-red text-opacity-70 animate-pulse">RETRYING CONNECTION IN 10s...</div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-terminal-black text-neon-green font-mono relative overflow-hidden">
      <div className="absolute inset-0 bg-grid-pattern opacity-10"></div>
      <div className="absolute inset-0 bg-radial-gradient"></div>

      <DashboardHeader user={session?.user} />

      <div className="flex flex-col md:flex-row">
        <DashboardNav />

        <motion.main
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="flex-1 p-4 relative z-10"
        >
          <Tabs defaultValue="overview" className="w-full">
            <TabsList className="grid grid-cols-5 mb-4 bg-terminal-black border-b border-neon-green/20">
              <TabsTrigger
                value="overview"
                className="data-[state=active]:border-b-2 data-[state=active]:border-neon-green data-[state=active]:bg-neon-green/10 rounded-none"
              >
                DASHBOARD
              </TabsTrigger>
              <TabsTrigger
                value="threats"
                className="data-[state=active]:border-b-2 data-[state=active]:border-neon-green data-[state=active]:bg-neon-green/10 rounded-none"
              >
                THREAT ACTORS
              </TabsTrigger>
              <TabsTrigger
                value="intelligence"
                className="data-[state=active]:border-b-2 data-[state=active]:border-neon-green data-[state=active]:bg-neon-green/10 rounded-none"
              >
                NEURAL INSIGHTS
              </TabsTrigger>
              <TabsTrigger
                value="predictions"
                className="data-[state=active]:border-b-2 data-[state=active]:border-neon-green data-[state=active]:bg-neon-green/10 rounded-none"
              >
                PREDICTIONS
              </TabsTrigger>
              <TabsTrigger
                value="worldmap"
                className="data-[state=active]:border-b-2 data-[state=active]:border-neon-green data-[state=active]:bg-neon-green/10 rounded-none"
              >
                GLOBAL THREATS
              </TabsTrigger>
            </TabsList>

            <TabsContent value="overview">
              <OverviewTab loading={false} />
            </TabsContent>

            <TabsContent value="threats">
              <ThreatActorsTab loading={false} />
            </TabsContent>

            <TabsContent value="intelligence">
              <IntelligenceTab loading={false} />
            </TabsContent>

            <TabsContent value="predictions">
              <PredictionsTab loading={false} />
            </TabsContent>

            <TabsContent value="worldmap">
              <WorldMapTab loading={false} />
            </TabsContent>
          </Tabs>
        </motion.main>
      </div>

      <footer className="border-t border-neon-green/20 p-4 text-xs text-neon-green relative z-10">
        <div className="flex justify-between">
          <div>ECHELON NEURAL NETWORK • VERSION 4.7.2 • QUANTUM-SECURED</div>
          <div className="flex items-center">
            <span className="pulse-dot mr-2"></span>
            LIVE DATA FEED • MODEL ACCURACY: 97.3%
          </div>
        </div>
      </footer>
    </div>
  )
}
