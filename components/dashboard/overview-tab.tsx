"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  fetchDashboardOverview,
  fetchDashboardAttackTechniques,
  fetchDashboardThreatActors,
  fetchDashboardTemporal,
} from "@/lib/api"
import TemporalForecastChart from "@/components/charts/temporal-forecast-chart"
import ThreatLevelGauge from "@/components/charts/threat-level-gauge"
import AttackTechniquesChart from "@/components/charts/attack-techniques-chart"
import LoadingSpinner from "@/components/ui/loading-spinner"
import { motion } from "framer-motion"
import { Shield, Zap, Target, Globe, AlertTriangle } from "lucide-react"

interface OverviewTabProps {
  loading: boolean
}

export default function OverviewTab({ loading }: OverviewTabProps) {
  const [overviewData, setOverviewData] = useState<any>(null)
  const [techniqueData, setTechniqueData] = useState<any>(null)
  const [actorData, setActorData] = useState<any>(null)
  const [temporalData, setTemporalData] = useState<any>(null)
  const [localLoading, setLocalLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [overview, techniques, actors, temporal] = await Promise.all([
          fetchDashboardOverview(),
          fetchDashboardAttackTechniques(),
          fetchDashboardThreatActors(),
          fetchDashboardTemporal(),
        ])

        setOverviewData(overview)
        setTechniqueData(techniques)
        setActorData(actors)
        setTemporalData(temporal)
        setLocalLoading(false)
      } catch (err) {
        console.error("Error fetching overview data:", err)
        setError("Failed to fetch data from Neural Network")
        setLocalLoading(false)
      }
    }

    fetchData()
  }, [])

  if (loading || localLoading) {
    return <LoadingSpinner message="INITIALIZING THREAT INTELLIGENCE FEED" />
  }

  if (error) {
    return (
      <div className="terminal-container p-6 text-center">
        <div className="text-xl mb-4 text-terminal-red font-orbitron">DATA FETCH ERROR</div>
        <p className="text-neon-green mb-4 font-mono">{error}</p>
      </div>
    )
  }

  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  }

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 },
  }

  return (
    <motion.div variants={container} initial="hidden" animate="show" className="space-y-6">
      {/* Threat level overview */}
      <motion.div variants={item}>
        <Card className="bg-black/60 border-neon-green/30 shadow-glow overflow-hidden">
          <CardHeader className="pb-2">
            <CardTitle className="text-xl font-orbitron flex items-center">
              <AlertTriangle className="mr-2 h-5 w-5 text-neon-green" />
              THREAT INTELLIGENCE OVERVIEW
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="col-span-1">
                <ThreatLevelGauge
                  level={overviewData?.threat_summary.overall_threat_level}
                  value={overviewData?.threat_summary.key_metrics.prediction_confidence * 100}
                />
              </div>
              <div className="col-span-1 md:col-span-2">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-black/40 border border-neon-green/20 p-4 rounded-md">
                    <div className="flex items-center mb-2">
                      <Shield className="h-4 w-4 mr-2 text-neon-green" />
                      <div className="text-xs text-neon-green/70">THREAT ACTORS</div>
                    </div>
                    <div className="text-3xl font-orbitron">{overviewData?.threat_summary.active_threat_actors}</div>
                  </div>

                  <div className="bg-black/40 border border-neon-green/20 p-4 rounded-md">
                    <div className="flex items-center mb-2">
                      <Zap className="h-4 w-4 mr-2 text-neon-green" />
                      <div className="text-xs text-neon-green/70">TECHNIQUES</div>
                    </div>
                    <div className="text-3xl font-orbitron">{overviewData?.threat_summary.observed_techniques}</div>
                  </div>

                  <div className="bg-black/40 border border-neon-green/20 p-4 rounded-md">
                    <div className="flex items-center mb-2">
                      <Target className="h-4 w-4 mr-2 text-neon-green" />
                      <div className="text-xs text-neon-green/70">SECTORS</div>
                    </div>
                    <div className="text-3xl font-orbitron">{overviewData?.threat_summary.targeted_sectors}</div>
                  </div>

                  <div className="bg-black/40 border border-neon-green/20 p-4 rounded-md">
                    <div className="flex items-center mb-2">
                      <Globe className="h-4 w-4 mr-2 text-neon-green" />
                      <div className="text-xs text-neon-green/70">COUNTRIES</div>
                    </div>
                    <div className="text-3xl font-orbitron">{overviewData?.threat_summary.affected_countries}</div>
                  </div>
                </div>

                <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-black/40 border border-neon-green/20 p-4 rounded-md">
                    <div className="text-xs text-neon-green/70 mb-2">KEY INTELLIGENCE</div>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>Top Threat Actor:</span>
                        <span className="text-white">{overviewData?.threat_summary.highlights.top_threat_actor}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Most Targeted:</span>
                        <span className="text-white">
                          {overviewData?.threat_summary.highlights.most_targeted_sector}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Peak Activity:</span>
                        <span className="text-white">{overviewData?.threat_summary.highlights.peak_activity_day}</span>
                      </div>
                    </div>
                  </div>

                  <div className="bg-black/40 border border-neon-green/20 p-4 rounded-md">
                    <div className="text-xs text-neon-green/70 mb-2">NEURAL METRICS</div>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>Data Points:</span>
                        <span className="text-white">
                          {overviewData?.threat_summary.key_metrics.data_points.toLocaleString()}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Real CVEs:</span>
                        <span className="text-white">
                          {overviewData?.threat_summary.key_metrics.real_cves.toLocaleString()}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Prediction Confidence:</span>
                        <span className="text-white">
                          {(overviewData?.threat_summary.key_metrics.prediction_confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Attack techniques and threat actors */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <motion.div variants={item}>
          <Card className="bg-black/60 border-neon-green/30 shadow-glow h-full">
            <CardHeader className="pb-2">
              <CardTitle className="text-xl font-orbitron border-b border-neon-green/30 pb-2">
                ATTACK TECHNIQUES
              </CardTitle>
            </CardHeader>
            <CardContent>{techniqueData && <AttackTechniquesChart data={techniqueData.techniques} />}</CardContent>
          </Card>
        </motion.div>

        <motion.div variants={item}>
          <Card className="bg-black/60 border-neon-green/30 shadow-glow h-full">
            <CardHeader className="pb-2">
              <CardTitle className="text-xl font-orbitron border-b border-neon-green/30 pb-2">
                ACTIVE THREAT ACTORS
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {actorData?.threat_actors.slice(0, 5).map((actor: any, index: number) => (
                  <div
                    key={index}
                    className="flex justify-between items-center bg-black/40 p-3 rounded-md border border-neon-green/20 hover:border-neon-green/50 transition-all"
                  >
                    <div className="flex items-center">
                      <div className="w-2 h-2 bg-neon-green rounded-full mr-2 pulse-dot"></div>
                      <div>
                        <div className="font-medium">{actor.name}</div>
                        <div className="text-xs text-neon-green/70">GROUP: {actor.group}</div>
                      </div>
                    </div>
                    <div className="flex flex-col items-end">
                      <div className="text-md font-bold">{actor.activity_count}</div>
                      <div className="text-xs text-neon-green/70">
                        CONF: {(actor.average_confidence * 100).toFixed(0)}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Temporal forecast */}
      <motion.div variants={item}>
        <Card className="bg-black/60 border-neon-green/30 shadow-glow">
          <CardHeader className="pb-2">
            <CardTitle className="text-xl font-orbitron border-b border-neon-green/30 pb-2">
              TEMPORAL FORECAST (30 DAYS)
            </CardTitle>
          </CardHeader>
          <CardContent>
            {temporalData && <TemporalForecastChart data={temporalData.temporal_forecast} />}
            <div className="flex justify-between text-xs text-neon-green/70 mt-2">
              <div>PEAK: {temporalData?.analysis.peak_day.date}</div>
              <div>TROUGH: {temporalData?.analysis.trough_day.date}</div>
              <div>AVG: {temporalData?.analysis.average_activity.toFixed(2)}</div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  )
}
