"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { fetchDashboardThreatActors, fetchPredictAttacks } from "@/lib/api"
import LoadingSpinner from "@/components/ui/loading-spinner"

interface ThreatActorsTabProps {
  loading: boolean
}

export default function ThreatActorsTab({ loading }: ThreatActorsTabProps) {
  const [actorsData, setActorsData] = useState<any>(null)
  const [predictData, setPredictData] = useState<any>(null)
  const [localLoading, setLocalLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [actors, predict] = await Promise.all([fetchDashboardThreatActors(), fetchPredictAttacks()])

        setActorsData(actors)
        setPredictData(predict)
        setLocalLoading(false)
      } catch (err) {
        console.error("Error fetching threat actors data:", err)
        setError("Failed to fetch data from API")
        setLocalLoading(false)
      }
    }

    fetchData()
  }, [])

  if (loading || localLoading) {
    return <LoadingSpinner message="INITIALIZING THREAT ACTOR DATABASE" />
  }

  if (error) {
    return (
      <div className="terminal-container p-6 text-center">
        <div className="text-xl mb-4 text-terminal-red">DATA FETCH ERROR</div>
        <p className="text-terminal-green mb-4">{error}</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <Card className="bg-black bg-opacity-70 border-terminal-green border-opacity-30 shadow-glow">
        <CardHeader className="pb-2">
          <CardTitle className="text-xl border-b border-terminal-green border-opacity-30 pb-2">
            ACTIVE THREAT ACTORS
          </CardTitle>
          <div className="text-xs mt-2">
            Total tracked APT groups: {actorsData?.analysis.total_actors} | Most active:{" "}
            {actorsData?.analysis.most_active}
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {actorsData?.threat_actors.map((actor: any, index: number) => (
              <div
                key={index}
                className="border border-terminal-green border-opacity-30 p-3 rounded hover:bg-terminal-green hover:bg-opacity-5 transition-all"
              >
                <div className="flex justify-between items-center">
                  <div className="text-lg font-bold">{actor.name}</div>
                  <div className="bg-terminal-green bg-opacity-20 px-2 py-1 rounded">{actor.activity_count}</div>
                </div>
                <div className="text-xs mt-2 text-terminal-green text-opacity-70">GROUP: {actor.group}</div>
                <div className="flex items-center mt-2">
                  <div className="text-sm mr-2">Confidence:</div>
                  <Progress value={actor.average_confidence * 100} className="h-2 flex-1" />
                  <div className="text-sm ml-2">{(actor.average_confidence * 100).toFixed(0)}%</div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card className="bg-black bg-opacity-70 border-terminal-green border-opacity-30 shadow-glow">
        <CardHeader className="pb-2">
          <CardTitle className="text-xl border-b border-terminal-green border-opacity-30 pb-2">
            ATTACK FORECAST
          </CardTitle>
          <div className="text-xs mt-2">
            Activity level: {predictData?.vulnerability_severity?.level?.toFixed(1)} | Forecast confidence:{" "}
            {(predictData?.activity_forecast?.[0]?.confidence * 100)?.toFixed(0)}%
          </div>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-terminal-green border-opacity-30">
                  <th className="text-left pb-2">Date</th>
                  <th className="text-right pb-2">Expected Activity</th>
                  <th className="text-right pb-2">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {predictData?.activity_forecast?.slice(0, 7).map((day: any, index: number) => (
                  <tr key={index} className="border-b border-terminal-green border-opacity-10">
                    <td className="py-2">{day.date}</td>
                    <td className="text-right py-2">{day.expected_activity.toFixed(2)}</td>
                    <td className="text-right py-2">{(day.confidence * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      <Card className="bg-black bg-opacity-70 border-terminal-green border-opacity-30 shadow-glow">
        <CardHeader className="pb-2">
          <CardTitle className="text-xl border-b border-terminal-green border-opacity-30 pb-2">
            TARGET SECTORS
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {predictData?.target_sectors?.map((sector: any, index: number) => (
              <div
                key={index}
                className="border border-terminal-green border-opacity-30 p-3 rounded text-center sector-glow"
              >
                <div className="text-xs mb-1 text-terminal-green text-opacity-70">SECTOR</div>
                <div className="mb-2 truncate">{sector.sector}</div>
                <div className="text-xs mb-1 text-terminal-green text-opacity-70">RISK SCORE</div>
                <div className="text-xl">{sector.risk_score}%</div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
