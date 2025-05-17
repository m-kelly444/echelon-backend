"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { fetchPredictAttacks } from "@/lib/api"
import LoadingSpinner from "@/components/ui/loading-spinner"

interface PredictionsTabProps {
  loading: boolean
}

export default function PredictionsTab({ loading }: PredictionsTabProps) {
  const [predictData, setPredictData] = useState<any>(null)
  const [localLoading, setLocalLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await fetchPredictAttacks()
        setPredictData(data)
        setLocalLoading(false)
      } catch (err) {
        console.error("Error fetching predictions data:", err)
        setError("Failed to fetch data from API")
        setLocalLoading(false)
      }
    }

    fetchData()
  }, [])

  if (loading || localLoading) {
    return <LoadingSpinner message="INITIALIZING PREDICTION ENGINE" />
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
            ATTACK PREDICTIONS
          </CardTitle>
          <div className="text-sm mt-2">
            Generated: {new Date(predictData?.generated_at).toLocaleString()} | Forecast Period:{" "}
            {predictData?.forecast_period.days} days
          </div>
        </CardHeader>
        <CardContent>
          <h3 className="text-lg mb-3">TARGET SECTORS</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
            {predictData?.target_sectors.map((sector: any, index: number) => (
              <div key={index} className="border border-terminal-green border-opacity-30 p-3 rounded text-center">
                <div className="text-xs mb-1 text-terminal-green text-opacity-70">SECTOR</div>
                <div className="mb-2 truncate">{sector.sector}</div>
                <div className="text-xs mb-1 text-terminal-green text-opacity-70">RISK</div>
                <div className="text-2xl">{sector.risk_score}%</div>
              </div>
            ))}
          </div>

          <h3 className="text-lg mb-3">THREAT ACTORS</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
            {predictData?.threat_actors.map((actor: any, index: number) => (
              <div key={index} className="border border-terminal-green border-opacity-30 p-3 rounded">
                <div className="flex justify-between items-center mb-2">
                  <div className="font-bold">{actor.name}</div>
                  <div className="text-xs">Group: {actor.group}</div>
                </div>
                <div className="flex justify-between text-sm">
                  <div>Activity: {actor.activity_count}</div>
                  <div>Confidence: {(actor.average_confidence * 100).toFixed(0)}%</div>
                </div>
              </div>
            ))}
          </div>

          <h3 className="text-lg mb-3">ACTIVITY FORECAST</h3>
          <div className="overflow-x-auto mb-6">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-terminal-green border-opacity-30">
                  <th className="text-left pb-2">DATE</th>
                  <th className="text-right pb-2">ACTIVITY</th>
                  <th className="text-right pb-2">CONFIDENCE</th>
                </tr>
              </thead>
              <tbody>
                {predictData?.activity_forecast.slice(0, 7).map((forecast: any, index: number) => (
                  <tr key={index} className="border-b border-terminal-green border-opacity-10">
                    <td className="py-2">{forecast.date}</td>
                    <td className="text-right py-2">{forecast.expected_activity.toFixed(2)}</td>
                    <td className="text-right py-2">{(forecast.confidence * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h3 className="text-lg mb-3">ATTACK VECTORS</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {predictData?.attack_vectors.map((vector: any, index: number) => (
              <div
                key={index}
                className="border border-terminal-green border-opacity-30 p-3 rounded flex justify-between items-center"
              >
                <div className="text-sm">{vector.vector}</div>
                <div className="bg-terminal-green bg-opacity-20 px-2 py-1 rounded">{vector.frequency}</div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
