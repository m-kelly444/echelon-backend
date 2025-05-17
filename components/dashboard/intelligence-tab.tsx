"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { fetchMlInsights, fetchMlInsightsTrends, fetchMlInsightsPredictions } from "@/lib/api"
import LoadingSpinner from "@/components/ui/loading-spinner"

interface IntelligenceTabProps {
  loading: boolean
}

export default function IntelligenceTab({ loading }: IntelligenceTabProps) {
  const [insightsData, setInsightsData] = useState<any>(null)
  const [trendsData, setTrendsData] = useState<any>(null)
  const [predictionsData, setPredictionsData] = useState<any>(null)
  const [localLoading, setLocalLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [insights, trends, predictions] = await Promise.all([
          fetchMlInsights(),
          fetchMlInsightsTrends(),
          fetchMlInsightsPredictions(),
        ])

        setInsightsData(insights)
        setTrendsData(trends)
        setPredictionsData(predictions)
        setLocalLoading(false)
      } catch (err) {
        console.error("Error fetching ML insights data:", err)
        setError("Failed to fetch data from API")
        setLocalLoading(false)
      }
    }

    fetchData()
  }, [])

  if (loading || localLoading) {
    return <LoadingSpinner message="INITIALIZING ML INSIGHTS" />
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
            ML THREAT ASSESSMENT
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="border border-terminal-green border-opacity-30 p-4 rounded">
              <div className="text-xs mb-1 text-terminal-green text-opacity-70">CURRENT LEVEL</div>
              <div className="text-2xl">{insightsData?.overall_threat_assessment.current_level}</div>
            </div>
            <div className="border border-terminal-green border-opacity-30 p-4 rounded">
              <div className="text-xs mb-1 text-terminal-green text-opacity-70">CONFIDENCE</div>
              <div className="text-2xl">{(insightsData?.overall_threat_assessment.confidence * 100).toFixed(1)}%</div>
            </div>
            <div className="border border-terminal-green border-opacity-30 p-4 rounded">
              <div className="text-xs mb-1 text-terminal-green text-opacity-70">TREND</div>
              <div className="text-2xl">{insightsData?.overall_threat_assessment.trend.direction.toUpperCase()}</div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="bg-black bg-opacity-70 border-terminal-green border-opacity-30 shadow-glow">
          <CardHeader className="pb-2">
            <CardTitle className="text-xl border-b border-terminal-green border-opacity-30 pb-2">
              SECTOR TRENDS
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-3">
              {trendsData?.trend_analysis.sector_trends.map((sector: any, index: number) => (
                <li key={index} className="flex justify-between items-center">
                  <span>{sector.sector}</span>
                  <div className="flex items-center">
                    <span
                      className={
                        sector.trend.direction === "increasing" ? "text-terminal-red mr-2" : "text-terminal-green mr-2"
                      }
                    >
                      {sector.trend.direction === "increasing" ? "▲" : "–"}
                    </span>
                    <span className="text-terminal-green">{sector.current_risk}%</span>
                  </div>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>

        <Card className="bg-black bg-opacity-70 border-terminal-green border-opacity-30 shadow-glow">
          <CardHeader className="pb-2">
            <CardTitle className="text-xl border-b border-terminal-green border-opacity-30 pb-2">
              TECHNIQUE TRENDS
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-3">
              {trendsData?.trend_analysis.technique_trends.map((technique: any, index: number) => (
                <li key={index} className="flex justify-between items-center">
                  <span>{technique.technique}</span>
                  <div className="flex items-center">
                    <span
                      className={
                        technique.trend.direction === "increasing"
                          ? "text-terminal-red mr-2"
                          : "text-terminal-green mr-2"
                      }
                    >
                      {technique.trend.direction === "increasing" ? "▲" : "–"}
                    </span>
                    <span className="text-terminal-green">{technique.current_count}</span>
                  </div>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </div>

      <Card className="bg-black bg-opacity-70 border-terminal-green border-opacity-30 shadow-glow">
        <CardHeader className="pb-2">
          <CardTitle className="text-xl border-b border-terminal-green border-opacity-30 pb-2">
            7-DAY FORECAST
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-7 gap-2">
            {predictionsData?.predictive_analytics.next_period_forecast.map((day: any, index: number) => (
              <div key={index} className="text-center">
                <div className="text-xs mb-1">DAY {day.day}</div>
                <div className="text-lg">{day.predicted_activity.toFixed(1)}</div>
                <div className="text-xs text-terminal-green text-opacity-70">{(day.confidence * 100).toFixed(0)}%</div>
              </div>
            ))}
          </div>
          <div className="mt-4 text-xs text-terminal-green text-opacity-70">
            CONFIDENCE INTERVALS: {predictionsData?.predictive_analytics.confidence_intervals.mean.toFixed(1)}[
            {predictionsData?.predictive_analytics.confidence_intervals.lower.toFixed(1)} -
            {predictionsData?.predictive_analytics.confidence_intervals.upper.toFixed(1)}]
          </div>
        </CardContent>
      </Card>

      <Card className="bg-black bg-opacity-70 border-terminal-green border-opacity-30 shadow-glow">
        <CardHeader className="pb-2">
          <CardTitle className="text-xl border-b border-terminal-green border-opacity-30 pb-2">DATA SOURCES</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <div className="text-xs mb-1 text-terminal-green text-opacity-70">CVEs</div>
              <div className="text-2xl">{insightsData?.data_based_on.cves}</div>
            </div>
            <div>
              <div className="text-xs mb-1 text-terminal-green text-opacity-70">APT MAPPINGS</div>
              <div className="text-2xl">{insightsData?.data_based_on.apt_mappings}</div>
            </div>
            <div>
              <div className="text-xs mb-1 text-terminal-green text-opacity-70">GEO LOCATIONS</div>
              <div className="text-2xl">{insightsData?.data_based_on.geo_locations}</div>
            </div>
          </div>
          <div className="mt-4 text-xs">
            <div className="flex justify-between">
              <span>MODEL ACCURACY</span>
              <span>{insightsData?.model_insights.accuracy * 100}%</span>
            </div>
            <div className="flex justify-between mt-1">
              <span>TRAINING DATE</span>
              <span>{new Date(insightsData?.model_insights.training_date).toLocaleString()}</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
