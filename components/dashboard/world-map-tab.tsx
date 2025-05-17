"use client"

import { useEffect, useState, useRef } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { fetchDashboardGeographic, fetchGeoData } from "@/lib/api"
import LoadingSpinner from "@/components/ui/loading-spinner"
import { motion } from "framer-motion"
import { Globe, AlertTriangle } from "lucide-react"
import * as d3 from "d3"
import { feature } from "topojson-client"

interface WorldMapTabProps {
  loading: boolean
}

export default function WorldMapTab({ loading }: WorldMapTabProps) {
  const [geoData, setGeoData] = useState<any>(null)
  const [attackData, setAttackData] = useState<any>(null)
  const [localLoading, setLocalLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const mapRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [geo, attacks] = await Promise.all([fetchGeoData(), fetchDashboardGeographic()])

        setGeoData(geo)
        setAttackData(attacks)
        setLocalLoading(false)
      } catch (err) {
        console.error("Error fetching geographic data:", err)
        setError("Failed to fetch geographic data from Neural Network")
        setLocalLoading(false)
      }
    }

    fetchData()
  }, [])

  useEffect(() => {
    if (!mapRef.current || !geoData || !attackData) return

    // Clear previous map
    d3.select(mapRef.current).selectAll("*").remove()

    const width = mapRef.current.clientWidth
    const height = 500

    // Create SVG
    const svg = d3.select(mapRef.current).append("svg").attr("width", width).attr("height", height)

    // Add glow effect
    const defs = svg.append("defs")
    const filter = defs.append("filter").attr("id", "glow-map")

    filter.append("feGaussianBlur").attr("stdDeviation", "2.5").attr("result", "coloredBlur")

    const feMerge = filter.append("feMerge")
    feMerge.append("feMergeNode").attr("in", "coloredBlur")
    feMerge.append("feMergeNode").attr("in", "SourceGraphic")

    // Create projection
    const projection = d3
      .geoMercator()
      .scale(width / 2 / Math.PI)
      .center([0, 20])
      .translate([width / 2, height / 2])

    // Create path generator
    const path = d3.geoPath().projection(projection)

    // Convert TopoJSON to GeoJSON
    const world = feature(geoData, geoData.objects.countries)

    // Draw countries
    svg
      .selectAll(".country")
      .data(world.features)
      .enter()
      .append("path")
      .attr("class", "country")
      .attr("d", path as any)
      .attr("fill", "rgba(0, 255, 136, 0.1)")
      .attr("stroke", "rgba(0, 255, 136, 0.3)")
      .attr("stroke-width", 0.5)

    // Add attack points
    const attackPoints = attackData.attacks || []

    // Scale for attack intensity
    const intensityScale = d3
      .scaleLinear()
      .domain([0, d3.max(attackPoints, (d: any) => d.intensity) || 10])
      .range([3, 15])

    // Add attack points
    svg
      .selectAll(".attack-point")
      .data(attackPoints)
      .enter()
      .append("circle")
      .attr("class", "attack-point")
      .attr("cx", (d: any) => projection([d.longitude, d.latitude])?.[0] || 0)
      .attr("cy", (d: any) => projection([d.longitude, d.latitude])?.[1] || 0)
      .attr("r", (d: any) => intensityScale(d.intensity))
      .attr("fill", "rgba(255, 0, 0, 0.5)")
      .attr("stroke", "#FF0000")
      .attr("stroke-width", 1)
      .attr("filter", "url(#glow-map)")
      .attr("opacity", 0.7)
      .append("title")
      .text((d: any) => `${d.country}: ${d.intensity} attacks`)

    // Add attack lines (connections between attacks)
    const connections = attackData.connections || []

    svg
      .selectAll(".connection")
      .data(connections)
      .enter()
      .append("line")
      .attr("class", "connection")
      .attr("x1", (d: any) => projection([d.source.longitude, d.source.latitude])?.[0] || 0)
      .attr("y1", (d: any) => projection([d.source.longitude, d.source.latitude])?.[1] || 0)
      .attr("x2", (d: any) => projection([d.target.longitude, d.target.latitude])?.[0] || 0)
      .attr("y2", (d: any) => projection([d.target.longitude, d.target.latitude])?.[1] || 0)
      .attr("stroke", "rgba(255, 0, 0, 0.3)")
      .attr("stroke-width", (d: any) => d.strength / 2)
      .attr("stroke-dasharray", "3,3")
      .attr("filter", "url(#glow-map)")

    // Add pulsing effect to attack points
    svg.selectAll(".attack-point").each(function () {
      const circle = d3.select(this)
      const radius = Number.parseFloat(circle.attr("r"))

      svg
        .append("circle")
        .attr("cx", circle.attr("cx"))
        .attr("cy", circle.attr("cy"))
        .attr("r", radius)
        .attr("fill", "none")
        .attr("stroke", "#FF0000")
        .attr("stroke-width", 1)
        .attr("opacity", 0.5)
        .attr("filter", "url(#glow-map)")
        .style("animation", `pulse-map ${2 + Math.random() * 3}s infinite`)
    })
  }, [geoData, attackData])

  if (loading || localLoading) {
    return <LoadingSpinner message="INITIALIZING GLOBAL THREAT MAP" />
  }

  if (error) {
    return (
      <div className="terminal-container p-6 text-center">
        <div className="text-xl mb-4 text-terminal-red font-orbitron">DATA FETCH ERROR</div>
        <p className="text-neon-green mb-4 font-mono">{error}</p>
      </div>
    )
  }

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }} className="space-y-6">
      <Card className="bg-black/60 border-neon-green/30 shadow-glow overflow-hidden">
        <CardHeader className="pb-2">
          <CardTitle className="text-xl font-orbitron flex items-center">
            <Globe className="mr-2 h-5 w-5 text-neon-green" />
            GLOBAL THREAT INTELLIGENCE MAP
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="relative">
            <div ref={mapRef} className="w-full h-[500px]"></div>

            <div className="absolute top-4 right-4 bg-black/80 border border-neon-green/30 p-3 rounded-md">
              <div className="text-xs text-neon-green/70 mb-2">THREAT LEGEND</div>
              <div className="flex items-center mb-1">
                <div className="w-3 h-3 rounded-full bg-red-500 mr-2"></div>
                <span className="text-xs">Active Attack</span>
              </div>
              <div className="flex items-center mb-1">
                <div className="w-3 h-3 rounded-full bg-yellow-500 mr-2"></div>
                <span className="text-xs">Emerging Threat</span>
              </div>
              <div className="flex items-center">
                <div className="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
                <span className="text-xs">Monitored Region</span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
            <div className="bg-black/40 border border-neon-green/20 p-4 rounded-md">
              <div className="text-xs text-neon-green/70 mb-2">ACTIVE HOTSPOTS</div>
              <div className="text-2xl font-orbitron">{attackData?.summary?.hotspots || 0}</div>
              <div className="text-xs text-neon-green/70 mt-1">
                {attackData?.summary?.hotspot_increase > 0
                  ? `↑ ${attackData?.summary?.hotspot_increase}% from last week`
                  : `↓ ${Math.abs(attackData?.summary?.hotspot_increase || 0)}% from last week`}
              </div>
            </div>

            <div className="bg-black/40 border border-neon-green/20 p-4 rounded-md">
              <div className="text-xs text-neon-green/70 mb-2">AFFECTED COUNTRIES</div>
              <div className="text-2xl font-orbitron">{attackData?.summary?.countries || 0}</div>
              <div className="text-xs text-neon-green/70 mt-1">
                Across {attackData?.summary?.continents || 0} continents
              </div>
            </div>

            <div className="bg-black/40 border border-neon-green/20 p-4 rounded-md">
              <div className="text-xs text-neon-green/70 mb-2">ATTACK VOLUME</div>
              <div className="text-2xl font-orbitron">{attackData?.summary?.total_attacks?.toLocaleString() || 0}</div>
              <div className="text-xs text-neon-green/70 mt-1">
                Last 24 hours: {attackData?.summary?.last_24h?.toLocaleString() || 0}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="bg-black/60 border-neon-green/30 shadow-glow">
          <CardHeader className="pb-2">
            <CardTitle className="text-xl font-orbitron border-b border-neon-green/30 pb-2">
              TOP TARGETED COUNTRIES
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {attackData?.top_countries?.slice(0, 5).map((country: any, index: number) => (
                <div
                  key={index}
                  className="flex justify-between items-center bg-black/40 p-3 rounded-md border border-neon-green/20"
                >
                  <div className="flex items-center">
                    <div className="w-2 h-2 bg-red-500 rounded-full mr-2 pulse-dot"></div>
                    <span>{country.name}</span>
                  </div>
                  <div className="flex items-center">
                    <span className="text-xs mr-2 text-neon-green/70">{country.percent}% of attacks</span>
                    <span className="text-md font-bold">{country.count}</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/60 border-neon-green/30 shadow-glow">
          <CardHeader className="pb-2">
            <CardTitle className="text-xl font-orbitron border-b border-neon-green/30 pb-2">
              ATTACK ORIGIN COUNTRIES
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {attackData?.origin_countries?.slice(0, 5).map((country: any, index: number) => (
                <div
                  key={index}
                  className="flex justify-between items-center bg-black/40 p-3 rounded-md border border-neon-green/20"
                >
                  <div className="flex items-center">
                    <div className="w-2 h-2 bg-yellow-500 rounded-full mr-2 pulse-dot"></div>
                    <span>{country.name}</span>
                  </div>
                  <div className="flex items-center">
                    <span className="text-xs mr-2 text-neon-green/70">{country.percent}% of origins</span>
                    <span className="text-md font-bold">{country.count}</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="bg-black/60 border-neon-green/30 shadow-glow">
        <CardHeader className="pb-2">
          <CardTitle className="text-xl font-orbitron border-b border-neon-green/30 pb-2">
            GLOBAL THREAT ALERTS
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {attackData?.alerts?.map((alert: any, index: number) => (
              <div key={index} className="bg-black/40 p-3 rounded-md border border-neon-green/20">
                <div className="flex items-center mb-1">
                  <AlertTriangle className="h-4 w-4 mr-2 text-red-500" />
                  <div className="text-sm font-bold">{alert.title}</div>
                  <div className="ml-auto text-xs text-neon-green/70">{alert.timestamp}</div>
                </div>
                <div className="text-xs">{alert.description}</div>
                <div className="flex items-center mt-2 text-xs text-neon-green/70">
                  <span>Affected regions: </span>
                  <span className="ml-1">{alert.regions.join(", ")}</span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
