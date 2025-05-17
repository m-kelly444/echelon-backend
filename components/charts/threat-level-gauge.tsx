"use client"

import { useEffect, useRef } from "react"
import * as d3 from "d3"

interface ThreatLevelGaugeProps {
  level: string
  value: number
}

export default function ThreatLevelGauge({ level, value }: ThreatLevelGaugeProps) {
  const gaugeRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!gaugeRef.current) return

    // Clear previous chart
    d3.select(gaugeRef.current).selectAll("*").remove()

    const width = gaugeRef.current.clientWidth
    const height = 200
    const radius = Math.min(width, height) / 2

    // Create SVG
    const svg = d3
      .select(gaugeRef.current)
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .append("g")
      .attr("transform", `translate(${width / 2},${height / 2})`)

    // Add glow effect
    const defs = svg.append("defs")
    const filter = defs.append("filter").attr("id", "glow-gauge")

    filter.append("feGaussianBlur").attr("stdDeviation", "3.5").attr("result", "coloredBlur")

    const feMerge = filter.append("feMerge")
    feMerge.append("feMergeNode").attr("in", "coloredBlur")
    feMerge.append("feMergeNode").attr("in", "SourceGraphic")

    // Determine color based on threat level
    const getColor = (level: string) => {
      switch (level) {
        case "CRITICAL":
          return "#FF0000"
        case "HIGH":
          return "#FF3300"
        case "MEDIUM":
          return "#FFCC00"
        case "LOW":
          return "#00FF88"
        default:
          return "#00FF88"
      }
    }

    const color = getColor(level)

    // Create gauge background
    const arc = d3
      .arc()
      .innerRadius(radius * 0.7)
      .outerRadius(radius)
      .startAngle(-Math.PI / 2)
      .endAngle(Math.PI / 2)

    svg
      .append("path")
      .attr("d", arc as any)
      .attr("fill", "rgba(0, 0, 0, 0.3)")
      .attr("stroke", "rgba(0, 255, 136, 0.3)")
      .attr("stroke-width", 1)

    // Create gauge value arc
    const valueScale = d3
      .scaleLinear()
      .domain([0, 100])
      .range([-Math.PI / 2, Math.PI / 2])

    const valueArc = d3
      .arc()
      .innerRadius(radius * 0.7)
      .outerRadius(radius)
      .startAngle(-Math.PI / 2)
      .endAngle(valueScale(value) as number)

    svg
      .append("path")
      .attr("d", valueArc as any)
      .attr("fill", color)
      .attr("filter", "url(#glow-gauge)")

    // Add center circle
    svg
      .append("circle")
      .attr("r", radius * 0.6)
      .attr("fill", "rgba(0, 0, 0, 0.5)")
      .attr("stroke", color)
      .attr("stroke-width", 2)
      .attr("filter", "url(#glow-gauge)")

    // Add threat level text
    svg
      .append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "-0.2em")
      .attr("fill", color)
      .attr("font-size", "24px")
      .attr("font-weight", "bold")
      .attr("font-family", "var(--font-orbitron)")
      .text(level)

    // Add confidence value text
    svg
      .append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "1.5em")
      .attr("fill", "#00FF88")
      .attr("font-size", "14px")
      .attr("font-family", "monospace")
      .text(`${value.toFixed(1)}% CONFIDENCE`)

    // Add tick marks
    const ticks = [0, 25, 50, 75, 100]
    const tickArc = d3
      .arc()
      .innerRadius(radius * 1.05)
      .outerRadius(radius * 1.05)
      .startAngle((d: any) => valueScale(d) as number)
      .endAngle((d: any) => valueScale(d) as number)

    svg
      .selectAll(".tick")
      .data(ticks)
      .enter()
      .append("line")
      .attr("x1", (d: any) => {
        const angle = valueScale(d)
        return Math.cos(angle - Math.PI / 2) * radius
      })
      .attr("y1", (d: any) => {
        const angle = valueScale(d)
        return Math.sin(angle - Math.PI / 2) * radius
      })
      .attr("x2", (d: any) => {
        const angle = valueScale(d)
        return Math.cos(angle - Math.PI / 2) * (radius * 1.1)
      })
      .attr("y2", (d: any) => {
        const angle = valueScale(d)
        return Math.sin(angle - Math.PI / 2) * (radius * 1.1)
      })
      .attr("stroke", "rgba(0, 255, 136, 0.5)")
      .attr("stroke-width", 1)

    svg
      .selectAll(".tick-label")
      .data(ticks)
      .enter()
      .append("text")
      .attr("x", (d: any) => {
        const angle = valueScale(d)
        return Math.cos(angle - Math.PI / 2) * (radius * 1.2)
      })
      .attr("y", (d: any) => {
        const angle = valueScale(d)
        return Math.sin(angle - Math.PI / 2) * (radius * 1.2)
      })
      .attr("text-anchor", (d: any) => {
        const angle = valueScale(d)
        return angle < 0 ? "end" : angle > 0 ? "start" : "middle"
      })
      .attr("dy", "0.3em")
      .attr("fill", "rgba(0, 255, 136, 0.7)")
      .attr("font-size", "10px")
      .attr("font-family", "monospace")
      .text((d: any) => d)
  }, [level, value])

  return <div ref={gaugeRef} className="w-full h-48"></div>
}
