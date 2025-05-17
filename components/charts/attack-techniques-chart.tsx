"use client"

import { useEffect, useRef } from "react"
import * as d3 from "d3"

interface AttackTechniquesChartProps {
  data: any[]
}

export default function AttackTechniquesChart({ data }: AttackTechniquesChartProps) {
  const chartRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!chartRef.current || !data || data.length === 0) return

    // Clear previous chart
    d3.select(chartRef.current).selectAll("*").remove()

    const margin = { top: 20, right: 30, bottom: 20, left: 120 }
    const width = chartRef.current.clientWidth - margin.left - margin.right
    const height = 300 - margin.top - margin.bottom

    // Sort data by count
    const sortedData = [...data].sort((a, b) => b.count - a.count)

    // Create SVG
    const svg = d3
      .select(chartRef.current)
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`)

    // X scale
    const x = d3
      .scaleLinear()
      .domain([0, (d3.max(sortedData, (d) => d.count) as number) * 1.1])
      .range([0, width])

    // Y scale
    const y = d3
      .scaleBand()
      .domain(sortedData.map((d) => d.technique))
      .range([0, height])
      .padding(0.3)

    // Add glow effect
    const defs = svg.append("defs")
    const filter = defs.append("filter").attr("id", "glow-bar")

    filter.append("feGaussianBlur").attr("stdDeviation", "2.5").attr("result", "coloredBlur")

    const feMerge = filter.append("feMerge")
    feMerge.append("feMergeNode").attr("in", "coloredBlur")
    feMerge.append("feMergeNode").attr("in", "SourceGraphic")

    // Add grid lines
    svg
      .append("g")
      .attr("class", "grid")
      .call(
        d3
          .axisBottom(x)
          .tickSize(height)
          .tickFormat(() => ""),
      )
      .attr("color", "rgba(0, 255, 136, 0.1)")
      .selectAll("line")
      .attr("stroke", "rgba(0, 255, 136, 0.1)")

    // Add bars
    svg
      .selectAll(".bar")
      .data(sortedData)
      .enter()
      .append("rect")
      .attr("class", "bar")
      .attr("y", (d) => y(d.technique) as number)
      .attr("height", y.bandwidth())
      .attr("x", 0)
      .attr("width", 0) // Start with width 0 for animation
      .attr("fill", "rgba(0, 255, 136, 0.7)")
      .attr("rx", 2)
      .attr("ry", 2)
      .attr("stroke", "#00FF88")
      .attr("stroke-width", 1)
      .attr("filter", "url(#glow-bar)")
      .transition()
      .duration(1000)
      .attr("width", (d) => x(d.count))

    // Add count labels
    svg
      .selectAll(".count")
      .data(sortedData)
      .enter()
      .append("text")
      .attr("class", "count")
      .attr("y", (d) => (y(d.technique) as number) + y.bandwidth() / 2)
      .attr("x", (d) => x(d.count) + 5)
      .attr("dy", "0.35em")
      .attr("fill", "#00FF88")
      .attr("font-family", "monospace")
      .attr("font-size", "12px")
      .text((d) => d.count)
      .style("opacity", 0) // Start with opacity 0 for animation
      .transition()
      .duration(1000)
      .style("opacity", 1)

    // Add Y axis
    svg
      .append("g")
      .call(d3.axisLeft(y))
      .selectAll("text")
      .attr("fill", "#00FF88")
      .attr("font-family", "monospace")
      .attr("font-size", "10px")

    // Remove axis lines
    svg.selectAll(".domain").attr("stroke", "rgba(0, 255, 136, 0.3)")

    svg.selectAll(".tick line").attr("stroke", "rgba(0, 255, 136, 0.3)")
  }, [data])

  return <div ref={chartRef} className="w-full h-72"></div>
}
