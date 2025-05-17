"use client"

import { useEffect, useRef } from "react"
import * as d3 from "d3"

interface TemporalForecastChartProps {
  data: any[]
}

export default function TemporalForecastChart({ data }: TemporalForecastChartProps) {
  const chartRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!chartRef.current || !data || data.length === 0) return

    // Clear previous chart
    d3.select(chartRef.current).selectAll("*").remove()

    const margin = { top: 20, right: 30, bottom: 40, left: 50 }
    const width = chartRef.current.clientWidth - margin.left - margin.right
    const height = 250 - margin.top - margin.bottom

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
      .scaleBand()
      .domain(data.map((d) => d.date))
      .range([0, width])
      .padding(0.1)

    // Y scale
    const y = d3
      .scaleLinear()
      .domain([0, (d3.max(data, (d) => d.expected_activity) as number) * 1.1])
      .range([height, 0])

    // Color scale based on confidence
    const confidenceColor = d3
      .scaleLinear<string>()
      .domain([0.4, 1])
      .range(["rgba(255, 51, 51, 0.7)", "rgba(0, 255, 136, 0.8)"])

    // Add glow effect
    const defs = svg.append("defs")
    const filter = defs.append("filter").attr("id", "glow")

    filter.append("feGaussianBlur").attr("stdDeviation", "2.5").attr("result", "coloredBlur")

    const feMerge = filter.append("feMerge")
    feMerge.append("feMergeNode").attr("in", "coloredBlur")
    feMerge.append("feMergeNode").attr("in", "SourceGraphic")

    // Add grid lines
    svg
      .append("g")
      .attr("class", "grid")
      .attr("transform", `translate(0,${height})`)
      .call(
        d3
          .axisBottom(x)
          .tickSize(-height)
          .tickFormat(() => ""),
      )
      .selectAll("line")
      .attr("stroke", "rgba(0, 255, 136, 0.1)")

    svg
      .append("g")
      .attr("class", "grid")
      .call(
        d3
          .axisLeft(y)
          .tickSize(-width)
          .tickFormat(() => ""),
      )
      .selectAll("line")
      .attr("stroke", "rgba(0, 255, 136, 0.1)")

    // Add area
    const area = d3
      .area<any>()
      .x((d) => (x(d.date) as number) + x.bandwidth() / 2)
      .y0(height)
      .y1((d) => y(d.expected_activity))
      .curve(d3.curveMonotoneX)

    svg.append("path").datum(data).attr("fill", "rgba(0, 255, 136, 0.1)").attr("d", area)

    // Add line
    const line = d3
      .line<any>()
      .x((d) => (x(d.date) as number) + x.bandwidth() / 2)
      .y((d) => y(d.expected_activity))
      .curve(d3.curveMonotoneX)

    svg
      .append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", "#00FF88")
      .attr("stroke-width", 2)
      .attr("d", line)
      .attr("filter", "url(#glow)")

    // Add dots
    svg
      .selectAll(".dot")
      .data(data)
      .enter()
      .append("circle")
      .attr("class", "dot")
      .attr("cx", (d) => (x(d.date) as number) + x.bandwidth() / 2)
      .attr("cy", (d) => y(d.expected_activity))
      .attr("r", 4)
      .attr("fill", (d) => confidenceColor(d.confidence))
      .attr("stroke", "#00FF88")
      .attr("stroke-width", 1)
      .attr("filter", "url(#glow)")

    // Add X axis with selected dates (to avoid crowding)
    const filteredDates = data.filter((_, i) => i % 5 === 0).map((d) => d.date)
    svg
      .append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x).tickValues(filteredDates))
      .selectAll("text")
      .attr("transform", "rotate(-45)")
      .style("text-anchor", "end")
      .attr("dx", "-.8em")
      .attr("dy", ".15em")
      .attr("fill", "#00FF88")
      .attr("font-family", "monospace")
      .attr("font-size", "10px")

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

  return <div ref={chartRef} className="w-full h-64"></div>
}
