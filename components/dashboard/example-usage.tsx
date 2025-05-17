"use client"

import { useState, useEffect } from "react"
import { EchelonAPI } from "@/lib/api-client"

export default function ExampleUsage() {
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function fetchData() {
      try {
        // Get token from localStorage (set during login)
        const token = localStorage.getItem("authToken")

        if (!token) {
          setError("Not authenticated")
          setLoading(false)
          return
        }

        // Fetch dashboard overview data using the token
        const overview = await EchelonAPI.getDashboardOverview(token)
        setData(overview)
      } catch (err) {
        console.error("Error fetching data:", err)
        setError(err instanceof Error ? err.message : "An error occurred")
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  if (loading) {
    return <div>Loading...</div>
  }

  if (error) {
    return <div>Error: {error}</div>
  }

  return (
    <div>
      <h2>Dashboard Overview</h2>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  )
}
