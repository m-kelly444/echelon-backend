// Base URL of the Echelon API - using environment variable with no fallback
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL

// Helper function to make authenticated API requests using NextAuth session
async function authenticatedRequest(endpoint: string, options = {}) {
  // No token fallback - must have valid session
  const headers = {
    "Content-Type": "application/json",
    ...(options as any).headers,
  }

  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers,
      credentials: "include", // Include cookies for NextAuth session
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.message || `API error: ${response.status}`)
    }

    return await response.json()
  } catch (error) {
    console.error(`Error fetching ${endpoint}:`, error)
    throw error
  }
}

// API functions that match the endpoints from the project description
export async function fetchApiRoot() {
  return authenticatedRequest("/")
}

export async function fetchApiStatus() {
  return authenticatedRequest("/status")
}

export async function fetchCVEs(limit = 50, offset = 0) {
  return authenticatedRequest(`/cves?limit=${limit}&offset=${offset}`)
}

export async function fetchGeoData() {
  return authenticatedRequest("/geo")
}

export async function fetchPredictAttacks() {
  return authenticatedRequest("/predict_attacks")
}

export async function fetchDashboardOverview() {
  return authenticatedRequest("/dashboard/overview")
}

export async function fetchDashboardAttackTechniques() {
  return authenticatedRequest("/dashboard/attack-techniques")
}

export async function fetchDashboardTemporal() {
  return authenticatedRequest("/dashboard/temporal")
}

export async function fetchDashboardGeographic() {
  return authenticatedRequest("/dashboard/geographic")
}

export async function fetchDashboardTargets() {
  return authenticatedRequest("/dashboard/targets")
}

export async function fetchDashboardThreatActors() {
  return authenticatedRequest("/dashboard/threat-actors")
}

export async function fetchMlInsights() {
  return authenticatedRequest("/ml-insights")
}

export async function fetchMlInsightsTrends() {
  return authenticatedRequest("/ml-insights/trends")
}

export async function fetchMlInsightsAnomalies() {
  return authenticatedRequest("/ml-insights/anomalies")
}

export async function fetchMlInsightsPredictions() {
  return authenticatedRequest("/ml-insights/predictions")
}
