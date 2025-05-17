/**
 * API client for making authenticated requests to the Echelon API
 * Uses the NEXT_PUBLIC_API_URL environment variable for the base URL
 */

// Get the API URL from environment variables
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL

// Types for request options
type RequestOptions = {
  method?: "GET" | "POST" | "PUT" | "DELETE" | "PATCH"
  headers?: Record<string, string>
  body?: any
  token?: string
}

/**
 * Makes an authenticated request to the Echelon API
 * @param endpoint The API endpoint to call
 * @param options Request options including method, headers, body, and token
 * @returns The API response data
 */
export async function apiRequest<T = any>(endpoint: string, options: RequestOptions = {}): Promise<T> {
  const { method = "GET", headers = {}, body, token } = options

  // Prepare request headers
  const requestHeaders: Record<string, string> = {
    "Content-Type": "application/json",
    ...headers,
  }

  // Add authorization header if token is provided
  if (token) {
    requestHeaders.Authorization = `Bearer ${token}`
  }

  // Prepare request options
  const requestOptions: RequestInit = {
    method,
    headers: requestHeaders,
    ...(body ? { body: JSON.stringify(body) } : {}),
  }

  try {
    // Make the request to the API
    const response = await fetch(`${API_BASE_URL}${endpoint}`, requestOptions)

    // Check if the response is OK
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.message || `API error: ${response.status}`)
    }

    // Parse and return the response data
    return await response.json()
  } catch (error) {
    console.error(`Error fetching ${endpoint}:`, error)
    throw error
  }
}

/**
 * Example API endpoints for the Echelon dashboard
 */
export const EchelonAPI = {
  // Authentication
  login: (email: string, password: string) =>
    apiRequest("/api/auth/login", {
      method: "POST",
      body: { email, password },
    }),

  register: (name: string, email: string, password: string) =>
    apiRequest("/api/auth/register", {
      method: "POST",
      body: { name, email, password },
    }),

  // Dashboard data
  getDashboardOverview: (token: string) => apiRequest("/dashboard/overview", { token }),

  getThreatActors: (token: string) => apiRequest("/dashboard/threat-actors", { token }),

  getMlInsights: (token: string) => apiRequest("/ml-insights", { token }),

  getPredictions: (token: string) => apiRequest("/predict_attacks", { token }),
}
