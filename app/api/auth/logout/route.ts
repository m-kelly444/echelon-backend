import { NextResponse } from "next/server"

export async function POST() {
  // Clear the auth token cookie
  const response = NextResponse.json({ message: "Logged out successfully" })
  response.cookies.delete("authToken")

  return response
}
