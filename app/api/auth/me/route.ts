import { NextResponse } from "next/server"
import { verify } from "jsonwebtoken"
import { db } from "@/lib/db"

export async function GET(req: Request) {
  try {
    // Get token from Authorization header
    const authHeader = req.headers.get("Authorization")

    if (!authHeader || !authHeader.startsWith("Bearer ")) {
      return NextResponse.json({ message: "Unauthorized" }, { status: 401 })
    }

    const token = authHeader.split(" ")[1]

    // Verify token
    const decoded = verify(token, process.env.JWT_SECRET || "fallback-secret-do-not-use-in-production") as {
      id: string
    }

    // Get user from database
    const user = await db.user.findUnique({
      where: { id: decoded.id },
    })

    if (!user) {
      return NextResponse.json({ message: "User not found" }, { status: 404 })
    }

    // Return user without password
    const { password, ...userWithoutPassword } = user

    return NextResponse.json({ user: userWithoutPassword })
  } catch (error) {
    console.error("Authentication error:", error)
    return NextResponse.json({ message: "Unauthorized" }, { status: 401 })
  }
}
