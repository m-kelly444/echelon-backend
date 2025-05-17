import { verify, sign, type JwtPayload } from "jsonwebtoken"

// Interface for our JWT payload
export interface UserJwtPayload extends JwtPayload {
  id: string
  email: string
  name?: string
}

/**
 * Verifies a JWT token using the JWT_SECRET environment variable
 * @param token The JWT token to verify
 * @returns The decoded token payload or null if invalid
 */
export function verifyToken(token: string): UserJwtPayload | null {
  try {
    // Use the JWT_SECRET environment variable to verify the token
    const decoded = verify(token, process.env.JWT_SECRET!) as UserJwtPayload
    return decoded
  } catch (error) {
    console.error("Token verification failed:", error)
    return null
  }
}

/**
 * Creates a new JWT token for a user
 * @param payload User data to encode in the token
 * @param expiresIn Token expiration time (default: 1 day)
 * @returns The signed JWT token
 */
export function createToken(payload: Omit<UserJwtPayload, "iat" | "exp">, expiresIn = "1d"): string {
  return sign(payload, process.env.JWT_SECRET!, { expiresIn })
}

/**
 * Extracts the token from the Authorization header
 * @param authHeader The Authorization header value
 * @returns The extracted token or null if not found
 */
export function extractTokenFromHeader(authHeader?: string): string | null {
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return null
  }

  return authHeader.split(" ")[1]
}
