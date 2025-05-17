import type React from "react"
import type { Metadata } from "next"
import { Share_Tech_Mono, Orbitron } from "next/font/google"
import "./globals.css"
import { ThemeProvider } from "@/components/theme-provider"
import { AuthProvider } from "@/components/auth/auth-provider"

const shareTechMono = Share_Tech_Mono({
  weight: "400",
  subsets: ["latin"],
  variable: "--font-share-tech-mono",
})

const orbitron = Orbitron({
  subsets: ["latin"],
  variable: "--font-orbitron",
})

export const metadata: Metadata = {
  title: "ECHELON | Neural Threat Intelligence",
  description: "Advanced cybersecurity threat intelligence with ML-based attack forecasting",
    generator: 'v0.dev'
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${shareTechMono.variable} ${orbitron.variable} font-mono bg-black text-neon-green`}>
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem disableTransitionOnChange>
          <AuthProvider>
            <div className="noise"></div>
            <div className="grid-pattern"></div>
            <div className="scanlines"></div>
            {children}
          </AuthProvider>
        </ThemeProvider>
      </body>
    </html>
  )
}
