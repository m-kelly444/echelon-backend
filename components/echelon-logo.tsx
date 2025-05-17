import type { SVGProps } from "react"

export function EchelonLogo(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
      <circle cx="50" cy="50" r="45" stroke="#00FF88" strokeWidth="2" />
      <circle cx="50" cy="50" r="35" stroke="#00FF88" strokeWidth="1.5" strokeDasharray="4 4" />
      <path
        d="M30 35L50 25L70 35M30 50L50 40L70 50M30 65L50 55L70 65"
        stroke="#00FF88"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <circle cx="50" cy="25" r="3" fill="#00FF88" />
      <circle cx="50" cy="40" r="3" fill="#00FF88" />
      <circle cx="50" cy="55" r="3" fill="#00FF88" />
      <circle cx="30" cy="35" r="3" fill="#00FF88" />
      <circle cx="30" cy="50" r="3" fill="#00FF88" />
      <circle cx="30" cy="65" r="3" fill="#00FF88" />
      <circle cx="70" cy="35" r="3" fill="#00FF88" />
      <circle cx="70" cy="50" r="3" fill="#00FF88" />
      <circle cx="70" cy="65" r="3" fill="#00FF88" />

      {/* Add glowing effect */}
      <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="2.5" result="blur" />
        <feComposite in="SourceGraphic" in2="blur" operator="over" />
      </filter>

      {/* Apply the filter to a duplicate of the logo */}
      <g filter="url(#glow)">
        <circle cx="50" cy="50" r="45" stroke="#00FF88" strokeWidth="2" strokeOpacity="0.5" />
        <circle cx="50" cy="50" r="35" stroke="#00FF88" strokeWidth="1.5" strokeDasharray="4 4" strokeOpacity="0.5" />
        <path
          d="M30 35L50 25L70 35M30 50L50 40L70 50M30 65L50 55L70 65"
          stroke="#00FF88"
          strokeWidth="2"
          strokeLinecap="round"
          strokeOpacity="0.5"
        />
      </g>
    </svg>
  )
}
