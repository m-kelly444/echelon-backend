interface LoadingSpinnerProps {
  message?: string
}

export default function LoadingSpinner({ message = "LOADING DATA" }: LoadingSpinnerProps) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-terminal-black">
      <div className="terminal-container p-8 text-center border border-terminal-green border-opacity-30 bg-black bg-opacity-70 rounded shadow-glow">
        <div className="text-xl mb-4 text-terminal-green">{message}</div>
        <div className="flex justify-center">
          <div className="w-16 h-16 relative">
            <div className="w-full h-full border-4 border-terminal-green border-opacity-30 rounded-full animate-spin"></div>
            <div className="absolute inset-2 border-4 border-terminal-green border-opacity-50 rounded-full animate-spin-slow"></div>
            <div className="absolute inset-4 border-4 border-terminal-green border-opacity-70 rounded-full animate-ping"></div>
          </div>
        </div>
      </div>
    </div>
  )
}
