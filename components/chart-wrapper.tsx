"use client"

import type React from "react"

import { useEffect, useState } from "react"

interface ChartWrapperProps {
  children: React.ReactNode
  fallback?: React.ReactNode
}

export default function ChartWrapper({ children, fallback }: ChartWrapperProps) {
  const [isClient, setIsClient] = useState(false)

  useEffect(() => {
    setIsClient(true)
  }, [])

  if (!isClient) {
    return (
      <div className="flex items-center justify-center h-64 bg-muted/20 rounded-lg">
        {fallback || (
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-2"></div>
            <p className="text-sm text-muted-foreground">Loading chart...</p>
          </div>
        )}
      </div>
    )
  }

  return <>{children}</>
}
