"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"
import ChartWrapper from "./chart-wrapper"

interface JobPosting {
  fraudulent_probability: number
}

interface FraudProbabilityChartProps {
  jobPostings: JobPosting[]
}

export default function FraudProbabilityChart({ jobPostings }: FraudProbabilityChartProps) {
  console.log("FraudProbabilityChart - jobPostings length:", jobPostings?.length)

  if (!jobPostings || jobPostings.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Fraud Probability Distribution</CardTitle>
          <CardDescription>No data available</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-64 text-muted-foreground">No data to display</div>
        </CardContent>
      </Card>
    )
  }

  // Create histogram data
  const bins = Array.from({ length: 10 }, (_, i) => ({
    range: `${i * 10}-${(i + 1) * 10}%`,
    count: 0,
    minValue: i * 0.1,
    maxValue: (i + 1) * 0.1,
  }))

  jobPostings.forEach((job) => {
    const binIndex = Math.min(Math.floor(job.fraudulent_probability * 10), 9)
    bins[binIndex].count++
  })

  console.log("Probability chart bins:", bins)

  return (
    <Card>
      <CardHeader>
        <CardTitle>Fraud Probability Distribution</CardTitle>
        <CardDescription>
          Histogram showing the distribution of fraud probabilities across all job postings
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ChartWrapper>
          <div style={{ width: "100%", height: "320px" }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={bins}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis dataKey="range" angle={-45} textAnchor="end" height={80} fontSize={12} stroke="#666" />
                <YAxis fontSize={12} stroke="#666" />
                <Tooltip
                  formatter={(value: any) => [value, "Job Count"]}
                  labelFormatter={(label: any) => `Probability Range: ${label}`}
                  contentStyle={{
                    backgroundColor: "white",
                    border: "1px solid #ccc",
                    borderRadius: "6px",
                    color: "black",
                  }}
                />
                <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartWrapper>
      </CardContent>
    </Card>
  )
}
