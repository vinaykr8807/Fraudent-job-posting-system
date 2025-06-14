"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from "recharts"
import ChartWrapper from "./chart-wrapper"

interface FraudDistributionChartProps {
  fraudulent: number
  genuine: number
}

export default function FraudDistributionChart({ fraudulent, genuine }: FraudDistributionChartProps) {
  console.log("FraudDistributionChart - fraudulent:", fraudulent, "genuine:", genuine)

  const data = [
    { name: "Genuine Jobs", value: genuine, color: "#10b981" },
    { name: "Fraudulent Jobs", value: fraudulent, color: "#ef4444" },
  ]

  const total = fraudulent + genuine

  if (total === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Job Classification Results</CardTitle>
          <CardDescription>No data available</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-64 text-muted-foreground">No data to display</div>
        </CardContent>
      </Card>
    )
  }

  console.log("Pie chart data:", data)

  return (
    <Card>
      <CardHeader>
        <CardTitle>Job Classification Results</CardTitle>
        <CardDescription>Overall distribution of genuine vs fraudulent job postings</CardDescription>
      </CardHeader>
      <CardContent>
        <ChartWrapper>
          <div style={{ width: "100%", height: "320px" }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={data}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value, percent }) => `${name}: ${value} (${(percent * 100).toFixed(1)}%)`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {data.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value: any) => [value, "Jobs"]}
                  contentStyle={{
                    backgroundColor: "white",
                    border: "1px solid #ccc",
                    borderRadius: "6px",
                    color: "black",
                  }}
                />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </ChartWrapper>
        <div className="mt-4 text-center text-sm text-muted-foreground">Total Jobs Analyzed: {total}</div>
      </CardContent>
    </Card>
  )
}
