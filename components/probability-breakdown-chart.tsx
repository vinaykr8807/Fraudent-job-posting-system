"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie,
  Legend,
} from "recharts"
import ChartWrapper from "./chart-wrapper"

interface JobPosting {
  fraudulent_probability: number
  is_fraudulent: boolean
}

interface ProbabilityBreakdownChartProps {
  jobPostings: JobPosting[]
}

export default function ProbabilityBreakdownChart({ jobPostings }: ProbabilityBreakdownChartProps) {
  console.log("ProbabilityBreakdownChart - jobPostings length:", jobPostings?.length)

  if (!jobPostings || jobPostings.length === 0) {
    return (
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-card-foreground">Risk Level Breakdown</CardTitle>
          <CardDescription className="text-muted-foreground">No data available for risk analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-64 text-muted-foreground">
            No job postings data to display
          </div>
        </CardContent>
      </Card>
    )
  }

  // Calculate risk level distributions
  const lowRisk = jobPostings.filter((job) => job.fraudulent_probability < 0.3).length
  const mediumRisk = jobPostings.filter(
    (job) => job.fraudulent_probability >= 0.3 && job.fraudulent_probability < 0.7,
  ).length
  const highRisk = jobPostings.filter((job) => job.fraudulent_probability >= 0.7).length
  const total = jobPostings.length

  // Data for risk level bar chart
  const riskData = [
    {
      level: "Low Risk",
      range: "0-30%",
      count: lowRisk,
      percentage: (lowRisk / total) * 100,
      color: "#10b981",
    },
    {
      level: "Medium Risk",
      range: "30-70%",
      count: mediumRisk,
      percentage: (mediumRisk / total) * 100,
      color: "#f59e0b",
    },
    {
      level: "High Risk",
      range: "70-100%",
      count: highRisk,
      percentage: (highRisk / total) * 100,
      color: "#ef4444",
    },
  ]

  // Data for pie chart
  const pieData = [
    { name: "Genuine Jobs", value: jobPostings.filter((job) => !job.is_fraudulent).length, color: "#10b981" },
    { name: "Fraudulent Jobs", value: jobPostings.filter((job) => job.is_fraudulent).length, color: "#ef4444" },
  ]

  // Calculate accuracy metrics for high-risk predictions
  const highRiskJobs = jobPostings.filter((job) => job.fraudulent_probability >= 0.7)
  const highRiskActualFraud = highRiskJobs.filter((job) => job.is_fraudulent).length
  const highRiskPrecision = highRiskJobs.length > 0 ? (highRiskActualFraud / highRiskJobs.length) * 100 : 0

  console.log("Risk data:", riskData)
  console.log("Pie data:", pieData)

  return (
    <div className="space-y-6">
      {/* Risk Level Breakdown */}
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-card-foreground">Risk Level Breakdown</CardTitle>
          <CardDescription className="text-muted-foreground">
            Distribution of job postings by fraud risk categories
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Bar Chart */}
            <ChartWrapper>
              <div style={{ width: "100%", height: "320px" }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={riskData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                    <XAxis dataKey="level" fontSize={12} stroke="#666" />
                    <YAxis fontSize={12} stroke="#666" />
                    <Tooltip
                      formatter={(value: any, name: any) => [
                        name === "count" ? value.toLocaleString() : `${value.toFixed(1)}%`,
                        name === "count" ? "Jobs" : "Percentage",
                      ]}
                      contentStyle={{
                        backgroundColor: "white",
                        border: "1px solid #ccc",
                        borderRadius: "6px",
                        color: "black",
                      }}
                    />
                    <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                      {riskData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </ChartWrapper>

            {/* Risk Level Statistics */}
            <div className="space-y-4">
              {riskData.map((risk, index) => (
                <div
                  key={index}
                  className="p-4 rounded-lg border"
                  style={{
                    backgroundColor: `${risk.color}10`,
                    borderColor: `${risk.color}40`,
                  }}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-semibold" style={{ color: risk.color }}>
                        {risk.level}
                      </h3>
                      <p className="text-sm text-muted-foreground">Probability: {risk.range}</p>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold" style={{ color: risk.color }}>
                        {risk.count.toLocaleString()}
                      </div>
                      <div className="text-sm text-muted-foreground">{risk.percentage.toFixed(1)}%</div>
                    </div>
                  </div>
                </div>
              ))}

              {/* High Risk Precision */}
              <div className="bg-yellow-50 dark:bg-yellow-950/20 p-4 rounded-lg border border-yellow-200 dark:border-yellow-800">
                <h3 className="font-semibold text-yellow-800 dark:text-yellow-200">High Risk Precision</h3>
                <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400 mt-1">
                  {highRiskPrecision.toFixed(1)}%
                </div>
                <p className="text-sm text-yellow-600 dark:text-yellow-300">
                  {highRiskActualFraud}/{highRiskJobs.length} high-risk predictions were correct
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Genuine vs Fraudulent Pie Chart */}
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-card-foreground">Classification Overview</CardTitle>
          <CardDescription className="text-muted-foreground">
            Visual breakdown of genuine vs fraudulent job classifications
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Pie Chart */}
            <ChartWrapper>
              <div style={{ width: "100%", height: "320px" }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, value, percent }) =>
                        `${name}: ${value.toLocaleString()} (${(percent * 100).toFixed(1)}%)`
                      }
                      outerRadius={100}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {pieData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip
                      formatter={(value: any) => [value.toLocaleString(), "Jobs"]}
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

            {/* Summary Statistics */}
            <div className="space-y-4">
              <div className="text-center p-6 bg-muted/50 rounded-lg">
                <h3 className="text-lg font-semibold text-card-foreground mb-2">Analysis Summary</h3>
                <div className="text-3xl font-bold text-primary mb-2">{total.toLocaleString()}</div>
                <p className="text-muted-foreground">Total jobs analyzed</p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-4 bg-green-50 dark:bg-green-950/20 rounded-lg border border-green-200 dark:border-green-800">
                  <div className="text-xl font-bold text-green-600 dark:text-green-400">
                    {((pieData[0].value / total) * 100).toFixed(1)}%
                  </div>
                  <p className="text-sm text-green-600 dark:text-green-300">Genuine</p>
                </div>
                <div className="text-center p-4 bg-red-50 dark:bg-red-950/20 rounded-lg border border-red-200 dark:border-red-800">
                  <div className="text-xl font-bold text-red-600 dark:text-red-400">
                    {((pieData[1].value / total) * 100).toFixed(1)}%
                  </div>
                  <p className="text-sm text-red-600 dark:text-red-300">Fraudulent</p>
                </div>
              </div>

              <div className="p-4 bg-blue-50 dark:bg-blue-950/20 rounded-lg border border-blue-200 dark:border-blue-800">
                <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">Detection Effectiveness</h4>
                <div className="text-sm text-blue-600 dark:text-blue-300 space-y-1">
                  <div>• {lowRisk.toLocaleString()} jobs flagged as low risk</div>
                  <div>• {mediumRisk.toLocaleString()} jobs require manual review</div>
                  <div>• {highRisk.toLocaleString()} jobs flagged as high risk</div>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
