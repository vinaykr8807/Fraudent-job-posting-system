"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts"
import ChartWrapper from "./chart-wrapper"

interface JobPosting {
  fraudulent_probability: number
  is_fraudulent: boolean
}

interface FraudHistogramChartProps {
  jobPostings: JobPosting[]
}

export default function FraudHistogramChart({ jobPostings }: FraudHistogramChartProps) {
  console.log("FraudHistogramChart - jobPostings length:", jobPostings?.length)

  if (!jobPostings || jobPostings.length === 0) {
    return (
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-card-foreground">Fraud Probability Distribution</CardTitle>
          <CardDescription className="text-muted-foreground">No data available for histogram analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-64 text-muted-foreground">
            No job postings data to display
          </div>
        </CardContent>
      </Card>
    )
  }

  // Create histogram data with 20 bins
  const bins = Array.from({ length: 20 }, (_, i) => ({
    range: `${(i * 5).toFixed(0)}-${((i + 1) * 5).toFixed(0)}%`,
    rangeStart: i * 0.05,
    rangeEnd: (i + 1) * 0.05,
    count: 0,
    genuine: 0,
    fraudulent: 0,
  }))

  // Populate bins with job data
  jobPostings.forEach((job) => {
    const binIndex = Math.min(Math.floor(job.fraudulent_probability * 20), 19)
    bins[binIndex].count++
    if (job.is_fraudulent) {
      bins[binIndex].fraudulent++
    } else {
      bins[binIndex].genuine++
    }
  })

  console.log("Histogram bins data:", bins.slice(0, 5)) // Log first 5 bins

  // Color function based on probability range
  const getBarColor = (rangeStart: number) => {
    if (rangeStart < 0.3) return "#10b981" // Green for low risk
    if (rangeStart < 0.7) return "#f59e0b" // Orange for medium risk
    return "#ef4444" // Red for high risk
  }

  // Calculate statistics
  const totalJobs = jobPostings.length
  const genuineJobs = jobPostings.filter((job) => !job.is_fraudulent).length
  const fraudulentJobs = jobPostings.filter((job) => job.is_fraudulent).length
  const meanProbability = jobPostings.reduce((sum, job) => sum + job.fraudulent_probability, 0) / totalJobs
  const sortedProbs = [...jobPostings].sort((a, b) => a.fraudulent_probability - b.fraudulent_probability)
  const medianProbability = sortedProbs[Math.floor(totalJobs / 2)]?.fraudulent_probability || 0

  return (
    <div className="space-y-6">
      {/* Fraud Probability Distribution Histogram */}
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-card-foreground">Fraud Probability Distribution</CardTitle>
          <CardDescription className="text-muted-foreground">
            Histogram showing the distribution of fraud probabilities across all job postings
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="mb-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="bg-muted/50 p-3 rounded-lg">
              <div className="font-semibold text-card-foreground">Total Jobs</div>
              <div className="text-2xl font-bold text-primary">{totalJobs.toLocaleString()}</div>
            </div>
            <div className="bg-muted/50 p-3 rounded-lg">
              <div className="font-semibold text-card-foreground">Mean Probability</div>
              <div className="text-2xl font-bold text-primary">{(meanProbability * 100).toFixed(1)}%</div>
            </div>
            <div className="bg-muted/50 p-3 rounded-lg">
              <div className="font-semibold text-card-foreground">Median Probability</div>
              <div className="text-2xl font-bold text-primary">{(medianProbability * 100).toFixed(1)}%</div>
            </div>
            <div className="bg-muted/50 p-3 rounded-lg">
              <div className="font-semibold text-card-foreground">High Risk Jobs</div>
              <div className="text-2xl font-bold text-destructive">
                {jobPostings.filter((job) => job.fraudulent_probability >= 0.7).length}
              </div>
            </div>
          </div>

          <ChartWrapper>
            <div style={{ width: "100%", height: "400px" }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={bins} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis
                    dataKey="range"
                    angle={-45}
                    textAnchor="end"
                    height={80}
                    fontSize={12}
                    interval={0}
                    stroke="#666"
                  />
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
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {bins.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={getBarColor(entry.rangeStart)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </ChartWrapper>

          {/* Risk Level Legend */}
          <div className="flex justify-center gap-6 mt-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-green-500 rounded"></div>
              <span className="text-card-foreground">Low Risk (0-30%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-orange-500 rounded"></div>
              <span className="text-card-foreground">Medium Risk (30-70%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-red-500 rounded"></div>
              <span className="text-card-foreground">High Risk (70-100%)</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Genuine vs Fraudulent Distribution */}
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-card-foreground">Job Classification Distribution</CardTitle>
          <CardDescription className="text-muted-foreground">
            Overall distribution of genuine vs fraudulent job postings
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Bar Chart */}
            <ChartWrapper>
              <div style={{ width: "100%", height: "300px" }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={[
                      { name: "Genuine Jobs", count: genuineJobs, percentage: (genuineJobs / totalJobs) * 100 },
                      {
                        name: "Fraudulent Jobs",
                        count: fraudulentJobs,
                        percentage: (fraudulentJobs / totalJobs) * 100,
                      },
                    ]}
                    margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                    <XAxis dataKey="name" fontSize={12} stroke="#666" />
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
                      <Cell fill="#10b981" />
                      <Cell fill="#ef4444" />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </ChartWrapper>

            {/* Statistics */}
            <div className="space-y-4">
              <div className="bg-green-50 dark:bg-green-950/20 p-4 rounded-lg border border-green-200 dark:border-green-800">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-semibold text-green-800 dark:text-green-200">Genuine Jobs</h3>
                    <p className="text-sm text-green-600 dark:text-green-300">Legitimate job postings</p>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                      {genuineJobs.toLocaleString()}
                    </div>
                    <div className="text-sm text-green-500 dark:text-green-300">
                      {((genuineJobs / totalJobs) * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-red-50 dark:bg-red-950/20 p-4 rounded-lg border border-red-200 dark:border-red-800">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-semibold text-red-800 dark:text-red-200">Fraudulent Jobs</h3>
                    <p className="text-sm text-red-600 dark:text-red-300">Suspicious job postings</p>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                      {fraudulentJobs.toLocaleString()}
                    </div>
                    <div className="text-sm text-red-500 dark:text-red-300">
                      {((fraudulentJobs / totalJobs) * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-blue-50 dark:bg-blue-950/20 p-4 rounded-lg border border-blue-200 dark:border-blue-800">
                <div className="text-center">
                  <h3 className="font-semibold text-blue-800 dark:text-blue-200">Total Analyzed</h3>
                  <div className="text-3xl font-bold text-blue-600 dark:text-blue-400 mt-2">
                    {totalJobs.toLocaleString()}
                  </div>
                  <p className="text-sm text-blue-600 dark:text-blue-300 mt-1">Job postings processed</p>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
