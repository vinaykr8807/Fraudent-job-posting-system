"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { TrendingUp, Eye } from "lucide-react"
import FraudProbabilityChart from "./fraud-probability-chart"
import FraudDistributionChart from "./fraud-distribution-chart"
import TopSuspiciousJobs from "./top-suspicious-jobs"
import FraudHistogramChart from "./fraud-histogram-chart"
import ProbabilityBreakdownChart from "./probability-breakdown-chart"

interface JobPosting {
  id: number
  title: string
  location: string
  company_profile: string
  description: string
  requirements: string
  benefits: string
  employment_type: string
  required_experience: string
  required_education: string
  industry: string
  function: string
  telecommuting: number
  has_company_logo: number
  fraudulent_probability: number
  is_fraudulent: boolean
}

interface DashboardProps {
  jobPostings: JobPosting[]
}

export default function Dashboard({ jobPostings }: DashboardProps) {
  const fraudulentJobs = jobPostings.filter((job) => job.is_fraudulent)
  const genuineJobs = jobPostings.filter((job) => !job.is_fraudulent)

  // Calculate risk distribution
  const riskDistribution = {
    low: jobPostings.filter((job) => job.fraudulent_probability < 0.3).length,
    medium: jobPostings.filter((job) => job.fraudulent_probability >= 0.3 && job.fraudulent_probability < 0.7).length,
    high: jobPostings.filter((job) => job.fraudulent_probability >= 0.7).length,
  }

  // Top industries with fraud
  const industryFraud = jobPostings.reduce(
    (acc, job) => {
      if (job.is_fraudulent && job.industry) {
        acc[job.industry] = (acc[job.industry] || 0) + 1
      }
      return acc
    },
    {} as Record<string, number>,
  )

  const topFraudIndustries = Object.entries(industryFraud)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5)

  return (
    <div className="space-y-6">
      {/* Risk Distribution */}
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-card-foreground">
            <TrendingUp className="h-5 w-5" />
            Risk Distribution
          </CardTitle>
          <CardDescription className="text-muted-foreground">
            Distribution of job postings by fraud risk level
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center p-4 bg-green-50 dark:bg-green-950/20 rounded-lg border border-green-200 dark:border-green-800">
              <div className="text-2xl font-bold text-green-600 dark:text-green-400">{riskDistribution.low}</div>
              <div className="text-sm text-green-700 dark:text-green-300">Low Risk ({"<"} 30%)</div>
            </div>
            <div className="text-center p-4 bg-yellow-50 dark:bg-yellow-950/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
              <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">{riskDistribution.medium}</div>
              <div className="text-sm text-yellow-700 dark:text-yellow-300">Medium Risk (30-70%)</div>
            </div>
            <div className="text-center p-4 bg-red-50 dark:bg-red-950/20 rounded-lg border border-red-200 dark:border-red-800">
              <div className="text-2xl font-bold text-red-600 dark:text-red-400">{riskDistribution.high}</div>
              <div className="text-sm text-red-700 dark:text-red-300">High Risk ({">"} 70%)</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <FraudProbabilityChart jobPostings={jobPostings} />
        <FraudDistributionChart fraudulent={fraudulentJobs.length} genuine={genuineJobs.length} />
      </div>

      {/* Histogram Charts */}
      <FraudHistogramChart jobPostings={jobPostings} />

      {/* Probability Breakdown */}
      <ProbabilityBreakdownChart jobPostings={jobPostings} />

      {/* Top Suspicious Jobs */}
      <TopSuspiciousJobs jobPostings={jobPostings} />

      {/* Industry Analysis */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Eye className="h-5 w-5" />
            Top Industries with Fraud
          </CardTitle>
          <CardDescription>Industries with the highest number of fraudulent job postings</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {topFraudIndustries.length > 0 ? (
              topFraudIndustries.map(([industry, count], index) => (
                <div key={industry} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <Badge variant="outline">{index + 1}</Badge>
                    <span className="font-medium">{industry}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="destructive">{count} fraudulent</Badge>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-gray-500 text-center py-4">No fraudulent jobs found in specific industries</p>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
