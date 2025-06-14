"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { AlertTriangle, MapPin, Building } from "lucide-react"

interface JobPosting {
  id: number
  title: string
  location: string
  company_profile: string
  fraudulent_probability: number
  is_fraudulent: boolean
}

interface TopSuspiciousJobsProps {
  jobPostings: JobPosting[]
}

export default function TopSuspiciousJobs({ jobPostings }: TopSuspiciousJobsProps) {
  const topSuspicious = jobPostings.sort((a, b) => b.fraudulent_probability - a.fraudulent_probability).slice(0, 10)

  const getRiskLevel = (probability: number) => {
    if (probability >= 0.8) return { level: "Critical", color: "destructive" as const }
    if (probability >= 0.6) return { level: "High", color: "destructive" as const }
    if (probability >= 0.4) return { level: "Medium", color: "secondary" as const }
    return { level: "Low", color: "secondary" as const }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <AlertTriangle className="h-5 w-5" />
          Top 10 Most Suspicious Job Postings
        </CardTitle>
        <CardDescription>Job postings with the highest fraud probability scores</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {topSuspicious.map((job, index) => {
            const risk = getRiskLevel(job.fraudulent_probability)
            return (
              <div key={job.id} className="p-4 border rounded-lg hover:bg-gray-50 transition-colors">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">#{index + 1}</Badge>
                    <h3 className="font-semibold text-lg">{job.title}</h3>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant={risk.color}>{risk.level}</Badge>
                    <Badge variant="outline">{(job.fraudulent_probability * 100).toFixed(1)}%</Badge>
                  </div>
                </div>

                <div className="flex items-center gap-4 text-sm text-gray-600">
                  {job.location && (
                    <div className="flex items-center gap-1">
                      <MapPin className="h-4 w-4" />
                      <span>{job.location}</span>
                    </div>
                  )}
                  {job.company_profile && (
                    <div className="flex items-center gap-1">
                      <Building className="h-4 w-4" />
                      <span className="truncate max-w-xs">
                        {job.company_profile.substring(0, 50)}
                        {job.company_profile.length > 50 ? "..." : ""}
                      </span>
                    </div>
                  )}
                </div>

                <div className="mt-2">
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${
                        job.fraudulent_probability >= 0.8
                          ? "bg-red-500"
                          : job.fraudulent_probability >= 0.6
                            ? "bg-orange-500"
                            : job.fraudulent_probability >= 0.4
                              ? "bg-yellow-500"
                              : "bg-green-500"
                      }`}
                      style={{ width: `${job.fraudulent_probability * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
