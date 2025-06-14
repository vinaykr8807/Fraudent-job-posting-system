"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Search, Download, Filter } from "lucide-react"

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

interface ResultsTableProps {
  jobPostings: JobPosting[]
}

export default function ResultsTable({ jobPostings }: ResultsTableProps) {
  const [searchTerm, setSearchTerm] = useState("")
  const [filterType, setFilterType] = useState("all")
  const [currentPage, setCurrentPage] = useState(1)
  const itemsPerPage = 20

  // Filter and search logic
  const filteredJobs = jobPostings.filter((job) => {
    const matchesSearch =
      job.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      job.location.toLowerCase().includes(searchTerm.toLowerCase()) ||
      job.industry.toLowerCase().includes(searchTerm.toLowerCase())

    const matchesFilter =
      filterType === "all" ||
      (filterType === "fraudulent" && job.is_fraudulent) ||
      (filterType === "genuine" && !job.is_fraudulent) ||
      (filterType === "high-risk" && job.fraudulent_probability > 0.7)

    return matchesSearch && matchesFilter
  })

  // Pagination
  const totalPages = Math.ceil(filteredJobs.length / itemsPerPage)
  const startIndex = (currentPage - 1) * itemsPerPage
  const paginatedJobs = filteredJobs.slice(startIndex, startIndex + itemsPerPage)

  const getRiskBadge = (probability: number, isFraudulent: boolean) => {
    if (isFraudulent) {
      return <Badge variant="destructive">Fraudulent</Badge>
    }
    if (probability > 0.7) {
      return <Badge variant="destructive">High Risk</Badge>
    }
    if (probability > 0.4) {
      return <Badge variant="secondary">Medium Risk</Badge>
    }
    return <Badge variant="outline">Low Risk</Badge>
  }

  const exportToCSV = () => {
    const headers = ["ID", "Title", "Location", "Industry", "Fraud Probability", "Classification"]
    const csvData = filteredJobs.map((job) => [
      job.id,
      `"${job.title}"`,
      `"${job.location}"`,
      `"${job.industry}"`,
      job.fraudulent_probability.toFixed(3),
      job.is_fraudulent ? "Fraudulent" : "Genuine",
    ])

    const csvContent = [headers, ...csvData].map((row) => row.join(",")).join("\n")
    const blob = new Blob([csvContent], { type: "text/csv" })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "fraud_detection_results.csv"
    a.click()
    window.URL.revokeObjectURL(url)
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Detailed Results</CardTitle>
        <CardDescription>Complete analysis results for all job postings with fraud detection scores</CardDescription>
      </CardHeader>
      <CardContent>
        {/* Filters and Search */}
        <div className="flex flex-col sm:flex-row gap-4 mb-6">
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <Input
                placeholder="Search by title, location, or industry..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
          </div>
          <Select value={filterType} onValueChange={setFilterType}>
            <SelectTrigger className="w-48">
              <Filter className="h-4 w-4 mr-2" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Jobs</SelectItem>
              <SelectItem value="fraudulent">Fraudulent Only</SelectItem>
              <SelectItem value="genuine">Genuine Only</SelectItem>
              <SelectItem value="high-risk">High Risk Only</SelectItem>
            </SelectContent>
          </Select>
          <Button onClick={exportToCSV} variant="outline">
            <Download className="h-4 w-4 mr-2" />
            Export CSV
          </Button>
        </div>

        {/* Results Summary */}
        <div className="mb-4 text-sm text-gray-600">
          Showing {paginatedJobs.length} of {filteredJobs.length} results
        </div>

        {/* Table */}
        <div className="border rounded-lg overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>ID</TableHead>
                <TableHead>Title</TableHead>
                <TableHead>Location</TableHead>
                <TableHead>Industry</TableHead>
                <TableHead>Employment Type</TableHead>
                <TableHead>Fraud Probability</TableHead>
                <TableHead>Classification</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {paginatedJobs.map((job) => (
                <TableRow key={job.id}>
                  <TableCell className="font-mono">{job.id}</TableCell>
                  <TableCell className="font-medium max-w-xs truncate">{job.title}</TableCell>
                  <TableCell>{job.location || "N/A"}</TableCell>
                  <TableCell>{job.industry || "N/A"}</TableCell>
                  <TableCell>{job.employment_type || "N/A"}</TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <span className="font-mono">{(job.fraudulent_probability * 100).toFixed(1)}%</span>
                      <div className="w-16 bg-gray-200 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full ${
                            job.fraudulent_probability >= 0.7
                              ? "bg-red-500"
                              : job.fraudulent_probability >= 0.4
                                ? "bg-yellow-500"
                                : "bg-green-500"
                          }`}
                          style={{ width: `${job.fraudulent_probability * 100}%` }}
                        />
                      </div>
                    </div>
                  </TableCell>
                  <TableCell>{getRiskBadge(job.fraudulent_probability, job.is_fraudulent)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between mt-6">
            <div className="text-sm text-gray-600">
              Page {currentPage} of {totalPages}
            </div>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage((prev) => Math.max(1, prev - 1))}
                disabled={currentPage === 1}
              >
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage((prev) => Math.min(totalPages, prev + 1))}
                disabled={currentPage === totalPages}
              >
                Next
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
