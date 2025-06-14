"use client"

import { useState } from "react"
import { Upload, FileText, AlertTriangle, CheckCircle, BarChart3 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription } from "@/components/ui/alert"
import FileUpload from "@/components/file-upload"
import Dashboard from "@/components/dashboard"
import ResultsTable from "@/components/results-table"
import ThemeToggle from "@/components/theme-toggle"

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

export default function FraudDetectionApp() {
  const [jobPostings, setJobPostings] = useState<JobPosting[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [uploadComplete, setUploadComplete] = useState(false)

  const handleFileProcessed = (data: JobPosting[]) => {
    setJobPostings(data)
    setUploadComplete(true)
  }

  const stats = {
    total: jobPostings.length,
    fraudulent: jobPostings.filter((job) => job.is_fraudulent).length,
    genuine: jobPostings.filter((job) => !job.is_fraudulent).length,
    highRisk: jobPostings.filter((job) => job.fraudulent_probability > 0.8).length,
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-background to-secondary/20 p-4 transition-colors duration-300">
      <div className="max-w-7xl mx-auto">
        {/* Header with Theme Toggle */}
        <div className="flex justify-between items-center mb-8">
          <div className="text-center flex-1">
            <h1 className="text-4xl font-bold text-foreground mb-2">Job Posting Fraud Detection System</h1>
            <p className="text-lg text-muted-foreground">
              Upload job posting data and get AI-powered fraud detection insights
            </p>
          </div>
          <ThemeToggle />
        </div>

        {/* Rest of the component remains the same... */}
        {/* Stats Cards */}
        {uploadComplete && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <Card className="border-border bg-card">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-card-foreground">Total Jobs</CardTitle>
                <FileText className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-card-foreground">{stats.total}</div>
              </CardContent>
            </Card>
            <Card className="border-border bg-card">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-card-foreground">Fraudulent</CardTitle>
                <AlertTriangle className="h-4 w-4 text-red-500" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-red-600 dark:text-red-400">{stats.fraudulent}</div>
                <p className="text-xs text-muted-foreground">
                  {((stats.fraudulent / stats.total) * 100).toFixed(1)}% of total
                </p>
              </CardContent>
            </Card>
            <Card className="border-border bg-card">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-card-foreground">Genuine</CardTitle>
                <CheckCircle className="h-4 w-4 text-green-500" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-green-600 dark:text-green-400">{stats.genuine}</div>
                <p className="text-xs text-muted-foreground">
                  {((stats.genuine / stats.total) * 100).toFixed(1)}% of total
                </p>
              </CardContent>
            </Card>
            <Card className="border-border bg-card">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-card-foreground">High Risk</CardTitle>
                <BarChart3 className="h-4 w-4 text-orange-500" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">{stats.highRisk}</div>
                <p className="text-xs text-muted-foreground">Probability {">"} 80%</p>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Main Content */}
        {!uploadComplete ? (
          <Card className="max-w-2xl mx-auto border-border bg-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-card-foreground">
                <Upload className="h-5 w-5" />
                Upload Job Posting Data
              </CardTitle>
              <CardDescription className="text-muted-foreground">
                Upload a CSV file containing job posting data to analyze for potential fraud
              </CardDescription>
            </CardHeader>
            <CardContent>
              <FileUpload
                onFileProcessed={handleFileProcessed}
                isProcessing={isProcessing}
                setIsProcessing={setIsProcessing}
              />

              <Alert className="mt-4 border-border bg-card">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription className="text-muted-foreground">
                  Expected CSV columns: title, location, company_profile, description, requirements, benefits,
                  employment_type, required_experience, required_education, industry, function, telecommuting,
                  has_company_logo
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>
        ) : (
          <Tabs defaultValue="dashboard" className="w-full">
            <TabsList className="grid w-full grid-cols-2 bg-muted">
              <TabsTrigger value="dashboard" className="data-[state=active]:bg-background">
                Dashboard
              </TabsTrigger>
              <TabsTrigger value="results" className="data-[state=active]:bg-background">
                Detailed Results
              </TabsTrigger>
            </TabsList>

            <TabsContent value="dashboard" className="space-y-4">
              <Dashboard jobPostings={jobPostings} />
            </TabsContent>

            <TabsContent value="results" className="space-y-4">
              <ResultsTable jobPostings={jobPostings} />
            </TabsContent>
          </Tabs>
        )}

        {uploadComplete && (
          <div className="mt-8 text-center">
            <Button
              onClick={() => {
                setJobPostings([])
                setUploadComplete(false)
              }}
              variant="outline"
              className="border-border hover:bg-accent"
            >
              Upload New File
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}
