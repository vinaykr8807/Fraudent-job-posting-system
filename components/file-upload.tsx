"use client"

import { useState, useCallback } from "react"
import { useDropzone } from "react-dropzone"
import { Upload, FileText, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import Papa from "papaparse"

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

interface FileUploadProps {
  onFileProcessed: (data: JobPosting[]) => void
  isProcessing: boolean
  setIsProcessing: (processing: boolean) => void
}

export default function FileUpload({ onFileProcessed, isProcessing, setIsProcessing }: FileUploadProps) {
  const [uploadProgress, setUploadProgress] = useState(0)

  // Simulate ML processing with fraud detection logic
  const processJobData = async (rawData: any[]): Promise<JobPosting[]> => {
    const processedData: JobPosting[] = []

    for (let i = 0; i < rawData.length; i++) {
      const job = rawData[i]

      // Simulate processing progress
      setUploadProgress((i / rawData.length) * 100)
      await new Promise((resolve) => setTimeout(resolve, 10))

      // Simple fraud detection logic based on patterns from your ML model
      const fraudScore = calculateFraudScore(job)

      processedData.push({
        id: i + 1,
        title: job.title || "",
        location: job.location || "",
        company_profile: job.company_profile || "",
        description: job.description || "",
        requirements: job.requirements || "",
        benefits: job.benefits || "",
        employment_type: job.employment_type || "",
        required_experience: job.required_experience || "",
        required_education: job.required_education || "",
        industry: job.industry || "",
        function: job.function || "",
        telecommuting: Number.parseInt(job.telecommuting) || 0,
        has_company_logo: Number.parseInt(job.has_company_logo) || 0,
        fraudulent_probability: fraudScore,
        is_fraudulent: fraudScore > 0.5,
      })
    }

    return processedData
  }

  // Simplified fraud scoring based on common patterns
  const calculateFraudScore = (job: any): number => {
    let score = 0

    // Check for suspicious patterns
    const text = `${job.title} ${job.description} ${job.requirements}`.toLowerCase()

    // High salary promises without clear requirements
    if (text.includes("high salary") || text.includes("easy money")) score += 0.3

    // Vague job descriptions
    if (!job.requirements || job.requirements.length < 50) score += 0.2

    // No company logo
    if (!job.has_company_logo || job.has_company_logo === "0") score += 0.15

    // Suspicious keywords
    const suspiciousWords = ["urgent", "immediate", "no experience", "work from home", "guaranteed"]
    suspiciousWords.forEach((word) => {
      if (text.includes(word)) score += 0.1
    })

    // Missing location
    if (!job.location || job.location.trim() === "") score += 0.2

    // Add some randomness to simulate ML model uncertainty
    score += (Math.random() - 0.5) * 0.3

    return Math.max(0, Math.min(1, score))
  }

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      const file = acceptedFiles[0]
      if (!file) return

      setIsProcessing(true)
      setUploadProgress(0)

      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: async (results) => {
          try {
            if (results.errors.length > 0) {
              console.warn("CSV parsing warnings:", results.errors)
            }

            if (!results.data || results.data.length === 0) {
              throw new Error("No data found in CSV file")
            }

            const processedData = await processJobData(results.data)
            onFileProcessed(processedData)
          } catch (error) {
            console.error("Error processing file:", error)
            alert("Error processing file. Please check the file format and try again.")
          } finally {
            setIsProcessing(false)
            setUploadProgress(0)
          }
        },
        error: (error) => {
          console.error("Error parsing CSV:", error)
          alert("Error parsing CSV file. Please check the file format.")
          setIsProcessing(false)
        },
      })
    },
    [onFileProcessed, setIsProcessing],
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "text/csv": [".csv"],
      "application/vnd.ms-excel": [".csv"],
    },
    multiple: false,
    disabled: isProcessing,
  })

  return (
    <div className="w-full">
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
          ${isDragActive ? "border-blue-400 bg-blue-50 dark:bg-blue-950/20" : "border-gray-300 hover:border-gray-400"}
          ${isProcessing ? "cursor-not-allowed opacity-50" : ""}
        `}
      >
        <input {...getInputProps()} />

        {isProcessing ? (
          <div className="space-y-4">
            <Loader2 className="h-12 w-12 mx-auto text-blue-500 animate-spin" />
            <div>
              <p className="text-lg font-medium">Processing job postings...</p>
              <p className="text-sm text-gray-500">Analyzing for fraud patterns</p>
            </div>
            <div className="max-w-xs mx-auto">
              <Progress value={uploadProgress} className="w-full" />
              <p className="text-xs text-gray-500 mt-1">{Math.round(uploadProgress)}% complete</p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <Upload className="h-12 w-12 mx-auto text-gray-400" />
            <div>
              <p className="text-lg font-medium">
                {isDragActive ? "Drop the CSV file here" : "Drag & drop a CSV file here"}
              </p>
              <p className="text-sm text-gray-500">or click to select a file</p>
            </div>
            <Button variant="outline" className="mt-4">
              <FileText className="h-4 w-4 mr-2" />
              Select CSV File
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}
