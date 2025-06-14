# Job Posting Fraud Detection System

A comprehensive machine learning system that detects fraudulent job postings using natural language processing and serves insights via an interactive dashboard.

## Features

### 🔍 **Fraud Detection Engine**
- Binary classifier trained on job posting data
- Text preprocessing with NLTK (stopword removal, lemmatization)
- TF-IDF vectorization with n-gram features
- Multiple ML algorithms tested (Passive Aggressive, Random Forest, SVM, XGBoost)
- ADASYN oversampling for handling imbalanced data
- Feature selection for optimal performance

### 📊 **Interactive Dashboard**
- **File Upload**: Drag & drop CSV files with job posting data
- **Real-time Processing**: Live progress tracking during analysis
- **Visual Insights**: 
  - Fraud probability distribution histogram
  - Pie chart showing genuine vs fraudulent breakdown
  - Risk level distribution (Low/Medium/High)
  - Top 10 most suspicious job postings
- **Detailed Results Table**: Searchable, filterable, and exportable results
- **Industry Analysis**: Industries with highest fraud rates

### 📈 **Key Metrics & Evaluation**
- **F1-Score optimization** for imbalanced dataset handling
- Confusion matrix visualization
- Classification reports with precision/recall
- Feature importance analysis
- ROC-AUC scoring

## Installation & Setup

### Prerequisites
- Node.js 18+ 
- Python 3.8+ (for model training)

### Quick Start

1. **Clone and Install**
   \`\`\`bash
   git clone <repository-url>
   cd fraud-detection-system
   npm install
   \`\`\`

2. **Start Development Server**
   \`\`\`bash
   npm run dev
   \`\`\`

3. **Open Application**
   Navigate to \`http://localhost:3000\`

### Model Training (Optional)

If you want to retrain the model with your own data:

1. **Install Python Dependencies**
   \`\`\`bash
   pip install pandas numpy scikit-learn nltk xgboost imbalanced-learn matplotlib seaborn wordcloud
   \`\`\`

2. **Run Training Script**
   \`\`\`bash
   python scripts/fraud_detection_model.py
   \`\`\`

## Usage

### 1. **Upload Data**
- Prepare a CSV file with job posting data
- Required columns: \`title\`, \`description\`, \`location\`, \`company_profile\`, \`requirements\`, \`benefits\`, \`employment_type\`, \`required_experience\`, \`required_education\`, \`industry\`, \`function\`, \`telecommuting\`, \`has_company_logo\`
- Drag and drop the file or click to select

### 2. **View Results**
- **Dashboard Tab**: Visual overview with charts and statistics
- **Detailed Results Tab**: Complete table with all predictions
- Export results as CSV for further analysis

### 3. **Interpret Results**
- **Fraud Probability**: 0-100% likelihood of being fraudulent
- **Risk Levels**: 
  - Low Risk: < 30%
  - Medium Risk: 30-70%
  - High Risk: > 70%
- **Classification**: Binary prediction (Genuine/Fraudulent)

## Technical Architecture

### Frontend (Next.js)
- **React Components**: Modular UI with shadcn/ui
- **File Processing**: CSV parsing with Papa Parse
- **Visualizations**: Recharts for interactive charts
- **State Management**: React hooks for data flow

### ML Pipeline (Python)
- **Text Preprocessing**: NLTK for NLP tasks
- **Feature Engineering**: TF-IDF vectorization
- **Model Selection**: Comparative analysis of multiple algorithms
- **Class Imbalance**: ADASYN oversampling
- **Feature Selection**: LinearSVC-based selection

### Key Algorithms Tested
1. **Passive Aggressive Classifier** ⭐ (Best performer)
2. Random Forest Classifier
3. Linear Support Vector Classifier
4. XGBoost Classifier

## Model Performance

Based on the training pipeline, the system achieves:
- **High F1-Score** on imbalanced fraud detection data
- **Robust feature selection** identifying key fraud indicators
- **Real-time prediction** capabilities
- **Interpretable results** with probability scores and feature importance

## Data Requirements

### Expected CSV Format
\`\`\`csv
title,location,company_profile,description,requirements,benefits,employment_type,required_experience,required_education,industry,function,telecommuting,has_company_logo
"Software Engineer","New York, NY","Tech company focused on innovation","We are looking for a skilled software engineer...","Bachelor's degree in CS","Health insurance, 401k","Full-time","3-5 years","Bachelor's","Technology","Engineering",0,1
\`\`\`

### Column Descriptions
- **title**: Job position title
- **location**: Job location (city, state/country)
- **company_profile**: Description of the hiring company
- **description**: Detailed job description
- **requirements**: Required qualifications and skills
- **benefits**: Offered benefits and perks
- **employment_type**: Full-time, Part-time, Contract, etc.
- **required_experience**: Experience level required
- **required_education**: Education requirements
- **industry**: Industry sector
- **function**: Job function/department
- **telecommuting**: 1 if remote work allowed, 0 otherwise
- **has_company_logo**: 1 if company has logo, 0 otherwise

## Fraud Detection Indicators

The model identifies several key patterns associated with fraudulent job postings:

### 🚩 **High-Risk Indicators**
- Vague or missing job requirements
- Unrealistic salary promises
- Poor grammar and spelling
- Missing company information
- Urgent hiring language
- Work-from-home emphasis without clear role definition
- No company logo or branding

### ✅ **Genuine Job Indicators**
- Detailed job descriptions
- Specific skill requirements
- Clear company information
- Professional language
- Realistic compensation
- Proper contact information

## API Integration (Future Enhancement)

The system is designed to support API endpoints for real-time fraud detection:

\`\`\`javascript
// Example API usage
const response = await fetch('/api/detect-fraud', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    title: "Software Engineer",
    description: "Job description...",
    // ... other fields
  })
});

const result = await response.json();
// { is_fraudulent: false, fraud_probability: 0.23 }
\`\`\`

## Deployment

### Production Deployment
\`\`\`bash
npm run build
npm start
\`\`\`

### Docker Deployment
\`\`\`dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
\`\`\`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation
- Review the model training logs for debugging

---

**Built with ❤️ using Next.js, React, and Machine Learning**
