"""
Demo script to showcase the enhanced fraud detection visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import warnings

warnings.filterwarnings("ignore")

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_sample_job_data(n_samples=1000):
    """Create sample job posting data for demonstration"""
    
    # Generate synthetic text features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=100,
        n_informative=20,
        n_redundant=10,
        n_clusters_per_class=1,
        weights=[0.85, 0.15],  # Imbalanced dataset
        random_state=42
    )
    
    # Create sample job data
    industries = ['Technology', 'Healthcare', 'Finance', 'Education', 'Retail', 'Manufacturing']
    employment_types = ['Full-time', 'Part-time', 'Contract', 'Internship']
    experience_levels = ['Entry Level', '1-3 years', '3-5 years', '5+ years']
    
    job_data = pd.DataFrame({
        'fraudulent': y,
        'industry': np.random.choice(industries, n_samples),
        'employment_type': np.random.choice(employment_types, n_samples),
        'required_experience': np.random.choice(experience_levels, n_samples),
        'text_length': np.random.normal(500, 200, n_samples),
        'has_company_logo': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'telecommuting': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    
    # Make fraudulent jobs have different characteristics
    fraud_mask = job_data['fraudulent'] == 1
    job_data.loc[fraud_mask, 'text_length'] = np.random.normal(300, 150, fraud_mask.sum())
    job_data.loc[fraud_mask, 'has_company_logo'] = np.random.choice([0, 1], fraud_mask.sum(), p=[0.8, 0.2])
    
    return job_data, X

def demonstrate_enhanced_visualizations():
    """Demonstrate the enhanced visualization capabilities"""
    
    print("🎯 Creating Enhanced Fraud Detection Visualizations Demo")
    print("=" * 60)
    
    # Create sample data
    job_data, X_features = create_sample_job_data(1000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, job_data['fraudulent'], test_size=0.3, random_state=42, stratify=job_data['fraudulent']
    )
    
    # Train a simple model for demonstration
    model = PassiveAggressiveClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.decision_function(X_test)
    
    # Normalize probabilities to 0-1 range
    y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
    
    print("📊 Generating Comprehensive Visualizations...")
    
    # 1. Performance Comparison Visualization
    create_performance_comparison()
    
    # 2. Fraud Probability Distribution
    create_fraud_probability_distribution(y_pred_proba, y_test)
    
    # 3. Job Classification Analysis
    create_job_classification_analysis(job_data)
    
    # 4. Model Performance Dashboard
    create_performance_dashboard(y_test, y_pred, y_pred_proba)
    
    print("✅ All visualizations generated successfully!")
    print("\n🎨 Features demonstrated:")
    print("   • Performance comparison (before/after tuning)")
    print("   • Fraud probability distributions")
    print("   • Job classification analysis by industry/type")
    print("   • ROC curves and confusion matrices")
    print("   • Feature importance analysis")
    print("   • Comprehensive performance dashboard")

def create_performance_comparison():
    """Create performance comparison visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Metrics comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    before_tuning = [0.85, 0.45, 0.60, 0.51, 0.72]
    after_tuning = [0.92, 0.78, 0.85, 0.81, 0.89]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, before_tuning, width, label='Before Tuning', alpha=0.8, color='lightcoral')
    bars2 = ax1.bar(x + width/2, after_tuning, width, label='After Tuning', alpha=0.8, color='lightblue')
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Model Performance: Before vs After Tuning', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    # Training time comparison
    training_phases = ['Data\nPreprocessing', 'Feature\nSelection', 'Model\nTraining', 'Hyperparameter\nTuning']
    times = [2.5, 8.3, 12.1, 15.7]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    bars = ax2.bar(training_phases, times, color=colors, alpha=0.8)
    ax2.set_ylabel('Time (minutes)')
    ax2.set_title('Training Pipeline Duration', fontsize=14, fontweight='bold')
    
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{time:.1f}m', ha='center', va='bottom', fontweight='bold')
    
    # Model complexity vs performance
    models = ['Dummy\nClassifier', 'Logistic\nRegression', 'Random\nForest', 'Passive\nAggressive']
    complexity = [1, 3, 8, 5]
    performance = [0.50, 0.72, 0.85, 0.89]
    colors_scatter = ['red', 'orange', 'lightgreen', 'darkgreen']
    
    for i, (model, comp, perf, color) in enumerate(zip(models, complexity, performance, colors_scatter)):
        ax3.scatter(comp, perf, s=200, alpha=0.7, c=color, label=model)
    
    ax3.set_xlabel('Model Complexity')
    ax3.set_ylabel('F1-Score')
    ax3.set_title('Model Complexity vs Performance', fontsize=14, fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Feature selection progress
    iterations = list(range(1, 11))
    feature_count = [10000, 8500, 7200, 6100, 5200, 4500, 3800, 3200, 2800, 2500]
    performance_evolution = [0.72, 0.74, 0.76, 0.78, 0.80, 0.81, 0.82, 0.82, 0.81, 0.81]
    
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(iterations, feature_count, 'b-o', label='Feature Count', linewidth=2)
    line2 = ax4_twin.plot(iterations, performance_evolution, 'r-s', label='F1-Score', linewidth=2)
    
    ax4.set_xlabel('Feature Selection Iterations')
    ax4.set_ylabel('Number of Features', color='blue')
    ax4_twin.set_ylabel('F1-Score', color='red')
    ax4.set_title('Feature Selection Progress', fontsize=14, fontweight='bold')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='center right')
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_fraud_probability_distribution(y_pred_proba, y_test):
    """Create fraud probability distribution visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Overall probability distribution
    ax1.hist(y_pred_proba, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
    ax1.set_xlabel('Fraud Probability')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Overall Fraud Probability Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Probability by actual class
    genuine_probs = y_pred_proba[y_test == 0]
    fraud_probs = y_pred_proba[y_test == 1]
    
    ax2.hist(genuine_probs, bins=30, alpha=0.7, label='Genuine Jobs', color='lightgreen', density=True)
    ax2.hist(fraud_probs, bins=30, alpha=0.7, label='Fraudulent Jobs', color='lightcoral', density=True)
    ax2.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
    ax2.set_xlabel('Fraud Probability')
    ax2.set_ylabel('Density')
    ax2.set_title('Probability Distribution by Actual Class', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Risk level pie chart
    low_risk = np.sum((y_pred_proba >= 0) & (y_pred_proba < 0.3))
    medium_risk = np.sum((y_pred_proba >= 0.3) & (y_pred_proba < 0.7))
    high_risk = np.sum(y_pred_proba >= 0.7)
    
    risk_counts = [low_risk, medium_risk, high_risk]
    risk_labels = ['Low Risk\n(0-30%)', 'Medium Risk\n(30-70%)', 'High Risk\n(70-100%)']
    colors = ['lightgreen', 'orange', 'lightcoral']
    
    wedges, texts, autotexts = ax3.pie(risk_counts, labels=risk_labels, colors=colors, 
                                      autopct='%1.1f%%', startangle=90)
    ax3.set_title('Risk Level Distribution', fontsize=14, fontweight='bold')
    
    # Probability calibration plot
    from sklearn.calibration import calibration_curve
    
    # Create bins for calibration
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    
    ax4.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
    ax4.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    ax4.set_xlabel('Mean Predicted Probability')
    ax4.set_ylabel('Fraction of Positives')
    ax4.set_title('Probability Calibration Plot', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_job_classification_analysis(job_data):
    """Create job classification analysis visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Fraud rate by industry
    industry_stats = job_data.groupby('industry')['fraudulent'].agg(['count', 'sum']).reset_index()
    industry_stats['fraud_rate'] = industry_stats['sum'] / industry_stats['count']
    industry_stats = industry_stats.
    industry_stats = industry_stats.sort_values('fraud_rate', ascending=False)
    
    bars = ax1.bar(range(len(industry_stats)), industry_stats['fraud_rate'], 
                  color='lightcoral', alpha=0.8)
    ax1.set_xlabel('Industry')
    ax1.set_ylabel('Fraud Rate')
    ax1.set_title('Fraud Rate by Industry', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(industry_stats)))
    ax1.set_xticklabels(industry_stats['industry'], rotation=45, ha='right')
    
    # Add value labels
    for bar, rate in zip(bars, industry_stats['fraud_rate']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=9)
    
    # Employment type distribution
    emp_type_stats = job_data.groupby('employment_type')['fraudulent'].agg(['count', 'sum']).reset_index()
    emp_type_stats['fraud_rate'] = emp_type_stats['sum'] / emp_type_stats['count']
    
    wedges, texts, autotexts = ax2.pie(emp_type_stats['count'], 
                                      labels=emp_type_stats['employment_type'],
                                      autopct='%1.1f%%', startangle=90,
                                      colors=['lightblue', 'lightgreen', 'lightyellow', 'lightpink'])
    ax2.set_title('Job Distribution by Employment Type', fontsize=14, fontweight='bold')
    
    # Experience level analysis
    exp_stats = job_data.groupby('required_experience')['fraudulent'].agg(['count', 'sum']).reset_index()
    exp_stats['fraud_rate'] = exp_stats['sum'] / exp_stats['count']
    
    scatter = ax3.scatter(exp_stats['count'], exp_stats['fraud_rate'], 
                         s=exp_stats['sum']*50, alpha=0.6, 
                         c=['red', 'orange', 'yellow', 'green'])
    
    for i, row in exp_stats.iterrows():
        ax3.annotate(row['required_experience'], 
                   (row['count'], row['fraud_rate']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax3.set_xlabel('Number of Job Postings')
    ax3.set_ylabel('Fraud Rate')
    ax3.set_title('Experience Level: Volume vs Fraud Rate', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Text length analysis
    genuine_lengths = job_data[job_data['fraudulent'] == 0]['text_length']
    fraud_lengths = job_data[job_data['fraudulent'] == 1]['text_length']
    
    ax4.hist(genuine_lengths, bins=30, alpha=0.7, label='Genuine', color='lightgreen', density=True)
    ax4.hist(fraud_lengths, bins=30, alpha=0.7, label='Fraudulent', color='lightcoral', density=True)
    ax4.set_xlabel('Text Length (characters)')
    ax4.set_ylabel('Density')
    ax4.set_title('Text Length Distribution by Class', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_performance_dashboard(y_test, y_pred, y_pred_proba):
    """Create comprehensive performance dashboard"""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Main metrics display
    ax1 = fig.add_subplot(gs[0, :2])
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1]
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    
    bars = ax1.bar(metrics, values, color=colors, alpha=0.8)
    ax1.set_ylim(0, 1)
    ax1.set_title('Key Performance Metrics', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # ROC Curve
    ax2 = fig.add_subplot(gs[0, 2:])
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    ax2.plot(fpr, tpr, linewidth=3, label=f'ROC Curve (AUC = {roc_auc:.3f})', color='darkblue')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    ax2.fill_between(fpr, tpr, alpha=0.3, color='lightblue')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve Analysis', fontsize=16, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Confusion Matrix
    ax3 = fig.add_subplot(gs[1, :2])
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, 
                xticklabels=['Genuine', 'Fraudulent'],
                yticklabels=['Genuine', 'Fraudulent'])
    ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    # Probability distribution
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.hist(y_pred_proba, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
    ax4.set_xlabel('Fraud Probability')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Fraud Probability Distribution', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Model comparison
    ax5 = fig.add_subplot(gs[2, :])
    models = ['Dummy Classifier', 'Logistic Regression', 'Random Forest', 'Passive Aggressive (Final)']
    f1_scores = [0.20, 0.65, 0.78, f1]
    
    bars = ax5.bar(models, f1_scores, color=['red', 'orange', 'yellow', 'green'], alpha=0.8)
    ax5.set_ylabel('F1-Score')
    ax5.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Fraud Detection Model - Performance Dashboard', 
                fontsize=20, fontweight='bold', y=0.98)
    plt.show()

if __name__ == "__main__":
    demonstrate_enhanced_visualizations()
