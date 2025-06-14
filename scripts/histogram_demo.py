"""
Demo script specifically for histogram visualizations in fraud detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_sample_fraud_data(n_samples=2000):
    """Create realistic sample fraud detection data"""
    
    # Generate synthetic features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=50,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        weights=[0.88, 0.12],  # Realistic fraud rate ~12%
        random_state=42
    )
    
    return X, y

def demonstrate_histogram_analysis():
    """Demonstrate comprehensive histogram analysis for fraud detection"""
    
    print("📊 FRAUD DETECTION HISTOGRAM ANALYSIS DEMO")
    print("=" * 60)
    
    # Create sample data
    X, y = create_sample_fraud_data(2000)
    
    # Split and train model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train model
    model = PassiveAggressiveClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.decision_function(X_test)
    
    # Normalize probabilities to 0-1 range
    y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
    
    print(f"📈 Generated {len(y_test)} test predictions")
    print(f"🎯 Actual fraud rate: {(y_test.sum() / len(y_test)) * 100:.1f}%")
    
    # Create comprehensive histogram visualizations
    create_fraud_probability_histogram(y_pred_proba, y_test)
    create_classification_distribution(y_test)
    create_risk_level_analysis(y_pred_proba, y_test)
    create_detailed_probability_analysis(y_pred_proba, y_test)
    
    print("✅ Histogram analysis completed!")
    print("\n🎨 Visualizations created:")
    print("   • Fraud probability distribution histogram")
    print("   • Genuine vs fraudulent job distribution")
    print("   • Risk level breakdown analysis")
    print("   • Detailed probability statistics")

def create_fraud_probability_histogram(y_pred_proba, y_test):
    """Create detailed fraud probability histogram"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1. Main probability distribution histogram
    n_bins = 30
    counts, bins, patches = ax1.hist(y_pred_proba, bins=n_bins, alpha=0.8, 
                                    edgecolor='black', linewidth=0.5)
    
    # Color bars based on risk levels
    for i, (patch, bin_left, bin_right) in enumerate(zip(patches, bins[:-1], bins[1:])):
        bin_center = (bin_left + bin_right) / 2
        if bin_center < 0.3:
            patch.set_facecolor('#10b981')  # Green
        elif bin_center < 0.7:
            patch.set_facecolor('#f59e0b')  # Orange
        else:
            patch.set_facecolor('#ef4444')  # Red
    
    # Add threshold lines
    ax1.axvline(0.3, color='green', linestyle='--', linewidth=2, alpha=0.8, label='Low Risk Threshold')
    ax1.axvline(0.5, color='red', linestyle='-', linewidth=3, alpha=0.9, label='Decision Threshold')
    ax1.axvline(0.7, color='darkred', linestyle='--', linewidth=2, alpha=0.8, label='High Risk Threshold')
    
    ax1.set_xlabel('Fraud Probability', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Job Postings', fontsize=14, fontweight='bold')
    ax1.set_title('Distribution of Fraud Probabilities Across All Job Postings', 
                  fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    mean_prob = np.mean(y_pred_proba)
    median_prob = np.median(y_pred_proba)
    std_prob = np.std(y_pred_proba)
    
    stats_text = f'Statistics:\nMean: {mean_prob:.3f}\nMedian: {median_prob:.3f}\nStd Dev: {std_prob:.3f}'
    ax1.text(0.75, 0.85, stats_text, transform=ax1.transAxes, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
             fontsize=11, verticalalignment='top', fontweight='bold')
    
    # 2. Overlapping histograms by class
    genuine_probs = y_pred_proba[y_test == 0]
    fraud_probs = y_pred_proba[y_test == 1]
    
    ax2.hist(genuine_probs, bins=25, alpha=0.7, label=f'Genuine Jobs (n={len(genuine_probs)})', 
             color='#10b981', density=True, edgecolor='darkgreen', linewidth=0.8)
    ax2.hist(fraud_probs, bins=25, alpha=0.7, label=f'Fraudulent Jobs (n={len(fraud_probs)})', 
             color='#ef4444', density=True, edgecolor='darkred', linewidth=0.8)
    
    ax2.axvline(0.5, color='black', linestyle='-', linewidth=3, alpha=0.8, label='Decision Threshold')
    ax2.set_xlabel('Fraud Probability', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=14, fontweight='bold')
    ax2.set_title('Probability Distribution by Actual Class', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative distribution
    sorted_probs = np.sort(y_pred_proba)
    cumulative = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
    
    ax3.plot(sorted_probs, cumulative, linewidth=3, color='blue', label='Cumulative Distribution')
    ax3.axvline(0.3, color='green', linestyle='--', linewidth=2, alpha=0.8, label='30% Threshold')
    ax3.axvline(0.5, color='red', linestyle='-', linewidth=2, alpha=0.8, label='50% Threshold')
    ax3.axvline(0.7, color='darkred', linestyle='--', linewidth=2, alpha=0.8, label='70% Threshold')
    
    ax3.set_xlabel('Fraud Probability', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Cumulative Proportion', fontsize=14, fontweight='bold')
    ax3.set_title('Cumulative Distribution of Fraud Probabilities', fontsize=16, fontweight='bold')
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. Box plot by risk level
    low_risk_probs = y_pred_proba[(y_pred_proba >= 0) & (y_pred_proba < 0.3)]
    medium_risk_probs = y_pred_proba[(y_pred_proba >= 0.3) & (y_pred_proba < 0.7)]
    high_risk_probs = y_pred_proba[y_pred_proba >= 0.7]
    
    box_data = [low_risk_probs, medium_risk_probs, high_risk_probs]
    box_labels = ['Low Risk\n(0-30%)', 'Medium Risk\n(30-70%)', 'High Risk\n(70-100%)']
    
    bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True, 
                     boxprops=dict(alpha=0.7), medianprops=dict(linewidth=2))
    
    colors = ['#10b981', '#f59e0b', '#ef4444']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax4.set_ylabel('Fraud Probability', fontsize=14, fontweight='bold')
    ax4.set_title('Probability Distribution by Risk Level', fontsize=16, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Fraud Probability Histogram Analysis', 
                 fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

def create_classification_distribution(y_test):
    """Create genuine vs fraudulent distribution visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Calculate counts
    genuine_count = np.sum(y_test == 0)
    fraudulent_count = np.sum(y_test == 1)
    total_count = len(y_test)
    
    # 1. Simple bar chart
    categories = ['Genuine Jobs', 'Fraudulent Jobs']
    counts = [genuine_count, fraudulent_count]
    colors = ['#10b981', '#ef4444']
    
    bars = ax1.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        percentage = (count / total_count) * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height + total_count*0.02,
                f'{count:,}\n({percentage:.1f}%)', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('Number of Job Postings', fontsize=14, fontweight='bold')
    ax1.set_title('Overall Distribution: Genuine vs Fraudulent Jobs', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Pie chart
    wedges, texts, autotexts = ax2.pie(counts, labels=categories, colors=colors, 
                                      autopct='%1.1f%%', startangle=90,
                                      textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax2.set_title('Classification Distribution', fontsize=16, fontweight='bold')
    
    # 3. Horizontal bar chart with percentages
    y_pos = np.arange(len(categories))
    bars = ax3.barh(y_pos, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(categories, fontsize=12, fontweight='bold')
    ax3.set_xlabel('Number of Job Postings', fontsize=14, fontweight='bold')
    ax3.set_title('Job Classification Breakdown', fontsize=16, fontweight='bold')
    
    # Add percentage labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        percentage = (count / total_count) * 100
        ax3.text(width + total_count*0.01, bar.get_y() + bar.get_height()/2.,
                f'{count:,} ({percentage:.1f}%)', 
                ha='left', va='center', fontsize=11, fontweight='bold')
    
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Summary statistics
    ax4.axis('off')
    
    # Create summary text
    fraud_rate = (fraudulent_count / total_count) * 100
    genuine_rate = (genuine_count / total_count) * 100
    
    summary_text = f"""
    CLASSIFICATION SUMMARY
    ═══════════════════════════════
    
    Total Jobs Analyzed: {total_count:,}
    
    Genuine Jobs: {genuine_count:,}
    • Percentage: {genuine_rate:.1f}%
    • Ratio: {genuine_count/fraudulent_count:.1f}:1
    
    Fraudulent Jobs: {fraudulent_count:,}
    • Percentage: {fraud_rate:.1f}%
    • Detection Rate: {fraud_rate:.1f}%
    
    Dataset Balance:
    • Imbalance Ratio: {max(counts)/min(counts):.1f}:1
    • Minority Class: {min(fraud_rate, genuine_rate):.1f}%
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             fontsize=12, fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
             verticalalignment='top')
    
    plt.suptitle('Job Classification Distribution Analysis', 
                 fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

def create_risk_level_analysis(y_pred_proba, y_test):
    """Create detailed risk level analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Calculate risk levels
    low_risk = np.sum((y_pred_proba >= 0) & (y_pred_proba < 0.3))
    medium_risk = np.sum((y_pred_proba >= 0.3) & (y_pred_proba < 0.7))
    high_risk = np.sum(y_pred_proba >= 0.7)
    total = len(y_pred_proba)
    
    # 1. Risk level bar chart
    risk_categories = ['Low Risk\n(0-30%)', 'Medium Risk\n(30-70%)', 'High Risk\n(70-100%)']
    risk_counts = [low_risk, medium_risk, high_risk]
    risk_colors = ['#10b981', '#f59e0b', '#ef4444']
    
    bars = ax1.bar(risk_categories, risk_counts, color=risk_colors, alpha=0.8, 
                   edgecolor='black', linewidth=2)
    
    # Add labels
    for bar, count in zip(bars, risk_counts):
        height = bar.get_height()
        percentage = (count / total) * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                f'{count:,}\n({percentage:.1f}%)', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Number of Job Postings', fontsize=14, fontweight='bold')
    ax1.set_title('Risk Level Distribution', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Risk level pie chart
    wedges, texts, autotexts = ax2.pie(risk_counts, labels=risk_categories, colors=risk_colors,
                                      autopct='%1.1f%%', startangle=90,
                                      textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax2.set_title('Risk Level Proportions', fontsize=16, fontweight='bold')
    
    # 3. Precision analysis for each risk level
    risk_ranges = [(0, 0.3), (0.3, 0.7), (0.7, 1.0)]
    precisions = []
    
    for low, high in risk_ranges:
        mask = (y_pred_proba >= low) & (y_pred_proba < high)
        if np.sum(mask) > 0:
            precision = np.sum(y_test[mask]) / np.sum(mask)
            precisions.append(precision * 100)
        else:
            precisions.append(0)
    
    bars = ax3.bar(risk_categories, precisions, color=risk_colors, alpha=0.8,
                   edgecolor='black', linewidth=2)
    
    # Add precision labels
    for bar, precision in zip(bars, precisions):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{precision:.1f}%', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax3.set_ylabel('Precision (%)', fontsize=14, fontweight='bold')
    ax3.set_title('Precision by Risk Level', fontsize=16, fontweight='bold')
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Detailed statistics table
    ax4.axis('off')
    
    # Create statistics table
    stats_data = []
    for i, (category, count, precision) in enumerate(zip(risk_categories, risk_counts, precisions)):
        percentage = (count / total) * 100
        actual_fraud = np.sum(y_test[(y_pred_proba >= risk_ranges[i][0]) & 
                                   (y_pred_proba < risk_ranges[i][1])])
        stats_data.append([category.replace('\n', ' '), f'{count:,}', f'{percentage:.1f}%', 
                          f'{actual_fraud}', f'{precision:.1f}%'])
    
    table_data = [['Risk Level', 'Count', 'Percentage', 'Actual Fraud', 'Precision']] + stats_data
    
    table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center',
                     colColours=['lightgray']*5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color code the rows
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            table[(i, j)].set_facecolor(risk_colors[i-1])
            table[(i, j)].set_alpha(0.3)
    
    ax4.set_title('Risk Level Statistics Summary', fontsize=16, fontweight='bold', pad=20)
    
    plt.suptitle('Comprehensive Risk Level Analysis', 
                 fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

def create_detailed_probability_analysis(y_pred_proba, y_test):
    """Create detailed probability analysis with multiple views"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Probability vs Actual Class Scatter
    jitter = np.random.normal(0, 0.05, len(y_test))
    colors = ['#10b981' if label == 0 else '#ef4444' for label in y_test]
    
    scatter = ax1.scatter(y_test + jitter, y_pred_proba, alpha=0.6, c=colors, s=30)
    ax1.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
    ax1.set_xlabel('Actual Class (0=Genuine, 1=Fraudulent)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Fraud Probability', fontsize=12, fontweight='bold')
    ax1.set_title('Predicted Probability vs Actual Class', fontsize=14, fontweight='bold')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Genuine', 'Fraudulent'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Probability density estimation
    from scipy.stats import gaussian_kde
    
    # Create probability density curves
    x_range = np.linspace(0, 1, 100)
    
    if len(y_pred_proba[y_test == 0]) > 1:
        genuine_kde = gaussian_kde(y_pred_proba[y_test == 0])
        genuine_density = genuine_kde(x_range)
        ax2.plot(x_range, genuine_density, color='#10b981', linewidth=3, 
                label=f'Genuine Jobs (n={np.sum(y_test == 0)})')
        ax2.fill_between(x_range, genuine_density, alpha=0.3, color='#10b981')
    
    if len(y_pred_proba[y_test == 1]) > 1:
        fraud_kde = gaussian_kde(y_pred_proba[y_test == 1])
        fraud_density = fraud_kde(x_range)
        ax2.plot(x_range, fraud_density, color='#ef4444', linewidth=3,
                label=f'Fraudulent Jobs (n={np.sum(y_test == 1)})')
        ax2.fill_between(x_range, fraud_density, alpha=0.3, color='#ef4444')
    
    ax2.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
    ax2.set_xlabel('Fraud Probability', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax2.set_title('Probability Density by Class', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Calibration analysis
    from sklearn.calibration import calibration_curve
    
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    
    ax3.plot(prob_pred, prob_true, marker='o', linewidth=3, markersize=8, 
             color='blue', label='Model Calibration')
    ax3.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, 
             label='Perfect Calibration')
    ax3.fill_between(prob_pred, prob_true, alpha=0.3, color='blue')
    
    ax3.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
    ax3.set_title('Probability Calibration Plot', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistical summary
    ax4.axis('off')
    
    # Calculate comprehensive statistics
    mean_prob = np.mean(y_pred_proba)
    median_prob = np.median(y_pred_proba)
    std_prob = np.std(y_pred_proba)
    min_prob = np.min(y_pred_proba)
    max_prob = np.max(y_pred_proba)
    
    # Percentiles
    p25 = np.percentile(y_pred_proba, 25)
    p75 = np.percentile(y_pred_proba, 75)
    p90 = np.percentile(y_pred_proba, 90)
    p95 = np.percentile(y_pred_proba, 95)
    
    stats_text = f"""
    PROBABILITY STATISTICS SUMMARY
    ═══════════════════════════════════
    
    Central Tendency:
    • Mean:           {mean_prob:.4f}
    • Median:         {median_prob:.4f}
    • Mode Range:     {np.argmax(np.histogram(y_pred_proba, bins=20)[0])*0.05:.2f}-{(np.argmax(np.histogram(y_pred_proba, bins=20)[0])+1)*0.05:.2f}
    
    Spread:
    • Std Deviation:  {std_prob:.4f}
    • Min:            {min_prob:.4f}
    • Max:            {max_prob:.4f}
    • Range:          {max_prob - min_prob:.4f}
    
    Percentiles:
    • 25th:           {p25:.4f}
    • 75th:           {p75:.4f}
    • 90th:           {p90:.4f}
    • 95th:           {p95:.4f}
    
    Risk Distribution:
    • Low Risk:       {np.sum(y_pred_proba < 0.3):,} ({np.sum(y_pred_proba < 0.3)/len(y_pred_proba)*100:.1f}%)
    • Medium Risk:    {np.sum((y_pred_proba >= 0.3) & (y_pred_proba < 0.7)):,} ({np.sum((y_pred_proba >= 0.3) & (y_pred_proba < 0.7))/len(y_pred_proba)*100:.1f}%)
    • High Risk:      {np.sum(y_pred_proba >= 0.7):,} ({np.sum(y_pred_proba >= 0.7)/len(y_pred_proba)*100:.1f}%)
    """ {np.sum(y_pred_proba >= 0.7):,} ({np.sum(y_pred_proba >= 0.7)/len(y_pred_proba)*100:.1f}%)
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
             fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
             verticalalignment='top')
    
    plt.suptitle('Detailed Probability Analysis', 
                 fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics to console
    print("\n" + "="*60)
    print("📊 DETAILED PROBABILITY STATISTICS")
    print("="*60)
    print(f"Total Samples: {len(y_pred_proba):,}")
    print(f"Mean Probability: {mean_prob:.4f}")
    print(f"Median Probability: {median_prob:.4f}")
    print(f"Standard Deviation: {std_prob:.4f}")
    print(f"Min Probability: {min_prob:.4f}")
    print(f"Max Probability: {max_prob:.4f}")
    print(f"\nPercentile Analysis:")
    print(f"  25th Percentile: {p25:.4f}")
    print(f"  75th Percentile: {p75:.4f}")
    print(f"  90th Percentile: {p90:.4f}")
    print(f"  95th Percentile: {p95:.4f}")
    print("="*60)

if __name__ == "__main__":
    demonstrate_histogram_analysis()
