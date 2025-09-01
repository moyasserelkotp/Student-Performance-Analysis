# Student Performance Insights: From Cleaning to Clustering to Classification


<img width="1438" height="744" alt="image" src="https://github.com/user-attachments/assets/cc9f6c5f-a88e-4192-86b1-0830398df6ed" />


## 🎯 Project Overview

This comprehensive data science project analyzes student performance data to develop actionable insights and predictive models for early identification of at-risk students. Using 649 student records from the UCI Machine Learning Repository, we implement a complete pipeline from data cleaning to machine learning model deployment.

**Key Achievements:**
- 🎯 **87.7% accuracy** in predicting student success using behavioral features
- 📊 **3 distinct behavioral clusters** identified with targeted intervention strategies  
- 🔍 **92.1% ROC-AUC** for binary pass/fail classification
- 📈 **10+ actionable insights** for educational policy and student support

---

## 📁 Project Structure

```
student-performance-analysis/
│
├── 📊 Notebooks/
│   ├── 1_data_preparation_and_cleaning.ipynb
│   ├── 2_data_transformation_and_feature_engineering.ipynb  
│   ├── 3_exploratory_data_analysis.ipynb
│   ├── 4_data_visualization.ipynb
│   ├── 5_kmeans_clustering_analysis.ipynb
│   └── 6_supervised_learning_classification.ipynb
│
├── 📋 Reports/
│   ├── Technical_Report_Student_Performance_Analysis.md
│   └── Executive_Summary_Slides.pdf (to be created)
│
├── 💾 Data/ (generated during execution)
│   ├── student_data_cleaned.csv
│   ├── student_data_no_leakage.csv
│   ├── student_data_with_leakage.csv
│   ├── student_data_clustered.csv
│   └── [additional processed datasets]
│
├── 📈 Visualizations/ (generated during execution)
│   ├── histograms_numeric_variables.png
│   ├── correlation_heatmap.png
│   ├── cluster_performance_analysis.png
│   ├── model_performance_comparison.png
│   └── [additional charts]
│
├── ⚙️ Configuration/
│   ├── requirements.txt
│   └── README.md (this file)
│
└── 🚀 Scripts/ (optional)
|    └── run_full_pipeline.py (to be created)
│
└── 🚀 dashboard/ 
    ├── streamlit_eda_dashboard.py
    └── streamlit_model_dashboard.py
```

---

## 🔧 Setup & Installation

### Prerequisites
- Python 3.8+ (Python 3.9+ recommended)
- Jupyter Notebook or JupyterLab
- Git (for version control)

### Quick Start

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd student-performance-analysis
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux  
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter**:
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

5. **Run notebooks in order** (1 → 2 → 3 → 4 → 5 → 6)

### Alternative Installation Methods

**Using Conda:**
```bash
conda create -n student-analysis python=3.9
conda activate student-analysis
pip install -r requirements.txt
```

**Using Poetry:**
```bash
poetry install
poetry shell
jupyter notebook
```

---

## 🚀 Usage Instructions

### Step-by-Step Execution

#### 1️⃣ Data Preparation (`1_data_preparation_and_cleaning.ipynb`)
- Loads UCI Student Performance dataset
- Validates data quality and schema
- Handles missing values and outliers
- Generates data quality report

**Expected Output:**
- `student_data_cleaned.csv`
- Data quality metrics and summary

#### 2️⃣ Data Transformation (`2_data_transformation_and_feature_engineering.ipynb`)
- One-hot encoding for categorical variables
- Feature scaling and normalization
- Creates engineered features (attendance_rate, study_efficiency, etc.)
- Prepares datasets with/without data leakage

**Expected Output:**
- `student_data_no_leakage.csv`
- `student_data_with_leakage.csv`  
- Train/test split files

#### 3️⃣ Exploratory Data Analysis (`3_exploratory_data_analysis.ipynb`)
- Descriptive statistics and correlation analysis
- Hypothesis testing (5 key educational hypotheses)
- Group comparisons and behavioral pattern analysis
- Statistical significance testing

**Expected Output:**
- Comprehensive EDA insights
- Hypothesis test results
- Feature relationship analysis

#### 4️⃣ Data Visualization (`4_data_visualization.ipynb`)  
- Histograms of key numeric variables
- Boxplots and violin plots by student segments
- Correlation heatmaps
- Pass/fail analysis visualizations

**Expected Output:**
- 6+ high-quality visualizations saved as PNG files
- Statistical interpretation of visual patterns

#### 5️⃣ Clustering Analysis (`5_kmeans_clustering_analysis.ipynb`)
- K-means clustering on behavioral features
- Elbow method and silhouette analysis for optimal k
- Cluster profiling and performance analysis
- Actionable intervention strategies by cluster

**Expected Output:**
- `student_data_clustered.csv`
- Cluster visualization charts
- Intervention recommendations per cluster

#### 6️⃣ Classification Models (`6_supervised_learning_classification.ipynb`)
- Trains 3 classification algorithms (Logistic, RF, SVM)
- Hyperparameter tuning with GridSearchCV
- 5-fold cross-validation and performance evaluation
- Feature importance analysis and model comparison

**Expected Output:**
- Model performance comparison tables
- ROC curves and confusion matrices
- Feature importance rankings
- Final model recommendations

---

## 📊 Key Results & Insights

### 🎯 Predictive Performance
- **Best Model:** Random Forest (87.7% accuracy, 92.1% ROC-AUC)
- **Early Prediction:** Strong performance without mi
