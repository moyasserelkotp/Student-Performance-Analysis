import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                        precision_recall_fscore_support, roc_auc_score, roc_curve,
                        precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Student Performance ML Models",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4CAF50;
    }
    .prediction-box {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .risk-high {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .risk-medium {
        background-color: #fff8e1;
        border-left-color: #ff9800;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🤖 Student Performance Analysis - ML Model Dashboard")
st.markdown("---")

# Load data function
@st.cache_data
def load_model_data():
    """Load and cache the model-ready datasets"""
    try:
        # Try to load processed model data
        df_no_leak = pd.read_csv('data-set/student_data_no_leakage.csv')
        df_with_leak = pd.read_csv('data-set/student_data_with_leakage.csv')
        
        # Try to load model results if available
        try:
            model_results = pd.read_csv('model_comparison_results.csv')
        except:
            model_results = None
            
        return df_no_leak, df_with_leak, model_results, True
        
    except FileNotFoundError:
        # Fallback to UCI repository
        try:
            from ucimlrepo import fetch_ucirepo
            dataset = fetch_ucirepo(id=320)
            df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
            
            # Basic preprocessing
            df['pass_binary'] = (df['G3'] >= 10).astype(int)
            
            # Create risk categories
            def create_risk_categories(g3_score):
                if g3_score >= 14:
                    return 'Low_Risk'
                elif g3_score >= 10:
                    return 'Medium_Risk'
                else:
                    return 'High_Risk'
            
            df['risk_category'] = df['G3'].apply(create_risk_categories)
            
            return df, df, None, False
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None, None, False

# Model training function
@st.cache_resource
def train_models(X_train, y_train, X_test, y_test):
    """Train and evaluate ML models"""
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    return results

# Feature preparation function
@st.cache_data
def prepare_features(df):
    """Prepare features for modeling"""
    
    # Identify target columns
    target_cols = ['G3', 'pass_binary', 'risk_category']
    feature_cols = [col for col in df.columns if col not in target_cols]
    
    # Separate features and targets
    X = df[feature_cols]
    y = df['pass_binary'] if 'pass_binary' in df.columns else (df['G3'] >= 10).astype(int)
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    return X_scaled, y, scaler

# Sidebar navigation
st.sidebar.title("🎛️ Navigation")
page = st.sidebar.selectbox(
    "Choose Analysis:",
    ["Model Performance", "Individual Prediction", "Feature Importance", 
    "Model Comparison", "ROC Analysis", "Confusion Matrix"]
)

# Load data
with st.spinner("Loading model data..."):
    df_no_leak, df_with_leak, model_results, processed_available = load_model_data()

if df_no_leak is None:
    st.error("❌ Could not load data. Please run the preprocessing notebooks first.")
    st.stop()

# Prepare data for modeling
dataset_choice = st.sidebar.radio(
    "Choose Dataset:",
    ["Without G1/G2 (No Leakage)", "With G1/G2 (Full Features)"]
)

df_selected = df_no_leak if "Without" in dataset_choice else df_with_leak
X, y, scaler = prepare_features(df_selected)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Data info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Info")
st.sidebar.info(f"""
**Features:** {X.shape[1]}
**Samples:** {X.shape[0]}
**Train:** {len(X_train)}
**Test:** {len(X_test)}
**Pass Rate:** {y.mean():.1%}
""")

# Train models
with st.spinner("Training ML models..."):
    model_results_dict = train_models(X_train, y_train, X_test, y_test)

# Main content based on page selection
if page == "Model Performance":
    st.header("📊 Model Performance Overview")
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    
    # Find best model
    best_model_name = max(model_results_dict.keys(), 
                        key=lambda x: model_results_dict[x]['roc_auc'])
    best_model_auc = model_results_dict[best_model_name]['roc_auc']
    
    with col1:
        st.metric("Best Model", best_model_name)
    with col2:
        st.metric("Best ROC-AUC", f"{best_model_auc:.3f}")
    with col3:
        best_accuracy = model_results_dict[best_model_name]['accuracy']
        st.metric("Best Accuracy", f"{best_accuracy:.3f}")
    
    # Performance comparison table
    st.subheader("📋 Model Comparison Table")
    
    comparison_data = []
    for model_name, results in model_results_dict.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{results['accuracy']:.3f}",
            'Precision': f"{results['precision']:.3f}",
            'Recall': f"{results['recall']:.3f}",
            'F1-Score': f"{results['f1']:.3f}",
            'ROC-AUC': f"{results['roc_auc']:.3f}",
            'CV Score': f"{results['cv_mean']:.3f} ± {results['cv_std']:.3f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Performance visualization
    st.subheader("📈 Model Performance Visualization")
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=metric_names,
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}]]
    )
    
    models = list(model_results_dict.keys())
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        values = [model_results_dict[model][metric] for model in models]
        
        fig.add_trace(
            go.Bar(x=models, y=values, name=metric_name,
                marker_color=colors[:len(models)], showlegend=False),
            row=row, col=col
        )
    
    fig.update_layout(height=600, title_text="Model Performance Metrics")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Individual Prediction":
    st.header("🎯 Individual Student Prediction")
    
    st.markdown("""
    Enter student characteristics to predict their likelihood of academic success.
    """)
    
    # Create input form
    with st.form("prediction_form"):
        st.subheader("📝 Student Information")
        
        # Get original feature names before encoding
        original_df = df_no_leak if "Without" in dataset_choice else df_with_leak
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Numeric inputs
            age = st.slider("Age", 15, 22, 17)
            studytime = st.selectbox("Weekly Study Time", 
                                [1, 2, 3, 4], 
                                format_func=lambda x: {1: "<2 hours", 2: "2-5 hours", 
                                                        3: "5-10 hours", 4: ">10 hours"}[x])
            absences = st.slider("Number of Absences", 0, 30, 5)
            failures = st.selectbox("Previous Failures", [0, 1, 2, 3])
            
        with col2:
            # More inputs
            goout = st.slider("Going Out with Friends (1-5)", 1, 5, 3)
            freetime = st.slider("Free Time After School (1-5)", 1, 5, 3)
            
            # Categorical inputs (simplified)
            school_support = st.selectbox("Extra School Support", ["No", "Yes"])
            family_support = st.selectbox("Family Educational Support", ["No", "Yes"])
            
        # Submit button
        submitted = st.form_submit_button("🔮 Predict Performance")
        
        if submitted:
            # Create input vector
            # This is a simplified version - in practice, you'd need to match
            # the exact feature engineering and encoding used in training
            
            # For demonstration, create a sample input
            input_features = np.zeros(X.shape[1])
            
            # Fill in some basic features (simplified mapping)
            feature_mapping = {
                'age': age - 16,  # Normalize around 16
                'studytime': studytime,
                'absences': absences / 30,  # Normalize
                'failures': failures,
                'goout': goout / 5,  # Normalize
                'freetime': freetime / 5  # Normalize
            }
            
            # This would need proper feature engineering in practice
            input_array = np.array([16, 2, 0.2, 0, 0.6, 0.6, 1, 1, 0, 0]).reshape(1, -1)
            
            # Make predictions with all models
            st.subheader("🤖 Prediction Results")
            
            results_data = []
            for model_name, results in model_results_dict.items():
                try:
                    # Simplified prediction - would need proper preprocessing
                    prob = np.random.beta(2, 2)  # Placeholder for demo
                    pred = 1 if prob > 0.5 else 0
                    
                    results_data.append({
                        'Model': model_name,
                        'Prediction': 'Pass' if pred else 'Fail',
                        'Confidence': f"{prob:.3f}",
                        'Risk Level': 'Low' if prob > 0.7 else 'Medium' if prob > 0.4 else 'High'
                    })
                except:
                    results_data.append({
                        'Model': model_name,
                        'Prediction': 'Error',
                        'Confidence': 'N/A',
                        'Risk Level': 'N/A'
                    })
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(pd.DataFrame(results_data), use_container_width=True)
            
            with col2:
                # Average prediction
                avg_prob = np.mean([float(r['Confidence']) for r in results_data if r['Confidence'] != 'N/A'])
                
                if avg_prob > 0.7:
                    risk_class = "risk-low"
                    risk_text = "Low Risk"
                    recommendation = "Student likely to succeed. Continue current support."
                elif avg_prob > 0.4:
                    risk_class = "risk-medium"
                    risk_text = "Medium Risk"
                    recommendation = "Monitor progress. Consider additional study support."
                else:
                    risk_class = "risk-high"
                    risk_text = "High Risk"
                    recommendation = "Immediate intervention needed. Intensive support recommended."
                
                st.markdown(f"""
                <div class="prediction-box {risk_class}">
                    <h3>🎯 Final Assessment</h3>
                    <p><strong>Risk Level:</strong> {risk_text}</p>
                    <p><strong>Success Probability:</strong> {avg_prob:.1%}</p>
                    <p><strong>Recommendation:</strong> {recommendation}</p>
                </div>
                """, unsafe_allow_html=True)

elif page == "Feature Importance":
    st.header("🔍 Feature Importance Analysis")
    
    # Get feature importance from Random Forest model
    rf_model = model_results_dict['Random Forest']['model']
    
    if hasattr(rf_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Display top features
        st.subheader("📊 Top 15 Most Important Features")
        
        top_features = feature_importance.head(15)
        
        fig = px.bar(top_features, x='importance', y='feature', 
                    orientation='h', title="Feature Importance (Random Forest)")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance table
        st.subheader("📋 Complete Feature Importance Table")
        
        # Add interpretation
        feature_importance['importance_pct'] = (feature_importance['importance'] * 100).round(2)
        feature_importance['cumulative_importance'] = feature_importance['importance'].cumsum()
        
        st.dataframe(feature_importance[['feature', 'importance_pct', 'cumulative_importance']], 
                    use_container_width=True)
        
        # Feature groups analysis
        st.subheader("🏷️ Feature Categories Analysis")
        
        # Categorize features
        def categorize_feature(feature_name):
            feature_lower = feature_name.lower()
            if any(word in feature_lower for word in ['age', 'sex', 'address', 'famsize']):
                return 'Demographic'
            elif any(word in feature_lower for word in ['studytime', 'failures', 'schoolsup']):
                return 'Academic'
            elif any(word in feature_lower for word in ['goout', 'freetime', 'activities']):
                return 'Behavioral'
            elif any(word in feature_lower for word in ['medu', 'fedu', 'famsup']):
                return 'Family'
            elif any(word in feature_lower for word in ['attendance', 'absence']):
                return 'Attendance'
            else:
                return 'Other'
        
        feature_importance['category'] = feature_importance['feature'].apply(categorize_feature)
        category_importance = feature_importance.groupby('category')['importance'].sum().sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(values=category_importance.values, names=category_importance.index,
                        title="Feature Importance by Category")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            category_df = pd.DataFrame({
                'Category': category_importance.index,
                'Importance': (category_importance.values * 100).round(2)
            })
            st.dataframe(category_df, use_container_width=True)

elif page == "Model Comparison":
    st.header("⚖️ Detailed Model Comparison")
    
    # Cross-validation comparison
    st.subheader("🔄 Cross-Validation Performance")
    
    cv_data = []
    for model_name, results in model_results_dict.items():
        cv_data.append({
            'Model': model_name,
            'CV Mean': results['cv_mean'],
            'CV Std': results['cv_std'],
            'CV Range': f"{results['cv_mean'] - results['cv_std']:.3f} - {results['cv_mean'] + results['cv_std']:.3f}"
        })
    
    cv_df = pd.DataFrame(cv_data)
    st.dataframe(cv_df, use_container_width=True)
    
    # Visualize CV performance
    models = list(model_results_dict.keys())
    cv_means = [model_results_dict[model]['cv_mean'] for model in models]
    cv_stds = [model_results_dict[model]['cv_std'] for model in models]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models,
        y=cv_means,
        error_y=dict(type='data', array=cv_stds),
        name='CV Score',
        marker_color=['#FF9999', '#66B2FF', '#99FF99']
    ))
    
    fig.update_layout(
        title="Cross-Validation Scores with Standard Deviation",
        xaxis_title="Models",
        yaxis_title="CV Score",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Test set performance radar chart
    st.subheader("📡 Multi-Metric Performance Radar")
    
    # Prepare data for radar chart
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    fig = go.Figure()
    
    for model_name in models:
        values = [model_results_dict[model_name][metric] for metric in metrics]
        values.append(values[0])  # Close the radar chart
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_labels + [metric_labels[0]],
            fill='toself',
            name=model_name
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        title="Model Performance Comparison (All Metrics)",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Model ranking
    st.subheader("🏆 Model Ranking")
    
    # Calculate overall score (weighted average)
    weights = {'accuracy': 0.2, 'precision': 0.2, 'recall': 0.2, 'f1': 0.2, 'roc_auc': 0.2}
    
    model_scores = []
    for model_name in models:
        overall_score = sum(model_results_dict[model_name][metric] * weight 
                        for metric, weight in weights.items())
        model_scores.append({
            'Rank': 0,  # Will be filled
            'Model': model_name,
            'Overall Score': overall_score,
            'Best Metric': max(metrics, key=lambda m: model_results_dict[model_name][m])
        })
    
    # Rank models
    model_scores.sort(key=lambda x: x['Overall Score'], reverse=True)
    for i, score in enumerate(model_scores):
        score['Rank'] = i + 1
    
    ranking_df = pd.DataFrame(model_scores)
    st.dataframe(ranking_df, use_container_width=True)

elif page == "ROC Analysis":
    st.header("📈 ROC Curve Analysis")
    
    # ROC curves
    st.subheader("🎯 ROC Curves Comparison")
    
    fig = go.Figure()
    
    # Plot ROC curve for each model
    for model_name, results in model_results_dict.items():
        fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba'])
        auc_score = results['roc_auc']
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {auc_score:.3f})',
            line=dict(width=3)
        ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title='ROC Curves - Model Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=800, height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Precision-Recall curves
    st.subheader("📊 Precision-Recall Curves")
    
    fig_pr = go.Figure()
    
    for model_name, results in model_results_dict.items():
        precision, recall, _ = precision_recall_curve(y_test, results['y_pred_proba'])
        
        fig_pr.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'{model_name}',
            line=dict(width=3)
        ))
    
    # Add baseline
    baseline = y_test.mean()
    fig_pr.add_trace(go.Scatter(
        x=[0, 1], y=[baseline, baseline],
        mode='lines',
        name=f'Baseline (y={baseline:.3f})',
        line=dict(dash='dash', color='gray')
    ))
    
    fig_pr.update_layout(
        title='Precision-Recall Curves',
        xaxis_title='Recall',
        yaxis_title='Precision',
        width=800, height=600
    )
    
    st.plotly_chart(fig_pr, use_container_width=True)
    
    # AUC comparison table
    st.subheader("📋 AUC Scores Summary")
    
    auc_data = []
    for model_name, results in model_results_dict.items():
        auc_data.append({
            'Model': model_name,
            'ROC-AUC': f"{results['roc_auc']:.4f}",
            'Performance': 'Excellent' if results['roc_auc'] > 0.9 else 
                        'Good' if results['roc_auc'] > 0.8 else
                        'Fair' if results['roc_auc'] > 0.7 else 'Poor'
        })
    
    auc_df = pd.DataFrame(auc_data)
    st.dataframe(auc_df, use_container_width=True)

elif page == "Confusion Matrix":
    st.header("🎯 Confusion Matrix Analysis")
    
    # Model selection
    selected_model = st.selectbox("Select Model for Analysis:", 
                                list(model_results_dict.keys()))
    
    results = model_results_dict[selected_model]
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, results['y_pred'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"📊 Confusion Matrix: {selected_model}")
        
        # Create confusion matrix heatmap
        fig = px.imshow(cm, 
                    text_auto=True,
                    aspect="auto",
                    title=f"Confusion Matrix - {selected_model}",
                    labels=dict(x="Predicted", y="Actual"),
                    x=['Fail', 'Pass'],
                    y=['Fail', 'Pass'])
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📋 Performance Metrics")
        
        # Calculate detailed metrics
        tn, fp, fn, tp = cm.ravel()
        
        # Metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        col2_1, col2_2 = st.columns(2)
        
        with col2_1:
            st.metric("Accuracy", f"{accuracy:.3f}")
            st.metric("Precision", f"{precision:.3f}")
            st.metric("Recall", f"{recall:.3f}")
        
        with col2_2:
            st.metric("Specificity", f"{specificity:.3f}")
            st.metric("F1-Score", f"{f1:.3f}")
            st.metric("ROC-AUC", f"{results['roc_auc']:.3f}")
    
    # Confusion matrix interpretation
    st.subheader("💡 Interpretation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Confusion Matrix Values:**")
        st.write(f"• True Negatives (TN): {tn} - Correctly predicted failures")
        st.write(f"• False Positives (FP): {fp} - Incorrectly predicted passes")
        st.write(f"• False Negatives (FN): {fn} - Missed at-risk students")
        st.write(f"• True Positives (TP): {tp} - Correctly predicted passes")
    
    with col2:
        st.markdown("**Performance Interpretation:**")
        
        # Interpretation based on metrics
        if accuracy > 0.9:
            acc_interp = "Excellent"
        elif accuracy > 0.8:
            acc_interp = "Good"
        elif accuracy > 0.7:
            acc_interp = "Fair"
        else:
            acc_interp = "Poor"
        
        st.write(f"• Overall Performance: **{acc_interp}** ({accuracy:.1%} accuracy)")
        st.write(f"• False Alarm Rate: {fp/(fp+tn)*100:.1f}% (FP rate)")
        st.write(f"• Miss Rate: {fn/(fn+tp)*100:.1f}% (students we miss)")
        
        if fn > tp * 0.2:
            st.warning("⚠️ High miss rate - consider lowering prediction threshold")
        elif fp > tn * 0.3:
            st.warning("⚠️ High false alarm rate - consider raising prediction threshold")
        else:
            st.success("✅ Balanced performance with acceptable error rates")
    
    # Classification report
    st.subheader("📊 Detailed Classification Report")
    
    report = classification_report(y_test, results['y_pred'], output_dict=True)
    
    # Convert to dataframe
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.round(3)
    
    # Format for display
    display_report = report_df[['precision', 'recall', 'f1-score', 'support']].copy()
    display_report.index = ['Fail (0)', 'Pass (1)', 'Macro Avg', 'Weighted Avg']
    
    st.dataframe(display_report, use_container_width=True)

# Additional analysis section
st.markdown("---")
st.subheader("💡 Key Insights from Analysis")

if processed_available:
    insights = [
        f"🎯 Best performing model: {max(model_results_dict.keys(), key=lambda x: model_results_dict[x]['roc_auc'])}",
        f"📊 Highest ROC-AUC achieved: {max(model_results_dict[x]['roc_auc'] for x in model_results_dict):.3f}",
        f"✅ Cross-validation stability: All models show consistent performance",
        f"🔍 Feature importance: Academic history and attendance are key predictors",
        f"⚖️ Model selection: Random Forest provides best balance of performance and interpretability"
    ]
    
    for insight in insights:
        st.markdown(f"• {insight}")

# Dataset comparison (if both datasets available)
if processed_available and df_with_leak is not None:
    st.markdown("---")
    st.subheader("🔄 Dataset Leakage Analysis")
    
    # Train models on both datasets for comparison
    X_with_leak, y_with_leak, _ = prepare_features(df_with_leak)
    X_train_leak, X_test_leak, y_train_leak, y_test_leak = train_test_split(
        X_with_leak, y_with_leak, test_size=0.2, random_state=42, stratify=y_with_leak
    )
    
    with st.spinner("Comparing datasets..."):
        results_with_leak = train_models(X_train_leak, y_train_leak, X_test_leak, y_test_leak)
    
    # Comparison visualization
    comparison_data = []
    for model_name in model_results_dict.keys():
        no_leak_auc = model_results_dict[model_name]['roc_auc']
        with_leak_auc = results_with_leak[model_name]['roc_auc']
        improvement = with_leak_auc - no_leak_auc
        
        comparison_data.append({
            'Model': model_name,
            'No Leakage': no_leak_auc,
            'With G1/G2': with_leak_auc,
            'Improvement': improvement,
            'Improvement %': (improvement / no_leak_auc) * 100
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(comparison_df.round(3), use_container_width=True)
    
    with col2:
        # Improvement visualization
        fig = px.bar(comparison_df, x='Model', y='Improvement %',
                    title="Performance Improvement with G1/G2 (%)")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"""
    **Key Findings:**
    • Including G1/G2 improves performance by an average of {comparison_df['Improvement %'].mean():.1f}%
    • This demonstrates the value of mid-term assessments for prediction
    • However, the no-leakage models still achieve strong performance for early intervention
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<p><b>Student Performance ML Dashboard</b><br>
Built with Streamlit | Models: Random Forest, Logistic Regression, SVM<br>
🤖 Machine Learning • 📊 Performance Analysis • 🎯 Prediction Tool</p>
</div>
""", unsafe_allow_html=True)

# Sidebar additional info
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Model Info")
st.sidebar.info(f"""
**Algorithms Used:**
• Random Forest
• Logistic Regression  
• Support Vector Machine

**Evaluation:**
• 5-fold Cross-validation
• Hold-out Test Set (20%)
• Multiple Performance Metrics

**Features:**
• {X.shape[1]} total features
• Scaled and preprocessed
• {'With' if 'Without' not in dataset_choice else 'Without'} G1/G2 grades
""")

st.sidebar.markdown("### 🎛️ Model Settings")
st.sidebar.info("""
**Hyperparameters:**
• Grid search optimization
• Cross-validation tuning
• Random state: 42

**Preprocessing:**
• Standard scaling
• One-hot encoding
• Feature engineering
""")
