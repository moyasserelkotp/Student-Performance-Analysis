import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind
import warnings

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Student Performance EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff6b6b;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: black !important;
    }
    .stMetric [data-testid="stMetricLabel"] {
        color: black !important;
    }
    .highlight {
        background-color: #e1f5fe;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Title and description
st.title("📊 Student Performance Analysis - EDA Dashboard")
st.markdown("---")


# Load data function
@st.cache_data
def load_data():
    """Load and cache the student performance dataset"""
    try:
        # Try to load processed data first
        df_no_leak = pd.read_csv("data-set/student_data_no_leakage.csv")
        df_clustered = pd.read_csv("data-set/student_data_clustered.csv")
        return df_no_leak, df_clustered, True
    except FileNotFoundError:
        # Fallback to UCI repository
        try:
            from ucimlrepo import fetch_ucirepo

            dataset = fetch_ucirepo(id=320)
            df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
            # Create basic binary target
            df["pass_binary"] = (df["G3"] >= 10).astype(int)
            return df, df, False
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None, False


# Sidebar for navigation and controls
st.sidebar.title("📋 Navigation")
analysis_type = st.sidebar.selectbox(
    "Choose Analysis Type:",
    [
        "Dataset Overview",
        "Univariate Analysis",
        "Bivariate Analysis",
        "Correlation Analysis",
        "Clustering Analysis",
        "Statistical Tests",
    ],
)

# Load data
with st.spinner("Loading student performance data..."):
    df, df_clustered, processed_available = load_data()

if df is None:
    st.error(
        "❌ Could not load data. Please ensure the data files are available or run the preprocessing notebooks first."
    )
    st.stop()

# Data info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Info")
st.sidebar.info(
    f"""
**Shape:** {df.shape[0]} students × {df.shape[1]} features
**Source:** UCI ML Repository
**Processed:** {'✅' if processed_available else '❌'}
"""
)

# Main dashboard content
if analysis_type == "Dataset Overview":
    st.header("🎯 Dataset Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Students", df.shape[0])

    with col2:
        st.metric("Features", df.shape[1])

    with col3:
        if "G3" in df.columns:
            avg_grade = df["G3"].mean()
            st.metric("Average Grade", f"{avg_grade:.1f}/20")
        else:
            st.metric("Data Status", "Raw")

    with col4:
        if "pass_binary" in df.columns:
            pass_rate = df["pass_binary"].mean() * 100
            st.metric("Pass Rate", f"{pass_rate:.1f}%")
        else:
            st.metric("Pass Rate", "N/A")

    # Dataset preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Missing values analysis
    st.subheader("🔍 Data Quality Assessment")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Missing Values:**")
        missing_data = df.isnull().sum()
        if missing_data.sum() == 0:
            st.success("✅ No missing values detected!")
        else:
            missing_df = pd.DataFrame(
                {
                    "Column": missing_data.index,
                    "Missing Count": missing_data.values,
                    "Missing %": (missing_data.values / len(df)) * 100,
                }
            )
            st.dataframe(missing_df[missing_df["Missing Count"] > 0])

    with col2:
        st.markdown("**Data Types:**")
        dtype_summary = df.dtypes.value_counts().to_dict()
        for dtype, count in dtype_summary.items():
            st.write(f"• **{dtype}:** {count} columns")

    # Feature categories
    st.subheader("📊 Feature Categories")

    # Identify feature types
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Categorical Features:**")
        for col in categorical_cols[:10]:  # Show first 10
            unique_count = df[col].nunique()
            st.write(f"• `{col}` ({unique_count} unique values)")
        if len(categorical_cols) > 10:
            st.write(f"... and {len(categorical_cols) - 10} more")

    with col2:
        st.markdown("**Numerical Features:**")
        for col in numerical_cols[:10]:  # Show first 10
            min_val, max_val = df[col].min(), df[col].max()
            st.write(f"• `{col}` (range: {min_val} - {max_val})")
        if len(numerical_cols) > 10:
            st.write(f"... and {len(numerical_cols) - 10} more")

elif analysis_type == "Univariate Analysis":
    st.header("📈 Univariate Analysis")

    # Feature selection
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Sidebar controls
    st.sidebar.markdown("### 🎛️ Controls")
    analysis_mode = st.sidebar.radio("Analysis Mode:", ["Numerical", "Categorical"])

    if analysis_mode == "Numerical" and numerical_cols:
        selected_feature = st.sidebar.selectbox(
            "Select Numerical Feature:", numerical_cols
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"📊 Distribution: {selected_feature}")

            # Create histogram with plotly
            fig = px.histogram(
                df,
                x=selected_feature,
                nbins=30,
                title=f"Distribution of {selected_feature}",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader(f"📋 Descriptive Statistics")

            # Statistics
            stats_data = df[selected_feature].describe()

            # Additional statistics
            skewness = df[selected_feature].skew()
            kurtosis = df[selected_feature].kurtosis()

            # Display as metrics
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("Mean", f"{stats_data['mean']:.2f}")
                st.metric("Median", f"{stats_data['50%']:.2f}")
                st.metric("Std Dev", f"{stats_data['std']:.2f}")

            with col2_2:
                st.metric("Min", f"{stats_data['min']:.2f}")
                st.metric("Max", f"{stats_data['max']:.2f}")
                st.metric("Skewness", f"{skewness:.2f}")

        # Box plot
        st.subheader(f"📦 Box Plot: {selected_feature}")
        fig_box = px.box(
            df, y=selected_feature, title=f"Box Plot of {selected_feature}"
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # Interpretation
        st.subheader("💡 Interpretation")
        if skewness > 1:
            skew_interp = "highly right-skewed"
        elif skewness > 0.5:
            skew_interp = "moderately right-skewed"
        elif skewness < -1:
            skew_interp = "highly left-skewed"
        elif skewness < -0.5:
            skew_interp = "moderately left-skewed"
        else:
            skew_interp = "approximately symmetric"

        st.markdown(
            f"""
        <div class="highlight">
        <b>Key Insights:</b><br>
        • Distribution is <b>{skew_interp}</b> (skewness = {skewness:.2f})<br>
        • Mean ({stats_data['mean']:.2f}) vs Median ({stats_data['50%']:.2f})<br>
        • Coefficient of Variation: {(stats_data['std']/stats_data['mean']*100):.1f}%
        </div>
        """,
            unsafe_allow_html=True,
        )

    elif analysis_mode == "Categorical" and categorical_cols:
        selected_feature = st.sidebar.selectbox(
            "Select Categorical Feature:", categorical_cols
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"📊 Distribution: {selected_feature}")

            # Value counts
            value_counts = df[selected_feature].value_counts()

            # Bar chart
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Distribution of {selected_feature}",
                labels={"x": selected_feature, "y": "Count"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader(f"📋 Category Summary")

            # Display value counts as table
            value_counts_df = pd.DataFrame(
                {
                    "Category": value_counts.index,
                    "Count": value_counts.values,
                    "Percentage": (value_counts.values / len(df)) * 100,
                }
            )

            st.dataframe(value_counts_df, use_container_width=True)

            # Metrics
            st.metric("Unique Categories", df[selected_feature].nunique())
            st.metric("Mode", value_counts.index[0])
            st.metric("Most Frequent %", f"{value_counts.iloc[0]/len(df)*100:.1f}%")

        # Pie chart
        st.subheader(f"🥧 Pie Chart: {selected_feature}")
        fig_pie = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=f"Proportion of {selected_feature}",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

elif analysis_type == "Bivariate Analysis":
    st.header("🔗 Bivariate Analysis")

    # Feature selection
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    all_cols = df.columns.tolist()

    # Sidebar controls
    st.sidebar.markdown("### 🎛️ Controls")
    x_feature = st.sidebar.selectbox("Select X Variable:", all_cols)
    y_feature = st.sidebar.selectbox("Select Y Variable:", all_cols)

    if x_feature != y_feature:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"📊 Scatter Plot")

            # Determine plot type based on variable types
            x_is_numeric = x_feature in numerical_cols
            y_is_numeric = y_feature in numerical_cols

            if x_is_numeric and y_is_numeric:
                # Scatter plot for numeric vs numeric
                fig = px.scatter(
                    df,
                    x=x_feature,
                    y=y_feature,
                    title=f"{y_feature} vs {x_feature}",
                    trendline="ols",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Correlation coefficient
                correlation = df[x_feature].corr(df[y_feature])
                st.metric("Correlation Coefficient", f"{correlation:.3f}")

            elif not x_is_numeric and y_is_numeric:
                # Box plot for categorical vs numeric
                fig = px.box(
                    df, x=x_feature, y=y_feature, title=f"{y_feature} by {x_feature}"
                )
                st.plotly_chart(fig, use_container_width=True)

            elif x_is_numeric and not y_is_numeric:
                # Box plot (swapped)
                fig = px.box(
                    df, x=y_feature, y=x_feature, title=f"{x_feature} by {y_feature}"
                )
                st.plotly_chart(fig, use_container_width=True)

            else:
                # Stacked bar for categorical vs categorical
                crosstab = pd.crosstab(df[x_feature], df[y_feature])
                fig = px.bar(crosstab, title=f"{y_feature} by {x_feature}")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader(f"📋 Statistical Summary")

            if x_is_numeric and y_is_numeric:
                # Correlation analysis
                correlation = df[x_feature].corr(df[y_feature])

                # Interpretation
                if abs(correlation) > 0.7:
                    strength = "Strong"
                elif abs(correlation) > 0.4:
                    strength = "Moderate"
                elif abs(correlation) > 0.2:
                    strength = "Weak"
                else:
                    strength = "Very weak"

                direction = "positive" if correlation > 0 else "negative"

                st.markdown(
                    f"""
                **Correlation Analysis:**
                - Coefficient: {correlation:.3f}
                - Strength: {strength}
                - Direction: {direction}
                """
                )

                # Regression statistics
                from scipy.stats import linregress

                slope, intercept, r_value, p_value, std_err = linregress(
                    df[x_feature], df[y_feature]
                )

                st.markdown(
                    f"""
                **Regression Statistics:**
                - R-squared: {r_value**2:.3f}
                - P-value: {p_value:.4f}
                - Slope: {slope:.3f}
                """
                )

            elif not x_is_numeric and y_is_numeric:
                # Group statistics
                group_stats = (
                    df.groupby(x_feature)[y_feature]
                    .agg(["count", "mean", "std"])
                    .round(2)
                )
                st.dataframe(group_stats)

                # ANOVA test
                groups = [
                    df[df[x_feature] == group][y_feature].values
                    for group in df[x_feature].unique()
                ]
                try:
                    f_stat, p_value = stats.f_oneway(*groups)
                    st.markdown(
                        f"""
                    **ANOVA Test:**
                    - F-statistic: {f_stat:.3f}
                    - P-value: {p_value:.4f}
                    - Significant: {'Yes' if p_value < 0.05 else 'No'}
                    """
                    )
                except:
                    st.write("Could not perform ANOVA test")

            else:
                # Chi-square test for categorical vs categorical
                try:
                    crosstab = pd.crosstab(df[x_feature], df[y_feature])
                    chi2, p_value, dof, expected = chi2_contingency(crosstab)

                    st.markdown(
                        f"""
                    **Chi-square Test:**
                    - Chi-square: {chi2:.3f}
                    - P-value: {p_value:.4f}
                    - Degrees of freedom: {dof}
                    - Significant: {'Yes' if p_value < 0.05 else 'No'}
                    """
                    )

                    st.dataframe(crosstab)
                except:
                    st.write("Could not perform chi-square test")

    # Multiple variable analysis
    if "G3" in df.columns:
        st.subheader("🎯 Performance Analysis")

        # Grade analysis by different factors
        performance_factor = st.selectbox(
            "Analyze G3 performance by:",
            [col for col in df.columns if col not in ["G3", "pass_binary"]],
        )

        if performance_factor in numerical_cols:
            # Correlation with grades
            correlation = df[performance_factor].corr(df["G3"])

            col1, col2 = st.columns(2)

            with col1:
                fig = px.scatter(
                    df,
                    x=performance_factor,
                    y="G3",
                    title=f"G3 vs {performance_factor}",
                    trendline="ols",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.metric("Correlation with G3", f"{correlation:.3f}")

                if abs(correlation) > 0.3:
                    impact = "Strong predictor"
                elif abs(correlation) > 0.1:
                    impact = "Moderate predictor"
                else:
                    impact = "Weak predictor"

                st.write(f"**Impact:** {impact}")
        else:
            # Box plot for categorical factors
            fig = px.box(
                df,
                x=performance_factor,
                y="G3",
                title=f"G3 Distribution by {performance_factor}",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Group statistics
            group_stats = (
                df.groupby(performance_factor)["G3"]
                .agg(["count", "mean", "std"])
                .round(2)
            )
            st.dataframe(group_stats)

elif analysis_type == "Correlation Analysis":
    st.header("🔗 Correlation Analysis")

    # Get numerical columns
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if len(numerical_cols) < 2:
        st.warning("Need at least 2 numerical features for correlation analysis")
    else:
        # Sidebar controls
        st.sidebar.markdown("### 🎛️ Controls")
        selected_features = st.sidebar.multiselect(
            "Select features for correlation analysis:",
            numerical_cols,
            default=numerical_cols[: min(10, len(numerical_cols))],
        )

        correlation_method = st.sidebar.selectbox(
            "Correlation Method:", ["pearson", "spearman", "kendall"]
        )

        if len(selected_features) >= 2:
            # Calculate correlation matrix
            corr_matrix = df[selected_features].corr(method=correlation_method)

            # Correlation heatmap
            st.subheader("🔥 Correlation Heatmap")

            fig = px.imshow(
                corr_matrix,
                title=f"{correlation_method.title()} Correlation Matrix",
                color_continuous_scale="RdYlBu_r",
                aspect="auto",
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Strong correlations
            st.subheader("💪 Strong Correlations")

            # Find strong correlations (excluding self-correlations)
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:  # Strong correlation threshold
                        strong_corr.append(
                            {
                                "Feature 1": corr_matrix.columns[i],
                                "Feature 2": corr_matrix.columns[j],
                                "Correlation": corr_val,
                                "Strength": (
                                    "Very Strong" if abs(corr_val) > 0.8 else "Strong"
                                ),
                            }
                        )

            if strong_corr:
                strong_corr_df = pd.DataFrame(strong_corr)
                strong_corr_df = strong_corr_df.sort_values(
                    "Correlation", key=abs, ascending=False
                )
                st.dataframe(strong_corr_df, use_container_width=True)
            else:
                st.info(
                    "No strong correlations (|r| > 0.5) found between selected features"
                )

            # Correlation with target variable
            if "G3" in selected_features:
                st.subheader("🎯 Correlations with Final Grade (G3)")

                g3_corrs = (
                    corr_matrix["G3"].drop("G3").sort_values(key=abs, ascending=False)
                )

                # Create bar chart
                fig = px.bar(
                    x=g3_corrs.index,
                    y=g3_corrs.values,
                    title="Correlations with Final Grade (G3)",
                    labels={"x": "Features", "y": "Correlation"},
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

                # Top correlations table
                top_corrs = pd.DataFrame(
                    {
                        "Feature": g3_corrs.index[:10],
                        "Correlation": g3_corrs.values[:10],
                        "Abs_Correlation": np.abs(g3_corrs.values[:10]),
                    }
                ).sort_values("Abs_Correlation", ascending=False)

                st.dataframe(
                    top_corrs[["Feature", "Correlation"]], use_container_width=True
                )

elif analysis_type == "Clustering Analysis" and processed_available:
    st.header("🎯 Clustering Analysis")

    if "cluster" in df_clustered.columns:
        # Cluster overview
        st.subheader("📊 Cluster Overview")

        cluster_counts = df_clustered["cluster"].value_counts().sort_index()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Number of Clusters", len(cluster_counts))

        with col2:
            st.metric("Largest Cluster", f"{cluster_counts.max()} students")

        with col3:
            st.metric("Smallest Cluster", f"{cluster_counts.min()} students")

        # Cluster distribution
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Cluster Size Distribution")
            fig = px.bar(
                x=cluster_counts.index,
                y=cluster_counts.values,
                title="Students per Cluster",
            )
            fig.update_layout(xaxis_title="Cluster", yaxis_title="Number of Students")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🥧 Cluster Proportions")
            fig = px.pie(
                values=cluster_counts.values,
                names=[f"Cluster {i}" for i in cluster_counts.index],
                title="Cluster Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Cluster characteristics
        st.subheader("🔍 Cluster Characteristics")

        # Select features to analyze
        numerical_cols = df_clustered.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        behavioral_features = [
            col
            for col in numerical_cols
            if col in ["studytime", "absences", "goout", "freetime", "age", "G3"]
        ]

        selected_features = st.multiselect(
            "Select features to compare across clusters:",
            behavioral_features,
            default=behavioral_features[:4],
        )

        if selected_features:
            # Cluster profiles
            cluster_profiles = (
                df_clustered.groupby("cluster")[selected_features].mean().round(2)
            )

            st.subheader("📋 Cluster Profiles (Mean Values)")
            st.dataframe(cluster_profiles, use_container_width=True)

            # Radar chart for cluster comparison
            if len(selected_features) >= 3:
                st.subheader("📡 Cluster Comparison Radar Chart")

                # Normalize data for radar chart
                from sklearn.preprocessing import MinMaxScaler

                scaler = MinMaxScaler()
                normalized_profiles = pd.DataFrame(
                    scaler.fit_transform(cluster_profiles.T).T,
                    columns=cluster_profiles.columns,
                    index=cluster_profiles.index,
                )

                fig = go.Figure()

                for cluster_id in normalized_profiles.index:
                    fig.add_trace(
                        go.Scatterpolar(
                            r=normalized_profiles.loc[cluster_id].values,
                            theta=normalized_profiles.columns,
                            fill="toself",
                            name=f"Cluster {cluster_id}",
                        )
                    )

                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="Cluster Profiles (Normalized)",
                    height=500,
                )

                st.plotly_chart(fig, use_container_width=True)

        # Performance by cluster
        if "G3" in df_clustered.columns:
            st.subheader("🎓 Academic Performance by Cluster")

            col1, col2 = st.columns(2)

            with col1:
                # Box plot of grades by cluster
                fig = px.box(
                    df_clustered,
                    x="cluster",
                    y="G3",
                    title="Grade Distribution by Cluster",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Performance metrics by cluster
                if "pass_binary" in df_clustered.columns:
                    cluster_performance = (
                        df_clustered.groupby("cluster")
                        .agg({"G3": ["mean", "std"], "pass_binary": "mean"})
                        .round(3)
                    )

                    cluster_performance.columns = [
                        "Avg_Grade",
                        "Std_Grade",
                        "Pass_Rate",
                    ]
                    cluster_performance["Pass_Rate"] = (
                        cluster_performance["Pass_Rate"] * 100
                    )

                    st.dataframe(cluster_performance, use_container_width=True)

                    # Pass rate by cluster
                    fig = px.bar(
                        x=cluster_performance.index,
                        y=cluster_performance["Pass_Rate"],
                        title="Pass Rate by Cluster (%)",
                    )
                    fig.update_xaxis(title="Cluster")
                    fig.update_yaxis(title="Pass Rate (%)")
                    st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning(
            "Clustering analysis requires processed data with cluster assignments. Please run the clustering notebook first."
        )

elif analysis_type == "Statistical Tests":
    st.header("🔬 Statistical Hypothesis Testing")

    st.markdown(
        """
    This section performs statistical tests to validate key hypotheses about student performance.
    """
    )

    # Hypothesis tests
    hypotheses = [
        {
            "name": "Higher study time → Better performance",
            "description": "Students with higher study time perform better academically",
            "test_type": "correlation_ttest",
        },
        {
            "name": "School support → Better performance",
            "description": "Students with school support perform better than those without",
            "test_type": "group_ttest",
        },
        {
            "name": "Higher absences → Lower performance",
            "description": "Students with more absences have lower academic performance",
            "test_type": "correlation_ttest",
        },
        {
            "name": "Past failures → Lower performance",
            "description": "Students with past failures perform worse currently",
            "test_type": "group_ttest",
        },
        {
            "name": "Family education → Better performance",
            "description": "Higher family education levels lead to better student performance",
            "test_type": "correlation_ttest",
        },
    ]

    # Test results
    test_results = []

    for hypothesis in hypotheses:
        st.subheader(f"🧪 {hypothesis['name']}")

        result = {"hypothesis": hypothesis["name"], "supported": False, "p_value": None}

        try:
            if hypothesis["test_type"] == "correlation_ttest":
                if (
                    hypothesis["name"].startswith("Higher study time")
                    and "studytime" in df.columns
                    and "G3" in df.columns
                ):
                    corr = df["studytime"].corr(df["G3"])
                    # T-test between high and low study time groups
                    high_study = (
                        df[df["studytime"] >= 3]["G3"]
                        if "studytime" in df.columns
                        else []
                    )
                    low_study = (
                        df[df["studytime"] <= 2]["G3"]
                        if "studytime" in df.columns
                        else []
                    )

                    if len(high_study) > 0 and len(low_study) > 0:
                        stat, p_val = ttest_ind(high_study, low_study)
                        result["p_value"] = p_val
                        result["supported"] = (
                            p_val < 0.05 and high_study.mean() > low_study.mean()
                        )

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Correlation", f"{corr:.3f}")
                        with col2:
                            st.metric("P-value", f"{p_val:.4f}")
                        with col3:
                            st.metric(
                                "Result",
                                (
                                    "✅ Supported"
                                    if result["supported"]
                                    else "❌ Not Supported"
                                ),
                            )

                        st.write(
                            f"High study time avg: {high_study.mean():.2f}, Low study time avg: {low_study.mean():.2f}"
                        )

                elif (
                    hypothesis["name"].startswith("Higher absences")
                    and "absences" in df.columns
                    and "G3" in df.columns
                ):
                    corr = df["absences"].corr(df["G3"])
                    high_abs = df[df["absences"] > df["absences"].median()]["G3"]
                    low_abs = df[df["absences"] <= df["absences"].median()]["G3"]

                    if len(high_abs) > 0 and len(low_abs) > 0:
                        stat, p_val = ttest_ind(high_abs, low_abs)
                        result["p_value"] = p_val
                        result["supported"] = p_val < 0.05 and corr < 0

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Correlation", f"{corr:.3f}")
                        with col2:
                            st.metric("P-value", f"{p_val:.4f}")
                        with col3:
                            st.metric(
                                "Result",
                                (
                                    "✅ Supported"
                                    if result["supported"]
                                    else "❌ Not Supported"
                                ),
                            )

                        st.write(
                            f"High absences avg G3: {high_abs.mean():.2f}, Low absences avg G3: {low_abs.mean():.2f}"
                        )

                elif hypothesis["name"].startswith("Family education"):
                    # Check for family education features
                    family_edu_cols = [
                        col
                        for col in df.columns
                        if "edu" in col.lower()
                        and (
                            "medu" in col.lower()
                            or "fedu" in col.lower()
                            or "family" in col.lower()
                        )
                    ]

                    if family_edu_cols and "G3" in df.columns:
                        edu_col = family_edu_cols[0]  # Use first available
                        corr = df[edu_col].corr(df["G3"])
                        high_edu = (
                            df[df[edu_col] >= 3]["G3"] if edu_col in df.columns else []
                        )
                        low_edu = (
                            df[df[edu_col] < 3]["G3"] if edu_col in df.columns else []
                        )

                        if len(high_edu) > 0 and len(low_edu) > 0:
                            stat, p_val = ttest_ind(high_edu, low_edu)
                            result["p_value"] = p_val
                            result["supported"] = p_val < 0.05 and corr > 0

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Correlation", f"{corr:.3f}")
                            with col2:
                                st.metric("P-value", f"{p_val:.4f}")
                            with col3:
                                st.metric(
                                    "Result",
                                    (
                                        "✅ Supported"
                                        if result["supported"]
                                        else "❌ Not Supported"
                                    ),
                                )

                            st.write(
                                f"High education avg G3: {high_edu.mean():.2f}, Low education avg G3: {low_edu.mean():.2f}"
                            )

            elif hypothesis["test_type"] == "group_ttest":
                if hypothesis["name"].startswith("School support"):
                    # Look for school support columns
                    support_cols = [
                        col for col in df.columns if "schoolsup" in col.lower()
                    ]

                    if support_cols and "G3" in df.columns:
                        support_col = support_cols[0]

                        if df[support_col].dtype == "object":
                            # Categorical variable
                            support_yes = (
                                df[df[support_col] == "yes"]["G3"]
                                if "yes" in df[support_col].values
                                else []
                            )
                            support_no = (
                                df[df[support_col] == "no"]["G3"]
                                if "no" in df[support_col].values
                                else []
                            )
                        else:
                            # Binary encoded variable
                            support_yes = df[df[support_col] == 1]["G3"]
                            support_no = df[df[support_col] == 0]["G3"]

                        if len(support_yes) > 0 and len(support_no) > 0:
                            stat, p_val = ttest_ind(support_yes, support_no)
                            result["p_value"] = p_val
                            result["supported"] = (
                                p_val < 0.05 and support_yes.mean() > support_no.mean()
                            )

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "With Support Avg", f"{support_yes.mean():.2f}"
                                )
                            with col2:
                                st.metric("P-value", f"{p_val:.4f}")
                            with col3:
                                st.metric(
                                    "Result",
                                    (
                                        "✅ Supported"
                                        if result["supported"]
                                        else "❌ Not Supported"
                                    ),
                                )

                            st.write(f"Without support avg: {support_no.mean():.2f}")

                elif hypothesis["name"].startswith("Past failures"):
                    if "failures" in df.columns and "G3" in df.columns:
                        no_failures = df[df["failures"] == 0]["G3"]
                        with_failures = df[df["failures"] > 0]["G3"]

                        if len(no_failures) > 0 and len(with_failures) > 0:
                            stat, p_val = ttest_ind(no_failures, with_failures)
                            result["p_value"] = p_val
                            result["supported"] = (
                                p_val < 0.05
                                and no_failures.mean() > with_failures.mean()
                            )

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "No Failures Avg", f"{no_failures.mean():.2f}"
                                )
                            with col2:
                                st.metric("P-value", f"{p_val:.4f}")
                            with col3:
                                st.metric(
                                    "Result",
                                    (
                                        "✅ Supported"
                                        if result["supported"]
                                        else "❌ Not Supported"
                                    ),
                                )

                            st.write(f"With failures avg: {with_failures.mean():.2f}")

        except Exception as e:
            st.error(f"Could not perform test: {e}")
            result["supported"] = False

        test_results.append(result)
        st.markdown("---")

    # Summary of all tests
    st.subheader("📋 Hypothesis Testing Summary")

    summary_df = pd.DataFrame(test_results)
    summary_df["Status"] = summary_df["supported"].map(
        {True: "✅ Supported", False: "❌ Not Supported"}
    )

    # Display summary
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Hypotheses Tested", len(test_results))
        st.metric("Supported", sum(result["supported"] for result in test_results))

    with col2:
        st.metric(
            "Not Supported", sum(not result["supported"] for result in test_results)
        )
        success_rate = (
            sum(result["supported"] for result in test_results)
            / len(test_results)
            * 100
        )
        st.metric("Success Rate", f"{success_rate:.0f}%")

    # Show detailed results
    display_df = summary_df[["hypothesis", "Status"]].copy()
    display_df.columns = ["Hypothesis", "Result"]
    st.dataframe(display_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666;'>
<p><b>Student Performance EDA Dashboard</b><br>
Built with Streamlit | Data from UCI ML Repository<br>
📊 Interactive Analysis • 🔍 Statistical Insights • 📈 Data Visualization</p>
</div>
""",
    unsafe_allow_html=True,
)
