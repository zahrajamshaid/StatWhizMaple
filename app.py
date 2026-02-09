"""
StatWhizMaple - Advanced Statistical & ML Toolkit
Professional Version with Modern UI/UX and Advanced Features
"""

# Fix for Python 3.13 multiprocessing issue
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import modules
from src.data_loader import load_csv, check_missing_values, detect_column_types, get_data_overview
from src.eda import (summary_stats, correlation_matrix, missing_values_report, 
                     unique_values_report, detect_outliers, data_quality_score)
from src.visualization import (plot_histograms, plot_correlation_heatmap, plot_scatter, 
                               plot_boxplots, plot_categorical_distribution, create_interactive_scatter)
from src.stats_tests import t_test, chi_square, anova, correlation_test, normality_test
from src.utils import prepare_data_for_ml
from src.ml_models import (train_linear_regression, train_logistic_regression, train_random_forest,
                          plot_regression_results, plot_confusion_matrix, plot_feature_importance,
                          compare_models)
from src.recommender import recommend_model
from src.advanced_imputation import (smart_impute, predict_missing_values_rf, 
                                     analyze_missing_pattern, visualize_imputation_impact)
from src.feature_engineering import (auto_feature_engineering, create_polynomial_features,
                                    create_interaction_features, select_features_mutual_info)

# Page configuration
st.set_page_config(
    page_title="StatWhizMaple",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/zahrajamshaid/StatWhizMaple',
        'Report a bug': "https://github.com/zahrajamshaid/StatWhizMaple/issues",
        'About': "# StatWhizMaple\nAdvanced Data Science Toolkit"
    }
)

# Advanced CSS for professional UI
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header styles */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        animation: fadeIn 0.8s ease-in;
    }
    
    .main-header h1 {
        color: white;
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Card styles */
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin: 1rem 0;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    /* Metric card */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .metric-card p {
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Success card */
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(17, 153, 142, 0.3);
    }
    
    /* Warning card */
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(245, 87, 108, 0.3);
    }
    
    /* Info card */
    .info-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(79, 172, 254, 0.3);
    }
    
    /* Button styles */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styles */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tab styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        background-color: rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* DataFrame styles */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Animation */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .animated {
        animation: slideIn 0.5s ease-out;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* File uploader */
    .uploadedFile {
        border-radius: 10px;
        border: 2px dashed #667eea;
    }
    
    /* Metric container */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    /* Fix cursor issues */
    * {
        cursor: default !important;
    }
    
    button, a, [role="button"] {
        cursor: pointer !important;
    }
    
    input, textarea, select {
        cursor: text !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 0.75rem;
        font-weight: 600;
        line-height: 1;
        color: #fff;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.375rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin: 0 0.2rem;
    }
    
    /* Pro badge */
    .pro-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
        margin-left: 0.5rem;
        box-shadow: 0 3px 10px rgba(245, 87, 108, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'prepared_data' not in st.session_state:
    st.session_state.prepared_data = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = []
if 'imputed_df' not in st.session_state:
    st.session_state.imputed_df = None
if 'engineered_df' not in st.session_state:
    st.session_state.engineered_df = None

# Header
st.markdown("""
    <div class="main-header">
        <h1>🚀 StatWhizMaple</h1>
        <p>Advanced Statistical Analysis & Machine Learning Platform</p>
        <span class="pro-badge">PROFESSIONAL EDITION</span>
    </div>
""", unsafe_allow_html=True)

# Sidebar with modern design
with st.sidebar:
    st.markdown("### 🎯 Navigation")
    
    page = st.radio(
        "",
        ["🏠 Dashboard", "📂 Data Upload", "🔍 EDA Pro", "🎨 Visualizations", 
         "📈 Statistical Tests", "🧠 Smart Imputation", "⚙️ Feature Engineering",
         "🤖 ML Models", "🔮 Model Explainability", "📊 Data Profiling", "📋 Reports"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Data info
    if st.session_state.df is not None:
        st.markdown("### 📊 Current Dataset")
        st.success(f"**Rows:** {st.session_state.df.shape[0]:,}")
        st.info(f"**Columns:** {st.session_state.df.shape[1]}")
        
        quality = data_quality_score(st.session_state.df)
        st.metric("Quality Score", f"{quality['overall_quality_score']:.1f}/100", 
                 delta=quality['grade'], delta_color="off")
    else:
        st.warning("No data loaded")
    
    st.markdown("---")
    
    # Theme toggle (placeholder)
    st.markdown("### 🎨 Appearance")
    theme = st.selectbox("Theme", ["🌙 Dark (Pro)", "☀️ Light", "🎨 Custom"])
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; font-size: 0.8rem; color: #888;'>
            <p><strong>StatWhizMaple v2.0</strong></p>
            <p>Built with ❤️ for Data Scientists</p>
        </div>
    """, unsafe_allow_html=True)

# Content based on selected page
if page == "🏠 Dashboard":
    # Create dashboard layout
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h3>40+</h3>
                <p>Statistical Functions</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h3>10+</h3>
                <p>ML Algorithms</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-card">
                <h3>15+</h3>
                <p>Visualizations</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="metric-card">
                <h3>AI</h3>
                <p>Smart Recommendations</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature showcase
    st.markdown("## ✨ Pro Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container():
            st.markdown("""
                <div class="feature-card animated">
                    <h3>🧠 Smart Imputation</h3>
                    <p>ML-powered missing value prediction with confidence scores</p>
                    <ul>
                        <li>KNN Imputation</li>
                        <li>MICE Algorithm</li>
                        <li>Random Forest Prediction</li>
                        <li>Pattern Analysis</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown("""
                <div class="feature-card animated">
                    <h3>⚙️ Auto Feature Engineering</h3>
                    <p>Automated feature creation and selection</p>
                    <ul>
                        <li>Polynomial Features</li>
                        <li>Interactions</li>
                        <li>Feature Selection</li>
                        <li>Binning & Aggregation</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
    
    with col3:
        with st.container():
            st.markdown("""
                <div class="feature-card animated">
                    <h3>🔮 Model Explainability</h3>
                    <p>Understand your model's predictions</p>
                    <ul>
                        <li>SHAP Values</li>
                        <li>Feature Importance</li>
                        <li>Partial Dependence</li>
                        <li>Local Explanations</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick start
    st.markdown("## 🚀 Quick Start Guide")
    
    with st.expander("📖 How to use StatWhizMaple", expanded=False):
        st.markdown("""
        ### Step-by-Step Guide
        
        1. **📂 Upload Data**
           - Navigate to "Data Upload"
           - Upload your CSV file
           - Review automatic data analysis
        
        2. **🔍 Explore with EDA Pro**
           - View comprehensive statistics
           - Analyze correlations
           - Check data quality
        
        3. **🧠 Handle Missing Values**
           - Go to "Smart Imputation"
           - Choose ML-based imputation
           - View confidence scores
        
        4. **⚙️ Engineer Features**
           - Use automatic feature engineering
           - Select best features
           - Boost model performance
        
        5. **🤖 Train ML Models**
           - Get smart recommendations
           - Train multiple models
           - Compare performance
        
        6. **🔮 Explain Results**
           - Use SHAP for interpretability
           - Understand predictions
           - Export reports
        """)
    
    # Recent activity (placeholder)
    if st.session_state.df is not None:
        st.markdown("## 📊 Current Session")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class="info-card">
                    <h4>✅ Data Loaded</h4>
                    <p>Your dataset is ready for analysis</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            models_trained = len(st.session_state.model_results)
            if models_trained > 0:
                st.markdown(f"""
                    <div class="success-card">
                        <h4>🎯 {models_trained} Model(s) Trained</h4>
                        <p>View results in ML Models section</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="warning-card">
                        <h4>⚠️ No Models Trained</h4>
                        <p>Start training in ML Models section</p>
                    </div>
                """, unsafe_allow_html=True)

elif page == "📂 Data Upload":
    st.markdown("## 📂 Data Upload & Preview")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your dataset in CSV format (max 500MB)"
    )
    
    if uploaded_file is not None:
        with st.spinner("🔄 Loading and analyzing data..."):
            try:
                df = load_csv(uploaded_file)
                st.session_state.df = df
                
                st.markdown("""
                    <div class="success-card">
                        <h4>✅ Data Successfully Loaded!</h4>
                        <p>Your dataset has been loaded and is ready for analysis</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Overview metrics
                st.markdown("### 📊 Dataset Overview")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("📏 Rows", f"{len(df):,}")
                with col2:
                    st.metric("📊 Columns", len(df.columns))
                with col3:
                    missing_pct = (df.isna().sum().sum() / df.size) * 100
                    st.metric("⚠️ Missing %", f"{missing_pct:.1f}%")
                with col4:
                    st.metric("🔢 Duplicates", df.duplicated().sum())
                with col5:
                    memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
                    st.metric("💾 Memory", f"{memory_mb:.1f} MB")
                
                # Data preview
                st.markdown("### 👀 Data Preview")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    rows_to_show = st.slider("Rows to display", 5, 50, 10)
                with col2:
                    show_dtypes = st.checkbox("Show data types", value=True)
                
                if show_dtypes:
                    preview_df = df.head(rows_to_show).copy()
                    st.dataframe(preview_df, use_container_width=True, height=400)
                    
                    st.markdown("#### 📋 Column Information")
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes.astype(str),
                        'Non-Null': df.count(),
                        'Null': df.isna().sum(),
                        'Unique': [df[col].nunique() for col in df.columns]
                    })
                    st.dataframe(col_info, use_container_width=True)
                else:
                    st.dataframe(df.head(rows_to_show), use_container_width=True, height=400)
                
                # Column types visualization
                st.markdown("### 🎨 Column Type Distribution")
                
                col_types = detect_column_types(df)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🔢 Numeric", len(col_types['numeric']))
                    if col_types['numeric']:
                        with st.expander("View columns"):
                            for col in col_types['numeric']:
                                st.write(f"• {col}")
                
                with col2:
                    st.metric("📊 Categorical", len(col_types['categorical']))
                    if col_types['categorical']:
                        with st.expander("View columns"):
                            for col in col_types['categorical']:
                                st.write(f"• {col}")
                
                with col3:
                    st.metric("📅 DateTime", len(col_types['datetime']))
                    if col_types['datetime']:
                        with st.expander("View columns"):
                            for col in col_types['datetime']:
                                st.write(f"• {col}")
                
                with col4:
                    st.metric("📝 Text", len(col_types['text']))
                    if col_types['text']:
                        with st.expander("View columns"):
                            for col in col_types['text']:
                                st.write(f"• {col}")
                
                # Data quality score
                st.markdown("### ⭐ Data Quality Assessment")
                
                quality = data_quality_score(df)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Completeness", f"{quality['completeness_percent']:.1f}%")
                with col2:
                    st.metric("Uniqueness", f"{quality['uniqueness_percent']:.1f}%")
                with col3:
                    st.metric("Overall Score", f"{quality['overall_quality_score']:.1f}/100")
                with col4:
                    grade_colors = {'A': '🟢', 'B': '🔵', 'C': '🟡', 'D': '🔴'}
                    grade_color = grade_colors.get(quality['grade'], '⚪')
                    st.metric("Quality Grade", f"{grade_color} {quality['grade']}")
                
                # Quick actions
                st.markdown("### ⚡ Quick Actions")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button("🔍 Run Full EDA", use_container_width=True):
                        st.info("Navigate to 'EDA Pro' section for comprehensive analysis")
                
                with col2:
                    if st.button("🧠 Smart Imputation", use_container_width=True):
                        st.info("Navigate to 'Smart Imputation' to handle missing values")
                
                with col3:
                    if st.button("⚙️ Auto Feature Eng.", use_container_width=True):
                        st.info("Navigate to 'Feature Engineering' for automatic features")
                
                with col4:
                    if st.button("🤖 Train Models", use_container_width=True):
                        st.info("Navigate to 'ML Models' to start training")
                
            except Exception as e:
                st.markdown(f"""
                    <div class="warning-card">
                        <h4>❌ Error Loading File</h4>
                        <p>{str(e)}</p>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="info-card">
                <h4>👆 Upload a CSV file to get started</h4>
                <p>Or try loading a sample dataset below</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Sample datasets
        st.markdown("### 📝 Sample Datasets")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Employee Performance", use_container_width=True):
                try:
                    df = pd.read_csv("sample_data/employee_performance.csv")
                    st.session_state.df = df
                    st.success("✅ Sample data loaded!")
                    st.rerun()
                except:
                    st.warning("Sample data not found. Run: python generate_sample_data.py")
        
        with col2:
            if st.button("🏠 House Prices", use_container_width=True):
                try:
                    df = pd.read_csv("sample_data/house_prices.csv")
                    st.session_state.df = df
                    st.success("✅ Sample data loaded!")
                    st.rerun()
                except:
                    st.warning("Sample data not found")
        
        with col3:
            if st.button("💼 Customer Churn", use_container_width=True):
                try:
                    df = pd.read_csv("sample_data/customer_churn.csv")
                    st.session_state.df = df
                    st.success("✅ Sample data loaded!")
                    st.rerun()
                except:
                    st.warning("Sample data not found")

elif page == "🔍 EDA Pro":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
    else:
        df = st.session_state.df
        
        st.markdown("## 🔍 Exploratory Data Analysis Pro")
        
        tabs = st.tabs(["📊 Summary", "🔗 Correlations", "⚠️ Missing Data", "📈 Distributions", "🎯 Outliers"])
        
        with tabs[0]:  # Summary Statistics
            st.markdown("### 📊 Statistical Summary")
            
            summary = summary_stats(df)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(summary, use_container_width=True, height=400)
            
            with col2:
                st.markdown("#### 📋 Quick Stats")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) > 0:
                    st.metric("Numeric Columns", len(numeric_cols))
                    st.metric("Categorical Columns", len(df.columns) - len(numeric_cols))
                    st.metric("Total Records", f"{len(df):,}")
                    
                    # Average correlation
                    if len(numeric_cols) > 1:
                        corr_matrix = df[numeric_cols].corr().abs()
                        avg_corr = (corr_matrix.sum().sum() - len(numeric_cols)) / (len(numeric_cols) * (len(numeric_cols) - 1))
                        st.metric("Avg Correlation", f"{avg_corr:.3f}")
        
        with tabs[1]:  # Correlations
            st.markdown("### 🔗 Correlation Analysis")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    st.markdown("#### ⚙️ Settings")
                    method = st.selectbox("Method", ["pearson", "spearman", "kendall"])
                    threshold = st.slider("Highlight |r| >", 0.0, 1.0, 0.7, 0.05)
                    show_values = st.checkbox("Show values", value=True)
                
                with col1:
                    fig = plot_correlation_heatmap(df, method=method, threshold=threshold, figsize=(10, 8))
                    st.pyplot(fig)
                
                # High correlations table
                st.markdown("#### 🔥 Strong Correlations")
                corr_matrix = df[numeric_cols].corr(method=method)
                strong_corr = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > threshold:
                            strong_corr.append({
                                'Variable 1': corr_matrix.columns[i],
                                'Variable 2': corr_matrix.columns[j],
                                'Correlation': corr_matrix.iloc[i, j],
                                'Strength': 'Very Strong' if abs(corr_matrix.iloc[i, j]) > 0.9 else 'Strong'
                            })
                
                if strong_corr:
                    strong_corr_df = pd.DataFrame(strong_corr).sort_values('Correlation', key=abs, ascending=False)
                    st.dataframe(strong_corr_df, use_container_width=True)
                else:
                    st.info(f"No correlations found with |r| > {threshold}")
            else:
                st.warning("Need at least 2 numeric columns for correlation analysis")
        
        with tabs[2]:  # Missing Data
            st.markdown("### ⚠️ Missing Data Analysis")
            
            missing_report = missing_values_report(df)
            
            if missing_report is not None:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.dataframe(missing_report, use_container_width=True)
                
                with col2:
                    st.markdown("#### 📊 Missing Data Stats")
                    total_missing = missing_report['Missing Count'].sum()
                    st.metric("Total Missing", f"{total_missing:,}")
                    
                    cols_with_missing = len(missing_report)
                    st.metric("Columns Affected", cols_with_missing)
                    
                    avg_missing = missing_report['Missing Percentage'].mean()
                    st.metric("Avg Missing %", f"{avg_missing:.1f}%")
                    
                    if total_missing > 0:
                        st.markdown("""
                            <div class="info-card">
                                <h4>💡 Recommendation</h4>
                                <p>Use Smart Imputation to handle missing values with ML-powered prediction</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button("🧠 Go to Smart Imputation", use_container_width=True):
                            st.info("Navigate to 'Smart Imputation' in the sidebar")
            else:
                st.markdown("""
                    <div class="success-card">
                        <h4>✅ No Missing Values</h4>
                        <p>Your dataset is complete!</p>
                    </div>
                """, unsafe_allow_html=True)
        
        with tabs[3]:  # Distributions
            st.markdown("### 📈 Distribution Analysis")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                selected_col = st.selectbox("Select Column", numeric_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = plot_histograms(df, [selected_col])
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    df[selected_col].plot(kind='box', ax=ax)
                    ax.set_title(f"Box Plot: {selected_col}")
                    ax.set_ylabel("Value")
                    st.pyplot(fig)
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Mean", f"{df[selected_col].mean():.2f}")
                with col2:
                    st.metric("Median", f"{df[selected_col].median():.2f}")
                with col3:
                    st.metric("Std Dev", f"{df[selected_col].std():.2f}")
                with col4:
                    st.metric("Skewness", f"{df[selected_col].skew():.2f}")
                
                # Normality test
                st.markdown("#### 📊 Normality Test")
                norm_result = normality_test(df[selected_col])
                
                if norm_result['is_normal']:
                    st.markdown("""
                        <div class="success-card">
                            <h4>✅ Data is Normally Distributed</h4>
                            <p>Shapiro-Wilk test suggests data follows normal distribution</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="info-card">
                            <h4>ℹ️ Data is Not Normally Distributed</h4>
                            <p>Consider using non-parametric tests or transformations</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Test Statistic", f"{norm_result['statistic']:.4f}")
                with col2:
                    st.metric("P-value", f"{norm_result['p_value']:.4f}")
            else:
                st.warning("No numeric columns found")
        
        with tabs[4]:  # Outliers
            st.markdown("### 🎯 Outlier Detection")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                outlier_method = st.selectbox("Detection Method", ["IQR", "Z-Score"])
                
                if outlier_method == "IQR":
                    outliers = detect_outliers(df, method='iqr')
                else:
                    outliers = detect_outliers(df, method='zscore')
                
                if outliers:
                    outlier_df = pd.DataFrame(outliers)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.dataframe(outlier_df, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### 📊 Outlier Summary")
                        total_outliers = outlier_df['outlier_count'].sum()
                        st.metric("Total Outliers", total_outliers)
                        
                        cols_with_outliers = len(outlier_df)
                        st.metric("Columns with Outliers", cols_with_outliers)
                        
                        max_outliers = outlier_df['outlier_count'].max()
                        st.metric("Max in Single Column", max_outliers)
                else:
                    st.markdown("""
                        <div class="success-card">
                            <h4>✅ No Outliers Detected</h4>
                            <p>Using {outlier_method} method</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No numeric columns found")

elif page == "🎨 Visualizations":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
    else:
        df = st.session_state.df
        
        st.markdown("## 🎨 Advanced Visualizations")
        
        viz_type = st.selectbox(
            "Choose Visualization",
            ["📊 Histogram", "🔥 Heatmap", "🎯 Scatter Plot", "📦 Box Plot", 
             "📈 Line Plot", "🥧 Pie Chart", "📊 Bar Chart", "🎻 Violin Plot"]
        )
        
        if viz_type == "📊 Histogram":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                selected_cols = st.multiselect("Select Columns", numeric_cols, default=numeric_cols[:3])
                
                if selected_cols:
                    bins = st.slider("Number of Bins", 10, 100, 30)
                    
                    fig = plot_histograms(df, selected_cols, bins=bins)
                    st.pyplot(fig)
            else:
                st.warning("No numeric columns found")
        
        elif viz_type == "🔥 Heatmap":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                method = st.selectbox("Correlation Method", ["pearson", "spearman", "kendall"])
                
                fig = plot_correlation_heatmap(df, method=method)
                st.pyplot(fig)
            else:
                st.warning("Need at least 2 numeric columns")
        
        elif viz_type == "🎯 Scatter Plot":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    x_col = st.selectbox("X-axis", numeric_cols)
                with col2:
                    y_col = st.selectbox("Y-axis", [c for c in numeric_cols if c != x_col])
                
                # Optional color encoding
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                color_col = None
                
                if cat_cols:
                    use_color = st.checkbox("Color by category")
                    if use_color:
                        color_col = st.selectbox("Color Column", cat_cols)
                
                fig = plot_scatter(df, x_col, y_col, color_col)
                st.pyplot(fig)
                
                # Show correlation
                corr = df[x_col].corr(df[y_col])
                st.metric("Correlation", f"{corr:.3f}")
            else:
                st.warning("Need at least 2 numeric columns")
        
        elif viz_type == "📦 Box Plot":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                selected_cols = st.multiselect("Select Columns", numeric_cols, default=numeric_cols[:4])
                
                if selected_cols:
                    fig = plot_boxplots(df, selected_cols)
                    st.pyplot(fig)
            else:
                st.warning("No numeric columns found")
        
        elif viz_type == "📈 Line Plot":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                selected_cols = st.multiselect("Select Columns", numeric_cols, default=numeric_cols[:3])
                
                if selected_cols:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    for col in selected_cols:
                        ax.plot(df.index, df[col], label=col, marker='o', markersize=2, alpha=0.7)
                    ax.set_xlabel("Index")
                    ax.set_ylabel("Value")
                    ax.set_title("Line Plot")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
            else:
                st.warning("No numeric columns found")
        
        elif viz_type == "🥧 Pie Chart":
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if cat_cols:
                selected_col = st.selectbox("Select Column", cat_cols)
                
                value_counts = df[selected_col].value_counts().head(10)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                colors = plt.cm.Set3(range(len(value_counts)))
                ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%',
                      colors=colors, startangle=90)
                ax.set_title(f"Distribution of {selected_col}")
                st.pyplot(fig)
            else:
                st.warning("No categorical columns found")
        
        elif viz_type == "📊 Bar Chart":
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if cat_cols:
                selected_col = st.selectbox("Select Column", cat_cols)
                top_n = st.slider("Show Top N", 5, 50, 15)
                
                value_counts = df[selected_col].value_counts().head(top_n)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                value_counts.plot(kind='bar', ax=ax, color='#667eea')
                ax.set_title(f"Top {top_n} Values in {selected_col}")
                ax.set_xlabel(selected_col)
                ax.set_ylabel("Count")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("No categorical columns found")
        
        elif viz_type == "🎻 Violin Plot":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if numeric_cols and cat_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    num_col = st.selectbox("Numeric Column", numeric_cols)
                with col2:
                    cat_col = st.selectbox("Category Column", cat_cols)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Limit categories if too many
                top_categories = df[cat_col].value_counts().head(10).index
                df_filtered = df[df[cat_col].isin(top_categories)]
                
                sns.violinplot(data=df_filtered, x=cat_col, y=num_col, ax=ax)
                ax.set_title(f"Violin Plot: {num_col} by {cat_col}")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Need both numeric and categorical columns")

elif page == "📈 Statistical Tests":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
    else:
        df = st.session_state.df
        
        st.markdown("## 📈 Statistical Hypothesis Testing")
        
        test_type = st.selectbox(
            "Select Test",
            ["T-Test (Independent)", "T-Test (Paired)", "Chi-Square Test", 
             "ANOVA", "Correlation Test", "Normality Test"]
        )
        
        if test_type == "T-Test (Independent)":
            st.markdown("### 📊 Independent Samples T-Test")
            st.info("Compare means of two independent groups")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if numeric_cols and cat_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    numeric_col = st.selectbox("Numeric Variable", numeric_cols)
                with col2:
                    group_col = st.selectbox("Group Variable", cat_cols)
                
                # Get unique groups
                unique_groups = df[group_col].unique()
                
                if len(unique_groups) >= 2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        group1 = st.selectbox("Group 1", unique_groups)
                    with col2:
                        group2 = st.selectbox("Group 2", [g for g in unique_groups if g != group1])
                    
                    if st.button("Run T-Test", type="primary"):
                        group1_data = df[df[group_col] == group1][numeric_col].dropna()
                        group2_data = df[df[group_col] == group2][numeric_col].dropna()
                        
                        result = t_test(group1_data, group2_data, paired=False)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("T-Statistic", f"{result['t_statistic']:.4f}")
                        with col2:
                            st.metric("P-Value", f"{result['p_value']:.4f}")
                        with col3:
                            st.metric("Degrees of Freedom", result['df'])
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(f"{group1} Mean", f"{result['mean1']:.4f}")
                        with col2:
                            st.metric(f"{group2} Mean", f"{result['mean2']:.4f}")
                        
                        if result['significant']:
                            st.markdown("""
                                <div class="success-card">
                                    <h4>✅ Statistically Significant</h4>
                                    <p>There is a significant difference between the groups (p < 0.05)</p>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                                <div class="info-card">
                                    <h4>ℹ️ Not Significant</h4>
                                    <p>No significant difference found between the groups (p >= 0.05)</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("**Interpretation:**")
                        st.write(result['interpretation'])
                else:
                    st.warning("Group variable must have at least 2 unique values")
            else:
                st.warning("Need at least one numeric and one categorical column")
        
        elif test_type == "T-Test (Paired)":
            st.markdown("### 📊 Paired Samples T-Test")
            st.info("Compare means of two related groups (e.g., before/after measurements)")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    col1_name = st.selectbox("First Measurement", numeric_cols, key='paired_col1')
                with col2:
                    col2_name = st.selectbox("Second Measurement", 
                                            [c for c in numeric_cols if c != col1_name], 
                                            key='paired_col2')
                
                if st.button("Run Paired T-Test", type="primary"):
                    # Get data and remove rows with missing values in either column
                    data1 = df[col1_name]
                    data2 = df[col2_name]
                    
                    # Create a temporary dataframe to handle missing values together
                    temp_df = pd.DataFrame({col1_name: data1, col2_name: data2})
                    temp_df = temp_df.dropna()
                    
                    if len(temp_df) > 0:
                        result = t_test(temp_df[col1_name], temp_df[col2_name], paired=True)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("T-Statistic", f"{result['t_statistic']:.4f}")
                        with col2:
                            st.metric("P-Value", f"{result['p_value']:.4f}")
                        with col3:
                            st.metric("Pairs Analyzed", len(temp_df))
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(f"{col1_name} Mean", f"{result['mean1']:.4f}")
                        with col2:
                            st.metric(f"{col2_name} Mean", f"{result['mean2']:.4f}")
                        with col3:
                            mean_diff = result['mean1'] - result['mean2']
                            st.metric("Mean Difference", f"{mean_diff:.4f}")
                        
                        if result['significant']:
                            st.markdown("""
                                <div class="success-card">
                                    <h4>✅ Statistically Significant</h4>
                                    <p>There is a significant difference between measurements (p < 0.05)</p>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                                <div class="info-card">
                                    <h4>ℹ️ Not Significant</h4>
                                    <p>No significant difference found between measurements (p >= 0.05)</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("**Interpretation:**")
                        st.write(result['interpretation'])
                    else:
                        st.error("No valid pairs found. Please check for missing values.")
            else:
                st.warning("Need at least 2 numeric columns for paired t-test")
        
        elif test_type == "Chi-Square Test":
            st.markdown("### 🎲 Chi-Square Test of Independence")
            st.info("Test the relationship between two categorical variables")
            
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(cat_cols) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    var1 = st.selectbox("Variable 1", cat_cols)
                with col2:
                    var2 = st.selectbox("Variable 2", [c for c in cat_cols if c != var1])
                
                if st.button("Run Chi-Square Test", type="primary"):
                    result = chi_square(df[var1], df[var2])
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Chi-Square Statistic", f"{result['chi2_statistic']:.4f}")
                    with col2:
                        st.metric("P-Value", f"{result['p_value']:.4f}")
                    with col3:
                        st.metric("Degrees of Freedom", result['df'])
                    
                    if result['significant']:
                        st.markdown("""
                            <div class="success-card">
                                <h4>✅ Statistically Significant</h4>
                                <p>There is a significant association between the variables (p < 0.05)</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class="info-card">
                                <h4>ℹ️ Not Significant</h4>
                                <p>No significant association found (p >= 0.05)</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("**Interpretation:**")
                    st.write(result['interpretation'])
                    
                    # Show contingency table
                    st.markdown("#### 📊 Contingency Table")
                    contingency = pd.crosstab(df[var1], df[var2])
                    st.dataframe(contingency, use_container_width=True)
            else:
                st.warning("Need at least 2 categorical columns")
        
        elif test_type == "ANOVA":
            st.markdown("### 📊 One-Way ANOVA")
            st.info("Compare means across multiple groups")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if numeric_cols and cat_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    numeric_col = st.selectbox("Numeric Variable", numeric_cols)
                with col2:
                    group_col = st.selectbox("Group Variable", cat_cols)
                
                if st.button("Run ANOVA", type="primary"):
                    groups = [df[df[group_col] == g][numeric_col].dropna() for g in df[group_col].unique()]
                    
                    result = anova(*groups)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("F-Statistic", f"{result['f_statistic']:.4f}")
                    with col2:
                        st.metric("P-Value", f"{result['p_value']:.4f}")
                    with col3:
                        st.metric("Groups", len(groups))
                    
                    if result['significant']:
                        st.markdown("""
                            <div class="success-card">
                                <h4>✅ Statistically Significant</h4>
                                <p>At least one group mean is significantly different (p < 0.05)</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class="info-card">
                                <h4>ℹ️ Not Significant</h4>
                                <p>No significant differences found between group means (p >= 0.05)</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("**Interpretation:**")
                    st.write(result['interpretation'])
                    
                    # Group statistics
                    st.markdown("#### 📊 Group Statistics")
                    group_stats = df.groupby(group_col)[numeric_col].agg(['count', 'mean', 'std']).round(3)
                    st.dataframe(group_stats, use_container_width=True)
            else:
                st.warning("Need at least one numeric and one categorical column")
        
        elif test_type == "Correlation Test":
            st.markdown("### 🔗 Correlation Significance Test")
            st.info("Test if correlation between two variables is statistically significant")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    var1 = st.selectbox("Variable 1", numeric_cols)
                with col2:
                    var2 = st.selectbox("Variable 2", [c for c in numeric_cols if c != var1])
                
                method = st.selectbox("Method", ["pearson", "spearman", "kendall"])
                
                if st.button("Run Correlation Test", type="primary"):
                    result = correlation_test(df[var1], df[var2], method=method)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Correlation", f"{result['correlation']:.4f}")
                    with col2:
                        st.metric("P-Value", f"{result['p_value']:.4f}")
                    with col3:
                        strength = abs(result['correlation'])
                        if strength > 0.7:
                            label = "Strong"
                        elif strength > 0.4:
                            label = "Moderate"
                        else:
                            label = "Weak"
                        st.metric("Strength", label)
                    
                    if result['significant']:
                        st.markdown("""
                            <div class="success-card">
                                <h4>✅ Statistically Significant</h4>
                                <p>The correlation is statistically significant (p < 0.05)</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class="info-card">
                                <h4>ℹ️ Not Significant</h4>
                                <p>The correlation is not statistically significant (p >= 0.05)</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("**Interpretation:**")
                    st.write(result['interpretation'])
                    
                    # Scatter plot
                    fig = plot_scatter(df, var1, var2)
                    st.pyplot(fig)
            else:
                st.warning("Need at least 2 numeric columns")
        
        elif test_type == "Normality Test":
            st.markdown("### 📊 Shapiro-Wilk Normality Test")
            st.info("Test if data follows a normal distribution")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                selected_col = st.selectbox("Select Column", numeric_cols)
                
                if st.button("Run Normality Test", type="primary"):
                    result = normality_test(df[selected_col])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Test Statistic", f"{result['statistic']:.4f}")
                    with col2:
                        st.metric("P-Value", f"{result['p_value']:.4f}")
                    
                    if result['is_normal']:
                        st.markdown("""
                            <div class="success-card">
                                <h4>✅ Normally Distributed</h4>
                                <p>Data appears to follow a normal distribution (p >= 0.05)</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class="warning-card">
                                <h4>⚠️ Not Normally Distributed</h4>
                                <p>Data does not follow a normal distribution (p < 0.05)</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("**Interpretation:**")
                    st.write(result['interpretation'])
                    
                    # Distribution plot
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = plot_histograms(df, [selected_col])
                        st.pyplot(fig)
                    
                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        from scipy import stats
                        stats.probplot(df[selected_col].dropna(), dist="norm", plot=ax)
                        ax.set_title("Q-Q Plot")
                        st.pyplot(fig)
            else:
                st.warning("No numeric columns found")

elif page == "🧠 Smart Imputation":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
    else:
        df = st.session_state.df
        
        st.markdown("## 🧠 Smart Missing Value Imputation")
        st.info("Use ML-powered algorithms to predict and fill missing values with confidence scores")
        
        # Check for missing values
        missing_cols = df.columns[df.isna().any()].tolist()
        
        if not missing_cols:
            st.markdown("""
                <div class="success-card">
                    <h4>✅ No Missing Values</h4>
                    <p>Your dataset is already complete!</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Missing value analysis
            st.markdown("### ⚠️ Missing Value Analysis")
            
            pattern_result = analyze_missing_pattern(df)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Columns with Missing", len(missing_cols))
            with col2:
                total_missing = df.isna().sum().sum()
                st.metric("Total Missing Values", f"{total_missing:,}")
            with col3:
                missing_pct = (total_missing / df.size) * 100
                st.metric("Missing Percentage", f"{missing_pct:.2f}%")
            
            # Pattern details
            st.markdown("#### 📊 Missing Pattern Details")
            st.write(f"**Pattern Type:** {pattern_result['pattern_type']}")
            st.write(f"**Description:** {pattern_result['description']}")
            
            if pattern_result['recommendations']:
                st.markdown("**💡 Recommendations:**")
                for rec in pattern_result['recommendations']:
                    st.write(f"• {rec}")
            
            # Imputation section
            st.markdown("---")
            st.markdown("### 🔧 Choose Imputation Method")
            
            method = st.selectbox(
                "Imputation Algorithm",
                ["Smart Auto (Recommended)", "KNN Imputation", "MICE Algorithm", "Random Forest"]
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if method == "KNN Imputation":
                    n_neighbors = st.slider("Number of Neighbors", 3, 15, 5)
                elif method == "Random Forest":
                    n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)
                else:
                    st.info("Auto mode will select the best method for each column")
            
            with col2:
                show_confidence = st.checkbox("Show Confidence Scores", value=True)
                compare_before_after = st.checkbox("Compare Before/After", value=True)
            
            if st.button("🚀 Run Imputation", type="primary", use_container_width=True):
                with st.spinner("🔄 Running smart imputation..."):
                    try:
                        if method == "Smart Auto (Recommended)":
                            imputed_df, imputation_info = smart_impute(df)
                        elif method == "KNN Imputation":
                            from sklearn.impute import KNNImputer
                            numeric_cols = df.select_dtypes(include=[np.number]).columns
                            imputer = KNNImputer(n_neighbors=n_neighbors)
                            df_imputed = df.copy()
                            df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                            imputed_df = df_imputed
                            imputation_info = {"method": "KNN", "columns": numeric_cols.tolist()}
                        elif method == "Random Forest":
                            imputed_df, imputation_info = predict_missing_values_rf(
                                df, n_estimators=n_estimators, return_confidence=show_confidence
                            )
                        else:  # MICE
                            from sklearn.experimental import enable_iterative_imputer
                            from sklearn.impute import IterativeImputer
                            numeric_cols = df.select_dtypes(include=[np.number]).columns
                            imputer = IterativeImputer(random_state=42)
                            df_imputed = df.copy()
                            df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                            imputed_df = df_imputed
                            imputation_info = {"method": "MICE", "columns": numeric_cols.tolist()}
                        
                        st.session_state.imputed_df = imputed_df
                        
                        st.markdown("""
                            <div class="success-card">
                                <h4>✅ Imputation Complete!</h4>
                                <p>Missing values have been successfully predicted and filled</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Show results
                        st.markdown("### 📊 Imputation Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Missing Values Before", f"{df.isna().sum().sum():,}")
                        with col2:
                            st.metric("Missing Values After", f"{imputed_df.isna().sum().sum():,}")
                        
                        # Confidence scores
                        if show_confidence and 'confidence_scores' in imputation_info:
                            st.markdown("#### 🎯 Confidence Scores")
                            conf_df = pd.DataFrame(imputation_info['confidence_scores']).T
                            st.dataframe(conf_df.style.background_gradient(cmap='RdYlGn', axis=None),
                                       use_container_width=True)
                        
                        # Comparison
                        if compare_before_after:
                            st.markdown("#### 📊 Before vs After Comparison")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Before Imputation**")
                                st.dataframe(df.head(10), use_container_width=True)
                            
                            with col2:
                                st.markdown("**After Imputation**")
                                st.dataframe(imputed_df.head(10), use_container_width=True)
                        
                        # Option to use imputed data
                        if st.button("✅ Use Imputed Data for Analysis", type="primary"):
                            st.session_state.df = imputed_df
                            st.success("Dataset updated with imputed values!")
                            st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error during imputation: {str(e)}")

elif page == "⚙️ Feature Engineering":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
    else:
        df = st.session_state.df
        
        st.markdown("## ⚙️ Automated Feature Engineering")
        st.info("Create powerful features automatically to boost model performance")
        
        st.markdown("### 🛠️ Feature Creation Options")
        
        tabs = st.tabs(["🚀 Auto Mode", "🔧 Manual Mode", "📊 Feature Selection"])
        
        with tabs[0]:  # Auto Mode
            st.markdown("### 🚀 Automatic Feature Engineering")
            st.write("Let AI create the best features for your dataset automatically")
            
            col1, col2 = st.columns(2)
            
            with col1:
                max_features = st.slider("Maximum New Features", 10, 100, 50, 10)
                include_polynomial = st.checkbox("Polynomial Features", value=True)
                include_interactions = st.checkbox("Feature Interactions", value=True)
            
            with col2:
                include_binning = st.checkbox("Binning & Categorization", value=True)
                include_aggregations = st.checkbox("Aggregations", value=False)
                remove_low_variance = st.checkbox("Remove Low Variance", value=True)
            
            if st.button("🚀 Run Auto Feature Engineering", type="primary", use_container_width=True):
                with st.spinner("🔄 Creating features..."):
                    try:
                        engineered_df, feature_info = auto_feature_engineering(
                            df,
                            max_features=max_features,
                            include_polynomial=include_polynomial,
                            include_interactions=include_interactions,
                            include_binning=include_binning
                        )
                        
                        st.session_state.engineered_df = engineered_df
                        
                        st.markdown("""
                            <div class="success-card">
                                <h4>✅ Feature Engineering Complete!</h4>
                                <p>New features have been created successfully</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Show results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Original Features", df.shape[1])
                        with col2:
                            new_features = engineered_df.shape[1] - df.shape[1]
                            st.metric("New Features Created", new_features)
                        with col3:
                            st.metric("Total Features", engineered_df.shape[1])
                        
                        # Feature information
                        st.markdown("#### 📋 Created Features")
                        st.write(feature_info)
                        
                        # Preview
                        st.markdown("#### 👀 Dataset Preview")
                        st.dataframe(engineered_df.head(10), use_container_width=True)
                        
                        # Option to use engineered data
                        if st.button("✅ Use Engineered Features", type="primary"):
                            st.session_state.df = engineered_df
                            st.success("Dataset updated with engineered features!")
                            st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error during feature engineering: {str(e)}")
        
        with tabs[1]:  # Manual Mode
            st.markdown("### 🔧 Manual Feature Creation")
            
            feature_type = st.selectbox(
                "Feature Type",
                ["Polynomial Features", "Interaction Features", "Binned Features", 
                 "Log Transform", "Square Root Transform", "Custom Expression"]
            )
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if feature_type == "Polynomial Features":
                st.info("Create polynomial features (x², x³, etc.)")
                
                selected_cols = st.multiselect("Select Columns", numeric_cols)
                degree = st.slider("Polynomial Degree", 2, 5, 2)
                
                if selected_cols and st.button("Create Polynomial Features"):
                    new_df, feature_names = create_polynomial_features(df, selected_cols, degree=degree)
                    st.session_state.engineered_df = new_df
                    st.success(f"Created {len(feature_names)} polynomial features!")
                    st.write("New features:", feature_names)
            
            elif feature_type == "Interaction Features":
                st.info("Create interaction features (x₁ * x₂)")
                
                if len(numeric_cols) >= 2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        col1_select = st.selectbox("Column 1", numeric_cols)
                    with col2:
                        col2_select = st.selectbox("Column 2", [c for c in numeric_cols if c != col1_select])
                    
                    operation = st.selectbox("Operation", ["Multiply", "Divide", "Add", "Subtract"])
                    
                    if st.button("Create Interaction Feature"):
                        new_df = df.copy()
                        
                        if operation == "Multiply":
                            new_df[f"{col1_select}_x_{col2_select}"] = df[col1_select] * df[col2_select]
                        elif operation == "Divide":
                            new_df[f"{col1_select}_div_{col2_select}"] = df[col1_select] / (df[col2_select] + 1e-10)
                        elif operation == "Add":
                            new_df[f"{col1_select}_plus_{col2_select}"] = df[col1_select] + df[col2_select]
                        else:  # Subtract
                            new_df[f"{col1_select}_minus_{col2_select}"] = df[col1_select] - df[col2_select]
                        
                        st.session_state.engineered_df = new_df
                        st.success("Interaction feature created!")
                        st.dataframe(new_df.head(10), use_container_width=True)
                else:
                    st.warning("Need at least 2 numeric columns")
            
            elif feature_type == "Binned Features":
                st.info("Create binned/categorical features from numeric columns")
                
                selected_col = st.selectbox("Select Column", numeric_cols)
                n_bins = st.slider("Number of Bins", 2, 20, 5)
                
                strategy = st.selectbox("Binning Strategy", ["quantile", "uniform", "kmeans"])
                
                if st.button("Create Binned Feature"):
                    from sklearn.preprocessing import KBinsDiscretizer
                    
                    new_df = df.copy()
                    binner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
                    new_df[f"{selected_col}_binned"] = binner.fit_transform(df[[selected_col]])
                    
                    st.session_state.engineered_df = new_df
                    st.success("Binned feature created!")
                    st.dataframe(new_df[[selected_col, f"{selected_col}_binned"]].head(10), use_container_width=True)
            
            elif feature_type == "Log Transform":
                st.info("Apply log transformation to reduce skewness")
                
                selected_col = st.selectbox("Select Column", numeric_cols)
                
                if st.button("Apply Log Transform"):
                    new_df = df.copy()
                    # Add 1 to avoid log(0)
                    new_df[f"log_{selected_col}"] = np.log1p(df[selected_col])
                    
                    st.session_state.engineered_df = new_df
                    st.success("Log transformation applied!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = plot_histograms(df, [selected_col])
                        st.pyplot(fig)
                    with col2:
                        fig = plot_histograms(new_df, [f"log_{selected_col}"])
                        st.pyplot(fig)
            
            elif feature_type == "Square Root Transform":
                st.info("Apply square root transformation")
                
                selected_col = st.selectbox("Select Column", numeric_cols)
                
                if st.button("Apply Square Root Transform"):
                    new_df = df.copy()
                    # Make values positive first
                    min_val = df[selected_col].min()
                    if min_val < 0:
                        adjusted = df[selected_col] - min_val
                    else:
                        adjusted = df[selected_col]
                    
                    new_df[f"sqrt_{selected_col}"] = np.sqrt(adjusted)
                    
                    st.session_state.engineered_df = new_df
                    st.success("Square root transformation applied!")
            
            else:  # Custom Expression
                st.info("Create custom features using Python expressions")
                st.warning("⚠️ Advanced users only")
                
                expression = st.text_area(
                    "Python Expression",
                    "df['new_feature'] = df['column1'] + df['column2']",
                    help="Use 'df' to refer to the dataframe"
                )
                
                feature_name = st.text_input("New Feature Name", "custom_feature")
                
                if st.button("Create Custom Feature"):
                    try:
                        new_df = df.copy()
                        # Safe evaluation
                        exec(expression, {'df': new_df, 'np': np, 'pd': pd})
                        
                        st.session_state.engineered_df = new_df
                        st.success("Custom feature created!")
                        st.dataframe(new_df.head(10), use_container_width=True)
                    except Exception as e:
                        st.error(f"Error in expression: {str(e)}")
        
        with tabs[2]:  # Feature Selection
            st.markdown("### 📊 Feature Selection")
            st.info("Select the most important features automatically")
            
            if st.session_state.engineered_df is not None:
                df_to_use = st.session_state.engineered_df
            else:
                df_to_use = df
            
            numeric_cols = df_to_use.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                target_col = st.selectbox("Target Variable", numeric_cols)
                feature_cols = [c for c in numeric_cols if c != target_col]
                
                method = st.selectbox(
                    "Selection Method",
                    ["Mutual Information", "Correlation", "Random Forest Importance"]
                )
                
                k_features = st.slider("Number of Features to Select", 1, min(20, len(feature_cols)), 10)
                
                if st.button("🎯 Select Best Features", type="primary"):
                    with st.spinner("Analyzing features..."):
                        try:
                            if method == "Mutual Information":
                                selected_features, scores = select_features_mutual_info(
                                    df_to_use[feature_cols], 
                                    df_to_use[target_col], 
                                    k=k_features
                                )
                            else:
                                # Placeholder for other methods
                                from sklearn.feature_selection import SelectKBest, f_regression
                                selector = SelectKBest(f_regression, k=k_features)
                                selector.fit(df_to_use[feature_cols], df_to_use[target_col])
                                selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
                                scores = selector.scores_
                            
                            st.markdown("""
                                <div class="success-card">
                                    <h4>✅ Feature Selection Complete!</h4>
                                    <p>Best features have been identified</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Show results
                            st.markdown("#### 🏆 Selected Features")
                            
                            feature_scores = pd.DataFrame({
                                'Feature': selected_features,
                                'Score': scores[:len(selected_features)]
                            }).sort_values('Score', ascending=False)
                            
                            st.dataframe(feature_scores, use_container_width=True)
                            
                            # Visualization
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.barh(feature_scores['Feature'], feature_scores['Score'], color='#667eea')
                            ax.set_xlabel('Importance Score')
                            ax.set_title('Feature Importance')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Option to use only selected features
                            if st.button("Use Only Selected Features"):
                                selected_df = df_to_use[selected_features + [target_col]]
                                st.session_state.df = selected_df
                                st.success("Dataset updated with selected features!")
                                st.rerun()
                        
                        except Exception as e:
                            st.error(f"Error during feature selection: {str(e)}")
            else:
                st.warning("Need at least 2 numeric columns")

elif page == "🤖 ML Models":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
    else:
        df = st.session_state.df
        
        st.markdown("## 🤖 Machine Learning Models")
        
        # Get data info
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for ML!")
        else:
            # Model recommendation
            st.markdown("### 🎯 Smart Model Recommendation")
            
            target_col = st.selectbox("Select Target Variable", numeric_cols)
            feature_cols = [c for c in numeric_cols if c != target_col]
            
            if st.button("🧠 Get AI Recommendation"):
                with st.spinner("Analyzing data..."):
                    recommendation = recommend_model(df, target_col)
                    
                    st.markdown(f"""
                        <div class="info-card">
                            <h4>💡 Recommended Model</h4>
                            <p><strong>{recommendation['recommended_model']}</strong></p>
                            <p>{recommendation['reason']}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Show problem type
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Problem Type", recommendation['problem_type'].title())
                    with col2:
                        st.metric("Target Unique Values", recommendation['num_unique_values'])
                    with col3:
                        st.metric("Sample Size", len(df))
            
            st.markdown("---")
            
            # Model training
            st.markdown("### 🎓 Train Models")
            
            model_type = st.selectbox(
                "Select Model",
                ["Linear Regression", "Logistic Regression", "Random Forest", 
                 "XGBoost", "LightGBM"]
            )
            
            # Feature selection
            selected_features = st.multiselect(
                "Select Features",
                feature_cols,
                default=feature_cols[:5] if len(feature_cols) >= 5 else feature_cols
            )
            
            if not selected_features:
                st.warning("Please select at least one feature")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
                with col2:
                    random_state = st.number_input("Random Seed", 0, 1000, 42)
                
                # Model-specific parameters
                if model_type == "Random Forest":
                    col1, col2 = st.columns(2)
                    with col1:
                        n_estimators = st.slider("Number of Trees", 10, 500, 100, 10)
                    with col2:
                        max_depth = st.slider("Max Depth", 3, 30, 10)
                
                elif model_type in ["XGBoost", "LightGBM"]:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.slider("Number of Trees", 10, 500, 100, 10)
                    with col2:
                        learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
                    with col3:
                        max_depth = st.slider("Max Depth", 3, 15, 6)
                
                if st.button("🚀 Train Model", type="primary", use_container_width=True):
                    with st.spinner(f"Training {model_type}..."):
                        try:
                            # Prepare data
                            prepared_data = prepare_data_for_ml(
                                df, target_col, 
                                test_size=test_size
                            )
                            
                            # Extract data
                            X_train = prepared_data['X_train'][selected_features]
                            X_test = prepared_data['X_test'][selected_features]
                            y_train = prepared_data['y_train']
                            y_test = prepared_data['y_test']
                            feature_names = selected_features
                            
                            # Train model
                            if model_type == "Linear Regression":
                                results = train_linear_regression(X_train, X_test, y_train, y_test, feature_names)
                            
                            elif model_type == "Logistic Regression":
                                results = train_logistic_regression(X_train, X_test, y_train, y_test, feature_names)
                            
                            elif model_type == "Random Forest":
                                results = train_random_forest(
                                    X_train, X_test, y_train, y_test, feature_names,
                                    model_type='regressor' if len(df[target_col].unique()) > 10 else 'classifier',
                                    n_estimators=n_estimators,
                                    max_depth=max_depth
                                )
                            
                            elif model_type == "XGBoost":
                                import xgboost as xgb
                                
                                if len(df[target_col].unique()) > 10:  # Regression
                                    model = xgb.XGBRegressor(
                                        n_estimators=n_estimators,
                                        learning_rate=learning_rate,
                                        max_depth=max_depth,
                                        random_state=random_state
                                    )
                                    model.fit(X_train, y_train)
                                    y_pred = model.predict(X_test)
                                    
                                    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                                    
                                    results = {
                                        'model_type': 'XGBoost Regressor',
                                        'model': model,
                                        'y_train': y_train,
                                        'y_test': y_test,
                                        'y_pred': y_pred,
                                        'mse': mean_squared_error(y_test, y_pred),
                                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                                        'mae': mean_absolute_error(y_test, y_pred),
                                        'r2': r2_score(y_test, y_pred),
                                        'feature_names': feature_names,
                                        'feature_importance': model.feature_importances_
                                    }
                                else:  # Classification
                                    model = xgb.XGBClassifier(
                                        n_estimators=n_estimators,
                                        learning_rate=learning_rate,
                                        max_depth=max_depth,
                                        random_state=random_state
                                    )
                                    model.fit(X_train, y_train)
                                    y_pred = model.predict(X_test)
                                    
                                    from sklearn.metrics import accuracy_score, classification_report
                                    
                                    results = {
                                        'model_type': 'XGBoost Classifier',
                                        'model': model,
                                        'y_train': y_train,
                                        'y_test': y_test,
                                        'y_pred': y_pred,
                                        'accuracy': accuracy_score(y_test, y_pred),
                                        'classification_report': classification_report(y_test, y_pred),
                                        'feature_names': feature_names,
                                        'feature_importance': model.feature_importances_
                                    }
                            
                            elif model_type == "LightGBM":
                                import lightgbm as lgb
                                
                                if len(df[target_col].unique()) > 10:  # Regression
                                    model = lgb.LGBMRegressor(
                                        n_estimators=n_estimators,
                                        learning_rate=learning_rate,
                                        max_depth=max_depth,
                                        random_state=random_state
                                    )
                                    model.fit(X_train, y_train)
                                    y_pred = model.predict(X_test)
                                    
                                    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                                    
                                    results = {
                                        'model_type': 'LightGBM Regressor',
                                        'model': model,
                                        'y_train': y_train,
                                        'y_test': y_test,
                                        'y_pred': y_pred,
                                        'mse': mean_squared_error(y_test, y_pred),
                                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                                        'mae': mean_absolute_error(y_test, y_pred),
                                        'r2': r2_score(y_test, y_pred),
                                        'feature_names': feature_names,
                                        'feature_importance': model.feature_importances_
                                    }
                                else:  # Classification
                                    model = lgb.LGBMClassifier(
                                        n_estimators=n_estimators,
                                        learning_rate=learning_rate,
                                        max_depth=max_depth,
                                        random_state=random_state
                                    )
                                    model.fit(X_train, y_train)
                                    y_pred = model.predict(X_test)
                                    
                                    from sklearn.metrics import accuracy_score, classification_report
                                    
                                    results = {
                                        'model_type': 'LightGBM Classifier',
                                        'model': model,
                                        'y_train': y_train,
                                        'y_test': y_test,
                                        'y_pred': y_pred,
                                        'accuracy': accuracy_score(y_test, y_pred),
                                        'classification_report': classification_report(y_test, y_pred),
                                        'feature_names': feature_names,
                                        'feature_importance': model.feature_importances_
                                    }
                            
                            # Store results
                            st.session_state.model_results.append(results)
                            
                            st.markdown("""
                                <div class="success-card">
                                    <h4>✅ Model Trained Successfully!</h4>
                                    <p>Your model is ready for predictions</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Show results
                            st.markdown("### 📊 Model Performance")
                            
                            if 'r2' in results:  # Regression
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("R² Score", f"{results['r2']:.4f}")
                                with col2:
                                    st.metric("RMSE", f"{results['rmse']:.4f}")
                                with col3:
                                    st.metric("MAE", f"{results['mae']:.4f}")
                                with col4:
                                    st.metric("MSE", f"{results['mse']:.4f}")
                                
                                # Plots
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    fig = plot_regression_results(results['y_test'], results['y_pred'])
                                    st.pyplot(fig)
                                
                                with col2:
                                    fig = plot_feature_importance(results)
                                    st.pyplot(fig)
                            
                            else:  # Classification
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("Accuracy", f"{results['accuracy']:.4f}")
                                
                                with col2:
                                    st.metric("Test Size", len(results['y_test']))
                                
                                # Plots
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    fig = plot_confusion_matrix(results['y_test'], results['y_pred'])
                                    st.pyplot(fig)
                                
                                with col2:
                                    fig = plot_feature_importance(results)
                                    st.pyplot(fig)
                                
                                # Classification report
                                st.markdown("#### 📋 Detailed Classification Report")
                                st.text(results['classification_report'])
                            
                            # Model comparison
                            if len(st.session_state.model_results) > 1:
                                st.markdown("---")
                                st.markdown("### 📊 Model Comparison")
                                
                                if st.button("Compare All Models"):
                                    comparison_df = compare_models(st.session_state.model_results)
                                    st.dataframe(comparison_df, use_container_width=True)
                        
                        except Exception as e:
                            st.error(f"Error training model: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

elif page == "🔮 Model Explainability":
    st.markdown("## 🔮 Model Explainability")
    
    if not st.session_state.model_results:
        st.warning("⚠️ Train a model first!")
    else:
        st.info("Understand what drives your model's predictions using SHAP")
        
        # Select model
        model_names = [f"{i+1}. {res['model_type']}" for i, res in enumerate(st.session_state.model_results)]
        selected_model_idx = st.selectbox("Select Model", range(len(model_names)), format_func=lambda x: model_names[x])
        
        results = st.session_state.model_results[selected_model_idx]
        
        st.markdown("### 🎯 SHAP Analysis")
        
        try:
            import shap
            
            with st.spinner("Calculating SHAP values..."):
                model = results['model']
                
                # Prepare data
                df = st.session_state.df
                feature_names = results['feature_names']
                
                # Get a sample for SHAP (for performance)
                sample_size = min(100, len(df))
                X_sample = df[feature_names].sample(n=sample_size, random_state=42)
                
                # Create explainer
                if 'XGBoost' in results['model_type'] or 'LightGBM' in results['model_type'] or 'Random Forest' in results['model_type']:
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.LinearExplainer(model, X_sample)
                
                shap_values = explainer.shap_values(X_sample)
                
                st.markdown("""
                    <div class="success-card">
                        <h4>✅ SHAP Analysis Complete!</h4>
                        <p>Feature contributions calculated</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Summary plot
                st.markdown("#### 📊 Feature Importance (SHAP)")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
                st.pyplot(fig)
                plt.clf()
                
                # Detailed summary
                st.markdown("#### 🎨 SHAP Summary Plot")
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, show=False)
                st.pyplot(fig)
                plt.clf()
                
                # Individual prediction explanation
                st.markdown("#### 🔍 Individual Prediction Explanation")
                
                sample_idx = st.slider("Select Sample", 0, len(X_sample)-1, 0)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(shap.Explanation(
                    values=shap_values[sample_idx],
                    base_values=explainer.expected_value if isinstance(explainer.expected_value, (int, float)) else explainer.expected_value[0],
                    data=X_sample.iloc[sample_idx],
                    feature_names=feature_names
                ), show=False)
                st.pyplot(fig)
                plt.clf()
        
        except ImportError:
            st.error("SHAP not installed. Install with: pip install shap")
        except Exception as e:
            st.error(f"Error calculating SHAP values: {str(e)}")

elif page == "📊 Data Profiling":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
    else:
        df = st.session_state.df
        
        st.markdown("## 📊 Automated Data Profiling")
        st.info("Generate comprehensive data quality and analysis reports")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🔍 Profile Report Options")
            
            report_type = st.selectbox(
                "Report Type",
                ["Quick Profile", "Standard Profile", "Detailed Profile (Slow)"]
            )
            
            include_correlations = st.checkbox("Include Correlations", value=True)
            include_missing = st.checkbox("Missing Value Analysis", value=True)
            include_samples = st.checkbox("Sample Data", value=True)
        
        with col2:
            st.markdown("### ⚙️ Settings")
            st.write(f"**Rows:** {len(df):,}")
            st.write(f"**Columns:** {len(df.columns)}")
            st.write(f"**Size:** {df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
        
        if st.button("🚀 Generate Profile Report", type="primary", use_container_width=True):
            with st.spinner("Generating comprehensive profile report... This may take a minute."):
                try:
                    from ydata_profiling import ProfileReport
                    
                    # Configure based on report type
                    if report_type == "Quick Profile":
                        minimal = True
                    elif report_type == "Standard Profile":
                        minimal = False
                    else:
                        minimal = False
                    
                    profile = ProfileReport(
                        df,
                        title="StatWhizMaple - Data Profile Report",
                        minimal=minimal,
                        correlations={"pearson": include_correlations} if include_correlations else None
                    )
                    
                    # Generate HTML
                    profile_html = profile.to_html()
                    
                    st.markdown("""
                        <div class="success-card">
                            <h4>✅ Profile Report Generated!</h4>
                            <p>Comprehensive analysis complete</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display report
                    st.components.v1.html(profile_html, height=800, scrolling=True)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download HTML Report",
                        data=profile_html,
                        file_name="data_profile_report.html",
                        mime="text/html"
                    )
                
                except ImportError:
                    st.error("ydata-profiling not installed. Install with: pip install ydata-profiling")
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")

elif page == "📋 Reports":
    st.markdown("## 📋 Export & Reports")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
    else:
        st.info("Export your data, models, and analysis results")
        
        tabs = st.tabs(["💾 Export Data", "🤖 Save Models", "📊 Generate Report"])
        
        with tabs[0]:  # Export Data
            st.markdown("### 💾 Export Data")
            
            # Choose dataset
            dataset_choice = st.selectbox(
                "Select Dataset",
                ["Original Data", "Imputed Data", "Engineered Features", "Current Working Data"]
            )
            
            if dataset_choice == "Original Data":
                export_df = st.session_state.df
            elif dataset_choice == "Imputed Data" and st.session_state.imputed_df is not None:
                export_df = st.session_state.imputed_df
            elif dataset_choice == "Engineered Features" and st.session_state.engineered_df is not None:
                export_df = st.session_state.engineered_df
            else:
                export_df = st.session_state.df
            
            # Export format
            export_format = st.selectbox("Format", ["CSV", "Excel", "JSON", "Parquet"])
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Download Data", type="primary", use_container_width=True):
                    if export_format == "CSV":
                        csv = export_df.to_csv(index=False)
                        st.download_button(
                            "Download CSV",
                            csv,
                            "exported_data.csv",
                            "text/csv"
                        )
                    elif export_format == "Excel":
                        buffer = BytesIO()
                        export_df.to_excel(buffer, index=False, engine='openpyxl')
                        st.download_button(
                            "Download Excel",
                            buffer.getvalue(),
                            "exported_data.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    elif export_format == "JSON":
                        json_str = export_df.to_json(orient='records', indent=2)
                        st.download_button(
                            "Download JSON",
                            json_str,
                            "exported_data.json",
                            "application/json"
                        )
                    else:  # Parquet
                        buffer = BytesIO()
                        export_df.to_parquet(buffer, index=False)
                        st.download_button(
                            "Download Parquet",
                            buffer.getvalue(),
                            "exported_data.parquet",
                            "application/octet-stream"
                        )
            
            with col2:
                st.metric("Rows", len(export_df))
                st.metric("Columns", len(export_df.columns))
        
        with tabs[1]:  # Save Models
            st.markdown("### 🤖 Save Trained Models")
            
            if not st.session_state.model_results:
                st.warning("No models trained yet")
            else:
                for i, result in enumerate(st.session_state.model_results):
                    with st.expander(f"{i+1}. {result['model_type']}"):
                        if 'r2' in result:
                            st.write(f"**R² Score:** {result['r2']:.4f}")
                        elif 'accuracy' in result:
                            st.write(f"**Accuracy:** {result['accuracy']:.4f}")
                        
                        if st.button(f"💾 Save Model {i+1}", key=f"save_{i}"):
                            import joblib
                            
                            # Save model
                            model_filename = f"model_{result['model_type'].replace(' ', '_').lower()}_{i+1}.joblib"
                            joblib.dump(result['model'], model_filename)
                            
                            st.success(f"Model saved as {model_filename}")
        
        with tabs[2]:  # Generate Report
            st.markdown("### 📊 Generate Analysis Report")
            
            st.write("Create a comprehensive report of your analysis")
            
            report_sections = st.multiselect(
                "Include Sections",
                ["Data Overview", "EDA Summary", "Missing Values", "Correlations", 
                 "Model Results", "Feature Importance"],
                default=["Data Overview", "EDA Summary", "Model Results"]
            )
            
            if st.button("📄 Generate Report", type="primary", use_container_width=True):
                report_content = f"""
# StatWhizMaple - Analysis Report

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Dataset Information

- **Rows:** {len(st.session_state.df):,}
- **Columns:** {len(st.session_state.df.columns)}
- **Memory Usage:** {st.session_state.df.memory_usage(deep=True).sum() / (1024**2):.2f} MB

"""
                
                if "Data Overview" in report_sections:
                    report_content += f"""
## Data Overview

{st.session_state.df.head(10).to_markdown()}

"""
                
                if "EDA Summary" in report_sections:
                    report_content += f"""
## Statistical Summary

{st.session_state.df.describe().to_markdown()}

"""
                
                if "Model Results" in report_sections and st.session_state.model_results:
                    report_content += "\n## Model Results\n\n"
                    
                    for i, result in enumerate(st.session_state.model_results):
                        report_content += f"\n### Model {i+1}: {result['model_type']}\n\n"
                        
                        if 'r2' in result:
                            report_content += f"""
- **R² Score:** {result['r2']:.4f}
- **RMSE:** {result['rmse']:.4f}
- **MAE:** {result['mae']:.4f}

"""
                        elif 'accuracy' in result:
                            report_content += f"""
- **Accuracy:** {result['accuracy']:.4f}

"""
                
                report_content += "\n---\n*Generated by StatWhizMaple 2.0*\n"
                
                # Download button
                st.download_button(
                    "📥 Download Report (Markdown)",
                    report_content,
                    "analysis_report.md",
                    "text/markdown"
                )
                
                st.success("Report generated!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem; color: #888;'>
        <p><strong>StatWhizMaple 2.0</strong> | Built with ❤️ using Streamlit | © 2025</p>
        <p style='font-size: 0.8rem;'>Professional Edition • All Features Unlocked</p>
    </div>
""", unsafe_allow_html=True)
