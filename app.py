import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import folium
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier # Added import
from streamlit_folium import folium_static
import google.generativeai as genai
import io
import base64
import re
from datetime import datetime, timedelta
import time # Import the time module
import warnings # Import locally to keep dependencies clear
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from mlxtend.frequent_patterns import apriori, association_rules # For Market Basket
from mlxtend.preprocessing import TransactionEncoder # For Market Basket
import nltk # For Sentiment Analysis
import networkx as nx # For Network Analysis
from wordcloud import WordCloud # For Text Profiler
from scipy.stats import norm, lognorm, expon, weibull_min # For Distribution Fitting
import matplotlib.cm as cm # For Distribution Fitting plot colors
from sklearn.preprocessing import LabelEncoder # For Decision Tree target encoding
from nltk.sentiment.vader import SentimentIntensityAnalyzer # For Sentiment Analysis
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(layout="wide", page_title="Advanced Dashboard Creator", page_icon="📊")

with st.sidebar:
    st.image("d2.jpg", caption="From rows to revelations.", use_container_width=True)


# Custom CSS for better styling
st.markdown("""
<style>
/* General App Style for Dark Theme */
body {
    color: #E0E0E0; /* Lighter text for dark backgrounds */
}

.metric-card {
    background: linear-gradient(135deg, #2C5364 0%, #203A43 50%, #0F2027 100%); /* Dark blue/grey gradient */
    padding: 25px;
    border-radius: 12px;
    color: #FFFFFF;
    text-align: center;
    margin: 10px 0;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    border: 1px solid #4A5568; /* Subtle border */
}
.metric-card h3 {
    font-size: 2.5em; /* Larger metric numbers */
    margin-bottom: 5px;
    font-weight: 700;
}
.metric-card p {
    font-size: 1em;
    font-weight: 300;
}

.insight-box {
    background: #2D3748; /* Darker background for insight box */
    padding: 20px;
    border-left: 5px solid #38B2AC; /* Teal accent border */
    border-radius: 8px;
    margin: 15px 0;
    color: #E2E8F0; /* Lighter text */
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
}
.insight-box strong {
    color: #63B3ED; /* Brighter color for strong text */
}

/* Styling for expander headers */
div[data-testid="stExpander"] > div:first-child {
    background-color: #2D3748; /* Darker background for expander header */
    padding: 10px 15px !important;
    border-radius: 8px 8px 0 0;
    border-bottom: 1px solid #4A5568;
}
div[data-testid="stExpander"] > div:first-child summary {
    color: #A0AEC0; /* Lighter text for expander title */
    font-weight: 600;
    font-size: 1.1em;
}
div[data-testid="stExpander"] > div:first-child summary:hover {
    color: #E2E8F0;
}

/* General Button Styling */
.stButton > button {
    border: 1px solid #38B2AC;
    background-color: transparent;
    color: #38B2AC;
    padding: 10px 20px;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background-color: #38B2AC;
    color: #1A202C;
    box-shadow: 0 2px 10px rgba(56, 178, 172, 0.4);
}
.stButton > button:active {
    background-color: #2C7A7B !important;
    color: #1A202C !important;
}

/* Styling for input widgets */
div[data-testid="stSelectbox"] label, div[data-testid="stMultiSelect"] label,
div[data-testid="stTextInput"] label, div[data-testid="stSlider"] label,
div[data-testid="stColorPicker"] label, div[data-testid="stNumberInput"] label {
    font-weight: 500;
    color: #CBD5E0; /* Lighter label text for dark theme */
}

/* CSS for Column Summary Cards */
.column-summary-card {
    background: #2D3748; /* Darker background, consistent with insight-box */
    padding: 15px;
    border-radius: 8px;
    color: #E2E8F0; /* Lighter text */
    margin-bottom: 15px; /* Space between cards */
    border: 1px solid #4A5568; /* Subtle border */
    height: 100%; /* Ensure cards in a row have same height if content varies slightly */
}
.column-summary-card h4 {
    font-size: 1.1em;
    color: #63B3ED; /* Brighter color for column name */
    margin-bottom: 8px;
    border-bottom: 1px solid #4A5568;
    padding-bottom: 5px;
    white-space: nowrap; /* Prevent long names from wrapping too much */
    overflow: hidden;
    text-overflow: ellipsis;
}
.column-summary-card p {
    font-size: 0.9em;
    margin-bottom: 4px;
}
.column-summary-card p strong {
    font-weight: 500;
    color: #A0AEC0; /* Lighter grey for labels */
}
</style>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("🛠️ Dashboard Controls & Options")
selected_theme = st.sidebar.selectbox("🎨 Theme Selection", options=['light', 'dark', 'cyberpunk'], index=1) # Default to dark
color_scheme = st.sidebar.selectbox("🌈 Chart Color Palette", options=['viridis', 'plasma', 'inferno', 'magma', 'cividis'], index=1) # Plasma default
custom_color = st.sidebar.color_picker("🎨 Custom Accent Color", "#38B2AC") # Teal default

# NEW FEATURE 1: Real-time data refresh
refresh_interval = st.sidebar.slider("⏱️ Auto-Refresh Interval (s)", 0, 60, 0)
if refresh_interval > 0:
    st.sidebar.info(f"Dashboard will refresh every {refresh_interval} seconds")

# Gemini API Integration
st.sidebar.header("🤖 AI-Powered Assistance")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)
    except Exception as e:
        st.sidebar.error(f"API Error: {str(e)}")

# Main title with metrics
st.title("Advanced Data Explorer & Visualizer")
st.image("d3.jpg", caption="Made for Analysts. Loved by Scientists.", use_container_width=True)
st.markdown("### 🔮 Upload your data to unlock insights and visualizations!")

# NEW FEATURE 2: Multiple file upload support
uploaded_files = st.file_uploader("Choose files", type=["csv", "xlsx", "json"], accept_multiple_files=True)

# NEW FEATURE 3: Data comparison mode
comparison_mode = st.checkbox("📊 Enable Data Comparison Mode")

if uploaded_files:
    # Process multiple files
    datasets = {}
    for i, file in enumerate(uploaded_files):
        try:
            if file.name.endswith('.csv'):
                datasets[f"Dataset_{i+1}"] = pd.read_csv(file)
            elif file.name.endswith('.xlsx'):
                datasets[f"Dataset_{i+1}"] = pd.read_excel(file)
            elif file.name.endswith('.json'):
                datasets[f"Dataset_{i+1}"] = pd.read_json(file)
        except Exception as e:
            st.error(f"Error reading {file.name}: {str(e)}")
    
    if datasets:
        # Dataset selector
        selected_dataset = st.selectbox("Select Primary Dataset", options=list(datasets.keys()))

        # Define the callback function for resetting model parameters
        # This function will be called when the reset button is clicked.
        def reset_all_model_parameters_callback():
            # Reset DT parameters to their initial defaults
            # Fallback values in .get() are for extreme safety, 
            # assuming INIT_DEFAULT keys are always set before button click.
            st.session_state.dt_max_depth = st.session_state.get("INIT_DEFAULT_dt_max_depth", 5)
            st.session_state.dt_min_samples_split = st.session_state.get("INIT_DEFAULT_dt_min_samples_split", 2)
            st.session_state.dt_min_samples_leaf = st.session_state.get("INIT_DEFAULT_dt_min_samples_leaf", 1)
            st.session_state.dt_criterion = st.session_state.get("INIT_DEFAULT_dt_criterion", "gini")
            
            # Reset RF parameters to their initial defaults
            st.session_state.rf_n_estimators = st.session_state.get("INIT_DEFAULT_rf_n_estimators", 100)
            st.session_state.rf_max_depth = st.session_state.get("INIT_DEFAULT_rf_max_depth", 10)
            st.session_state.rf_min_samples_split = st.session_state.get("INIT_DEFAULT_rf_min_samples_split", 2)
            st.session_state.rf_min_samples_leaf = st.session_state.get("INIT_DEFAULT_rf_min_samples_leaf", 1)
            st.session_state.rf_criterion = st.session_state.get("INIT_DEFAULT_rf_criterion", "gini")

        df = datasets[selected_dataset]
        
        # NEW FEATURE: Automatic Data Overview Table (before Advanced Profiling)
        with st.expander("📄 Quick Data Overview", expanded=True):
            st.subheader(f"Overview of: {selected_dataset}")
            
            st.markdown("#### First 5 Rows:")
            st.dataframe(df.head())
            
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.markdown("#### Data Dimensions:")
                st.write(f"Rows: {df.shape[0]:,}")
                st.write(f"Columns: {df.shape[1]}")
            with col_info2:
                st.markdown("#### Missing Values (per column):")
                st.dataframe(df.isnull().sum().rename("Missing Count").to_frame().T)
            st.markdown("#### Column Data Types:")
            st.dataframe(df.dtypes.rename("Data Type").to_frame().T)
            
        # NEW FEATURE 4: Advanced data profiling
        with st.expander("📊 Advanced Data Profiling", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="metric-card"><h3>{df.shape[0]:,}</h3><p>Total Rows</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><h3>{df.shape[1]}</h3><p>Columns</p></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card"><h3>{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB</h3><p>Memory Usage</p></div>', unsafe_allow_html=True)
            with col4:
                duplicates = df.duplicated().sum()
                st.markdown(f'<div class="metric-card"><h3>{duplicates}</h3><p>Duplicates</p></div>', unsafe_allow_html=True)
            
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            quality_score = max(0, 100 - missing_pct - (duplicates/df.shape[0]*10))
            st.markdown(f'<div class="insight-box"><strong>Data Quality Score: {quality_score:.1f}/100</strong><br>Based on missing values and duplicate records</div>', unsafe_allow_html=True)
        
        # NEW FEATURE 5: Smart data type detection and conversion
        with st.expander("💡 Smart Data Type Detection & Conversion"):
            st.subheader("Suggested Data Type Conversions")
            suggestions = []
            
            for col in df.columns:
                if df[col].dtype == 'object': # Check object columns for potential conversion
                    # Check if it's actually numeric
                    try:
                        pd.to_numeric(df[col].dropna())
                        suggestions.append((col, "numeric", "Contains numeric values"))
                    except:
                        # Check if it's a date
                        try:
                            pd.to_datetime(df[col].dropna().head(100))
                            suggestions.append((col, "datetime", "Contains date patterns"))
                        except:
                            pass
            
            if suggestions:
                for col, suggested_type, reason in suggestions:
                    if st.checkbox(f"Convert '{col}' to {suggested_type} ({reason})", key=f"convert_{col}_{suggested_type}"):
                        try:
                            if suggested_type == "numeric":
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            elif suggested_type == "datetime":
                                df[col] = pd.to_datetime(df[col], errors='coerce')
                            st.success(f"Converted {col} to {suggested_type}")
                        except Exception as e:
                            st.error(f"Conversion failed: {str(e)}")
            else:
                st.info("No conversion suggestions found")

        # Globally define/update column type lists after potential conversions in Feature 5
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include='datetime').columns.tolist()

        # NEW FEATURE 6: Advanced filtering system
        with st.expander("⚙️ Advanced Data Filtering System"):
            st.subheader("Multi-Column Filters")
            
            # Numeric filters
            if numeric_cols:
                st.write("**Numeric Filters:**")
                for col in numeric_cols[:3]:  # Limit to 3 for space
                    if not df[col].isna().all():
                        min_val, max_val = float(df[col].min()), float(df[col].max())
                        filter_range = st.slider(f"{col} Range", min_val, max_val, (min_val, max_val), key=f"filter_{col}")
                        df = df[(df[col] >= filter_range[0]) & (df[col] <= filter_range[1])]
            
            # Categorical filters
            if categorical_cols:
                st.write("**Categorical Filters:**")
                for col in categorical_cols[:2]:  # Limit to 2 for space
                    unique_values = df[col].dropna().unique()
                    if len(unique_values) <= 20:  # Only show if manageable number of categories
                        selected_values = st.multiselect(f"Filter {col}", unique_values, default=unique_values, key=f"cat_filter_{col}")
                        if selected_values:
                            df = df[df[col].isin(selected_values)]
            
            st.success(f"Filtered dataset: {df.shape[0]} rows × {df.shape[1]} columns")

            if not df.empty:
                st.markdown("#### Filtered Data Preview (First 100 rows):")
                st.dataframe(df.head(100)) # Show a preview

                # Download button for the filtered data
                @st.cache_data # Cache the conversion to prevent re-computation on every rerun
                def convert_df_to_csv(df_to_convert):
                    return df_to_convert.to_csv(index=False).encode('utf-8')

                csv_filtered = convert_df_to_csv(df)
                st.download_button(
                    label="📥 Download Filtered Data as CSV",
                    data=csv_filtered,
                    file_name=f"filtered_{selected_dataset.lower().replace('dataset_','')}.csv",
                    mime="text/csv",
                )
            else:
                st.warning("The current filter selection results in an empty dataset.")

        # NEW FEATURE 7: Automated statistical testing
        with st.expander("📈 Statistical Analysis Suite"):
            if len(numeric_cols) >= 2:
                st.subheader("Correlation Analysis")
                corr_method = st.selectbox("Correlation Method", ["pearson", "spearman", "kendall"])
                corr_matrix = df[numeric_cols].corr(method=corr_method)
                
                # Heatmap
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                               title=f"Correlation Matrix ({corr_method})")
                st.plotly_chart(fig, use_container_width=True)
                
                # Strongest correlations
                corr_pairs = []
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        corr_val = corr_matrix.iloc[i, j]
                        if not np.isnan(corr_val):
                            corr_pairs.append((numeric_cols[i], numeric_cols[j], corr_val))
                
                corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                
                if corr_pairs:
                    st.write("**Top Correlations:**")
                    for col1, col2, corr_val in corr_pairs[:3]:
                        strength = "Strong" if abs(corr_val) > 0.7 else "Moderate" if abs(corr_val) > 0.3 else "Weak"
                        direction = "positive" if corr_val > 0 else "negative"
                        st.write(f"• {strength} {direction} correlation: **{col1}** ↔ **{col2}** ({corr_val:.3f})")

        # NEW FEATURE 8: Machine Learning Pipeline
        with st.expander("🧠 AutoML Predictive Pipeline"):
            if len(numeric_cols) >= 2:
                st.subheader("Automated Model Building")
                
                # Target selection
                target_col = st.selectbox("Select Target Variable", numeric_cols)
                feature_cols = [col for col in numeric_cols if col != target_col]
                
                if feature_cols:
                    # Prepare data
                    X = df[feature_cols].dropna()
                    y = df.loc[X.index, target_col]
                    
                    if len(X) > 10:  # Minimum data requirement
                        # Train-test split
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        # Polynomial degree selection for Polynomial Regression
                        poly_degree = st.number_input("Polynomial Degree (for Polynomial Regression)", min_value=2, max_value=5, value=2, step=1, key="automl_poly_degree")
                        
                        # Model selection
                        models = {
                            "Linear Regression": LinearRegression(),
                            "Polynomial Regression": Pipeline([
                                ("poly_features", PolynomialFeatures(degree=poly_degree, include_bias=False)),
                                ("lin_reg", LinearRegression())
                            ]),
                            "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42)
                        }
                        
                        results = {}
                        for name, model in models.items():
                            try:
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)
                                r2 = r2_score(y_test, y_pred)
                                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                                results[name] = {"R²": r2, "RMSE": rmse, "Model": model}
                            except Exception as e:
                                st.error(f"Error training {name}: {str(e)}")
                        
                        if results:
                            # Display results
                            col1, col2 = st.columns(2)
                            for i, (name, metrics) in enumerate(results.items()):
                                with col1 if i % 2 == 0 else col2:
                                    st.metric(f"{name} R²", f"{metrics['R²']:.3f}")
                                    st.metric(f"{name} RMSE", f"{metrics['RMSE']:.3f}")
                            
                            # Feature importance (for Random Forest)
                            if "Random Forest" in results:
                                rf_model = results["Random Forest"]["Model"]
                                importance_df = pd.DataFrame({
                                    'Feature': feature_cols,
                                    'Importance': rf_model.feature_importances_
                                }).sort_values('Importance', ascending=False)
                                
                                fig = px.bar(importance_df, x='Importance', y='Feature', 
                                           orientation='h', title="Feature Importance")
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Need more data points for modeling")
                else:
                    st.warning("Need multiple numeric columns for modeling")

        # NEW FEATURE 9: Data comparison dashboard
        if comparison_mode and len(datasets) > 1:
            with st.expander("🆚 Dataset Comparison Dashboard"):
                st.subheader("Compare Multiple Datasets")
                
                compare_datasets = st.multiselect("Select datasets to compare", 
                                                list(datasets.keys()), 
                                                default=list(datasets.keys())[:2])
                
                if len(compare_datasets) >= 2:
                    comparison_data = []
                    for name in compare_datasets:
                        ds = datasets[name]
                        comparison_data.append({
                            'Dataset': name,
                            'Rows': ds.shape[0],
                            'Columns': ds.shape[1],
                            'Numeric Cols': len(ds.select_dtypes(include=['number']).columns),
                            'Missing %': (ds.isnull().sum().sum() / (ds.shape[0] * ds.shape[1]) * 100)
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df)
                    
                    # Visual comparison
                    fig = px.bar(comparison_df, x='Dataset', y='Rows', title="Dataset Size Comparison")
                    st.plotly_chart(fig, use_container_width=True)

        # NEW FEATURE 10: Export and reporting system
        with st.expander("📄 Advanced Export & Reporting System"):
            st.subheader("Generate Analysis Report")
            
            report_sections = st.multiselect("Include in Report", 
                                           ["Data Summary", "Correlation Analysis", "Missing Data Report", 
                                            "Statistical Summary", "Data Quality Assessment"],
                                           default=["Data Summary", "Data Quality Assessment"])
            
            if st.button("Generate Report"):
                report = f"""
# Data Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview
- **Rows**: {df.shape[0]:,}
- **Columns**: {df.shape[1]}
- **Memory Usage**: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB

"""
                
                if "Data Quality Assessment" in report_sections:
                    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                    duplicates = df.duplicated().sum()
                    quality_score = max(0, 100 - missing_pct - (duplicates/df.shape[0]*10))
                    
                    report += f"""
## Data Quality Assessment
- **Quality Score**: {quality_score:.1f}/100
- **Missing Data**: {missing_pct:.1f}%
- **Duplicate Rows**: {duplicates}

"""
                
                if "Statistical Summary" in report_sections and numeric_cols:
                    report += f"""
## Statistical Summary
{df[numeric_cols].describe().to_string()}

"""
                
                st.text_area("Generated Report", report, height=300)
                st.download_button("Download Report", report, file_name="analysis_report.txt")

        # AI-Powered Insights with Gemini API
        with st.expander("💬 AI-Powered Insights (Gemini)", expanded=True):
            st.subheader("Ask Questions About Your Data")
            
            if gemini_api_key:
                user_question = st.text_input("Ask a question about your data:", 
                                             placeholder="E.g., What are the key trends in this dataset?")
                
                if user_question:
                    # Prepare data summary
                    buffer = io.StringIO()
                    df.describe().to_csv(buffer)
                    data_summary = buffer.getvalue()
                    
                    # Prepare data sample
                    buffer = io.StringIO()
                    df.head(5).to_csv(buffer)
                    data_sample = buffer.getvalue()
                    
                    # Prepare column info
                    column_info = "\n".join([
                        f"- {col} ({df[col].dtype}): {df[col].nunique()} unique values" 
                        for col in df.columns
                    ])
                    
                    # Create prompt
                    prompt = f"""
                    I have a dataset with {df.shape[0]} rows and {df.shape[1]} columns.
                    
                    Column information:
                    {column_info}
                    
                    Data sample:
                    {data_sample}
                    
                    Summary statistics:
                    {data_summary}
                    
                    Question: {user_question}
                    
                    Please provide a helpful, concise, and accurate answer based on this data.
                    """
                    
                    try:
                        model = genai.GenerativeModel("gemini-2.0-flash")
                        response = model.generate_content(prompt)
                        st.write("AI Response:")
                        st.write(response.text)
                    except Exception as e:
                        st.error(f"Gemini API Error: {str(e)}")
            else:
                st.info("Enter your Gemini API key in the sidebar to enable AI insights.")

        # Interactive visualization builder
        with st.expander("🎨 Quick Visualization Builder"):
            if numeric_cols or categorical_cols:
                viz_type = st.selectbox("Chart Type", ["Scatter", "Line", "Bar", "Histogram", "Box"])
                
                if viz_type in ["Scatter", "Line"] and len(numeric_cols) >= 2:
                    x_col = st.selectbox("X-axis", numeric_cols)
                    y_col = st.selectbox("Y-axis", [col for col in numeric_cols if col != x_col])
                    
                    if viz_type == "Scatter":
                        fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                    else:
                        fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Histogram" and numeric_cols:
                    col = st.selectbox("Column", numeric_cols)
                    bins = st.slider("Bins", 10, 100, 30)
                    fig = px.histogram(df, x=col, nbins=bins, title=f"Distribution of {col}")
                    st.plotly_chart(fig, use_container_width=True)

        # NEW FEATURE 11: Anomaly Detection Dashboard
        with st.expander("🚨 Anomaly Detection Dashboard"):
            if numeric_cols:
                st.subheader("Interactive Outlier Detection")
                anomaly_col = st.selectbox("Select Column for Anomaly Detection", numeric_cols)
                method = st.selectbox("Detection Method", ["IQR", "Z-Score", "Isolation Forest"])
                
                if method == "IQR":
                    Q1, Q3 = df[anomaly_col].quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    anomalies = (df[anomaly_col] < lower_bound) | (df[anomaly_col] > upper_bound)
                elif method == "Z-Score":
                    z_scores = np.abs((df[anomaly_col] - df[anomaly_col].mean()) / df[anomaly_col].std())
                    anomalies = z_scores > 3
                else:  # Isolation Forest
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    anomalies = iso_forest.fit_predict(df[[anomaly_col]].dropna()) == -1
                
                anomaly_count = anomalies.sum() if hasattr(anomalies, 'sum') else len([x for x in anomalies if x])
                st.metric("Anomalies Found", anomaly_count)
                
                # Visualization
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=df[anomaly_col], mode='markers', name='Normal', 
                                       marker=dict(color='blue', size=4)))
                if anomaly_count > 0:
                    anomaly_data = df[anomalies] if hasattr(anomalies, 'sum') else df.iloc[np.where(anomalies)[0]]
                    fig.add_trace(go.Scatter(y=anomaly_data[anomaly_col], mode='markers', name='Anomalies',
                                           marker=dict(color='red', size=8)))
                fig.update_layout(title=f"Anomaly Detection: {anomaly_col}")
                st.plotly_chart(fig, use_container_width=True)

        # NEW FEATURE 12: Time Series Analysis
        # Uses globally defined date_cols, numeric_cols
        if date_cols and numeric_cols:
            with st.expander("⏳ Time Series Analysis & Trends"):
                st.subheader("Trend Analysis & Forecasting")
                date_col = st.selectbox("Date Column", date_cols, key="tsa_date_col")
                value_col = st.selectbox("Value Column", numeric_cols, key="tsa_value_col")
                
                # Prepare time series data
                ts_data = df[[date_col, value_col]].dropna().sort_values(date_col)
                ts_data = ts_data.set_index(date_col).resample('D')[value_col].mean().dropna()
                
                if len(ts_data) > 10:
                    # Basic trend analysis
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values, mode='lines+markers', name='Actual'))
                    
                    # Simple moving average
                    window = min(7, len(ts_data)//3)
                    ma = ts_data.rolling(window=window).mean()
                    fig.add_trace(go.Scatter(x=ma.index, y=ma.values, mode='lines', name=f'{window}-day MA'))
                    
                    fig.update_layout(title=f"Time Series: {value_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Trend", "Upward" if ts_data.iloc[-1] > ts_data.iloc[0] else "Downward")
                    with col2:
                        volatility = ts_data.std() / ts_data.mean() * 100
                        st.metric("Volatility", f"{volatility:.1f}%")

        # NEW FEATURE 13: Data Relationship Mapper
        with st.expander("🕸️ Data Relationship Mapper"):
            st.subheader("Column Relationship Network")
            if len(numeric_cols) >= 3:
                # Create correlation network
                corr_matrix = df[numeric_cols].corr().abs()
                
                # Find strong relationships (>0.5 correlation)
                relationships = []
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        corr_val = corr_matrix.iloc[i, j]
                        if corr_val > 0.5:
                            relationships.append((numeric_cols[i], numeric_cols[j], corr_val))
                
                if relationships:
                    st.write("**Strong Relationships Found:**")
                    for col1, col2, strength in sorted(relationships, key=lambda x: x[2], reverse=True)[:5]:
                        st.write(f"• {col1} ↔ {col2}: {strength:.3f}")
                    
                    # Create network visualization (simplified)
                    nodes = list(set([r[0] for r in relationships] + [r[1] for r in relationships]))
                    fig = go.Figure()
                    
                    # Add nodes
                    for i, node in enumerate(nodes):
                        fig.add_trace(go.Scatter(x=[i], y=[0], mode='markers+text', text=[node],
                                               textposition="middle center", marker=dict(size=20)))
                    
                    fig.update_layout(title="Data Relationship Network", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No strong relationships found (correlation > 0.5)")

        # NEW FEATURE 14: A/B Testing Suite
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            with st.expander("🧪 A/B Testing & ANOVA Suite", expanded=False): # Keep it collapsed by default
                st.subheader("Statistical Significance Testing")

                group_col = st.selectbox("Group Column (A/B)", categorical_cols)
                metric_col = st.selectbox("Metric Column", numeric_cols)

                if group_col and metric_col:
                    groups = df[group_col].unique()
                    if len(groups) == 2:
                        group_a_data = df[df[group_col] == groups[0]][metric_col].dropna()
                        group_b_data = df[df[group_col] == groups[1]][metric_col].dropna()

                        if len(group_a_data) > 5 and len(group_b_data) > 5:
                            # Basic stats
                            st.markdown("#### Group Statistics")
                            col_stats1, col_stats2 = st.columns(2)
                            with col_stats1:
                                st.metric(f"{groups[0]} Mean", f"{group_a_data.mean():.2f}")
                                st.metric(f"{groups[0]} Size", len(group_a_data))
                            with col_stats2:
                                st.metric(f"{groups[1]} Mean", f"{group_b_data.mean():.2f}")
                                st.metric(f"{groups[1]} Size", len(group_b_data))

                            # Statistical Test (T-test)
                            # from scipy import stats # Already imported globally or can be kept local if preferred

                            # Perform t-test
                            t_stat, p_value = stats.ttest_ind(group_a_data, group_b_data, equal_var=False) # Welch's t-test by default

                            # Effect size (Cohen's d)
                            # For Welch's t-test, pooled_std calculation is slightly different or can be approximated
                            # A common way to calculate Cohen's d with unequal variances:
                            mean_diff = group_a_data.mean() - group_b_data.mean()
                            pooled_std = np.sqrt((group_a_data.var() + group_b_data.var()) / 2) # Simpler approximation
                            if pooled_std == 0: # Avoid division by zero
                                cohens_d = np.nan
                            else:
                                cohens_d = mean_diff / pooled_std

                            # Statistical significance
                            alpha = st.slider("Significance Level (Alpha)", 0.01, 0.10, 0.05, 0.01)
                            is_significant = p_value < alpha

                            # Display results
                            st.markdown("#### Test Results")
                            res_col1, res_col2, res_col3 = st.columns(3)

                            with res_col1:
                                st.metric("T-statistic", f"{t_stat:.3f}")
                            with res_col2:
                                st.metric("P-value", f"{p_value:.4f}")
                            with res_col3:
                                st.metric("Cohen's d", f"{cohens_d:.3f}" if not np.isnan(cohens_d) else "N/A")

                            # Significance interpretation
                            if is_significant:
                                st.success(f"✅ **Statistically Significant** (p < {alpha})")
                                st.write(f"There is a significant difference in '{metric_col}' between '{groups[0]}' and '{groups[1]}'.")
                            else:
                                st.warning(f"❌ **Not Statistically Significant** (p ≥ {alpha})")
                                st.write(f"No significant difference found in '{metric_col}' between '{groups[0]}' and '{groups[1]}'.")

                            # Effect size interpretation
                            if not np.isnan(cohens_d):
                                if abs(cohens_d) < 0.2:
                                    effect_interpretation = "Small effect"
                                elif abs(cohens_d) < 0.5:
                                    effect_interpretation = "Medium effect"
                                else:
                                    effect_interpretation = "Large effect"
                                st.info(f"**Effect Size**: {effect_interpretation} (|d| = {abs(cohens_d):.3f})")
                            
                            # Confidence interval for difference in means
                            # Using stats.t.interval for more accuracy with t-distribution
                            diff_mean_val = group_a_data.mean() - group_b_data.mean()
                            se_diff = np.sqrt(group_a_data.var()/len(group_a_data) + group_b_data.var()/len(group_b_data))
                            dof = (se_diff**4) / ( ( (group_a_data.var()/len(group_a_data))**2 / (len(group_a_data)-1) ) + ( (group_b_data.var()/len(group_b_data))**2 / (len(group_b_data)-1) ) ) # Welch-Satterthwaite equation for DoF
                            
                            if se_diff > 0 and not np.isnan(dof) and dof > 0 : # Check for valid SE and DoF
                                ci = stats.t.interval(1-alpha, dof, loc=diff_mean_val, scale=se_diff)
                                st.write(f"**{((1-alpha)*100):.0f}% Confidence Interval for difference**: [{ci[0]:.3f}, {ci[1]:.3f}]")
                            else:
                                st.write("**Confidence Interval for difference**: Could not be calculated (likely due to zero variance or insufficient data).")

                            # Distribution comparison
                            fig_hist = go.Figure()
                            fig_hist.add_trace(go.Histogram(x=group_a_data, name=str(groups[0]), opacity=0.7, nbinsx=20, marker_color=custom_color))
                            fig_hist.add_trace(go.Histogram(x=group_b_data, name=str(groups[1]), opacity=0.7, nbinsx=20))
                            fig_hist.update_layout(title="Distribution Comparison", barmode='overlay', xaxis_title=metric_col, yaxis_title="Frequency")
                            st.plotly_chart(fig_hist, use_container_width=True)

                            # Box plot comparison
                            fig_box = go.Figure()
                            fig_box.add_trace(go.Box(y=group_a_data, name=str(groups[0]), marker_color=custom_color))
                            fig_box.add_trace(go.Box(y=group_b_data, name=str(groups[1])))
                            fig_box.update_layout(title="Box Plot Comparison", yaxis_title=metric_col)
                            st.plotly_chart(fig_box, use_container_width=True)

                        else:
                            st.warning("Need at least 6 samples in each group for reliable testing.")
                    elif len(groups) > 2:
                        st.subheader(f"ANOVA Test: Comparing {len(groups)} Groups for '{metric_col}'")
                        st.markdown(f"Since you have more than two groups in '{group_col}', an ANOVA (Analysis of Variance) test will be performed to check for significant differences in the means of '{metric_col}' across these groups.")

                        # Prepare data for ANOVA: list of arrays, one for each group
                        group_data_for_anova = [df[df[group_col] == group_name][metric_col].dropna() for group_name in groups]
                        
                        # Filter out groups with insufficient data for ANOVA (e.g., less than 2 samples)
                        group_data_for_anova_filtered = [g for g in group_data_for_anova if len(g) >= 2]

                        if len(group_data_for_anova_filtered) >= 2: # Need at least two groups for ANOVA
                            from scipy.stats import f_oneway # Specific import for ANOVA
                            f_stat, p_value_anova = f_oneway(*group_data_for_anova_filtered)

                            st.metric("ANOVA F-statistic", f"{f_stat:.3f}")
                            st.metric("ANOVA P-value", f"{p_value_anova:.4f}")

                            alpha_anova = st.slider("ANOVA Significance Level (Alpha)", 0.01, 0.10, 0.05, 0.01, key="anova_alpha")
                            if p_value_anova < alpha_anova:
                                st.success(f"✅ **Statistically Significant** (p < {alpha_anova}). There is a significant difference in '{metric_col}' means across the groups.")
                            else:
                                st.warning(f"❌ **Not Statistically Significant** (p ≥ {alpha_anova}). No significant difference found in '{metric_col}' means across the groups.")
                        else:
                            st.warning("ANOVA requires at least two groups with sufficient data (>=2 samples per group) for comparison.")
                    elif len(groups) < 2:
                        st.warning(f"The selected group column '{group_col}' has fewer than 2 distinct groups. A/B testing requires at least two groups to compare.")
                else:
                    st.info("Select a categorical group column and a numeric metric column to perform A/B testing.")
                    
        # NEW FEATURE 15: Geospatial Data Visualization
        with st.expander("🗺️ Geospatial Data Visualization"):
            st.subheader("Map Data Points")
            potential_lat_cols = [col for col in numeric_cols if 'lat' in col.lower()]
            potential_lon_cols = [col for col in numeric_cols if 'lon' in col.lower() or 'lng' in col.lower()]

            if not potential_lat_cols and not potential_lon_cols:
                 st.info("No obvious latitude/longitude columns found. Please select manually if available.")

            lat_col_default = potential_lat_cols[0] if potential_lat_cols else None
            lon_col_default = potential_lon_cols[0] if potential_lon_cols else None

            lat_col = st.selectbox("Select Latitude Column", numeric_cols, index=numeric_cols.index(lat_col_default) if lat_col_default and lat_col_default in numeric_cols else 0, key="geo_lat")
            lon_col = st.selectbox("Select Longitude Column", numeric_cols, index=numeric_cols.index(lon_col_default) if lon_col_default and lon_col_default in numeric_cols else (1 if len(numeric_cols) > 1 else 0), key="geo_lon")
            
            if lat_col and lon_col:
                map_df_viz = df[[lat_col, lon_col]].copy().dropna()
                map_df_viz.columns = ['lat', 'lon'] # st.map expects columns named 'lat' and 'lon'

                if not map_df_viz.empty and pd.api.types.is_numeric_dtype(map_df_viz['lat']) and pd.api.types.is_numeric_dtype(map_df_viz['lon']):
                    # Filter out invalid lat/lon values
                    map_df_viz = map_df_viz[(map_df_viz['lat'] >= -90) & (map_df_viz['lat'] <= 90)]
                    map_df_viz = map_df_viz[(map_df_viz['lon'] >= -180) & (map_df_viz['lon'] <= 180)]

                    if not map_df_viz.empty:
                        st.map(map_df_viz)

                        if st.checkbox("Show Advanced Folium Map (more options)", key="folium_map_cb"):
                            m = folium.Map(location=[map_df_viz['lat'].mean(), map_df_viz['lon'].mean()], zoom_start=5)
                            tooltip_col_options = [None] + categorical_cols + numeric_cols
                            tooltip_col = st.selectbox("Select Tooltip Column (Optional)", tooltip_col_options, key="folium_tooltip")
                            
                            # Iterate over original df to get tooltip and ensure correct lat/lon mapping
                            for idx, row in df.dropna(subset=[lat_col, lon_col]).iterrows():
                                popup_text = str(row[tooltip_col]) if tooltip_col and tooltip_col in df.columns and pd.notna(row[tooltip_col]) else f"Lat: {row[lat_col]:.4f}, Lon: {row[lon_col]:.4f}"
                                folium.Marker(
                                    [row[lat_col], row[lon_col]],
                                    popup=popup_text
                                ).add_to(m)
                            folium_static(m)
                    else:
                        st.warning("No valid latitude/longitude data points to display after filtering.")
                elif not map_df_viz.empty:
                    st.warning("Selected latitude/longitude columns must be numeric.")
                else:
                    st.info("No data to display on map after dropping NaNs from selected columns.")
            else:
                st.info("Select valid latitude and longitude columns to display the map.")

        # NEW FEATURE 16: K-Means Clustering Analysis
        with st.expander("💠 K-Means Clustering Analysis"):
            st.subheader("Unsupervised Clustering")
            if len(numeric_cols) >= 2:
                cluster_features = st.multiselect("Select Features for Clustering", numeric_cols, default=numeric_cols[:min(2, len(numeric_cols))], key="kmeans_features")
                if len(cluster_features) >= 2:
                    num_clusters = st.slider("Number of Clusters (K)", 2, 10, 3, key="kmeans_k")
                    
                    cluster_data = df[cluster_features].dropna()
                    if len(cluster_data) > num_clusters: # Ensure enough data points for clustering
                        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
                        cluster_data_copy = cluster_data.copy() # Avoid SettingWithCopyWarning
                        cluster_data_copy['Cluster'] = kmeans.fit_predict(cluster_data_copy)
                        
                        st.write("Cluster Centers:")
                        st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=cluster_features))
                        
                        st.write(f"Data points per cluster:")
                        st.dataframe(cluster_data_copy['Cluster'].value_counts().reset_index().rename(columns={'index':'Cluster', 'Cluster':'Count'}))

                        if len(cluster_features) == 2:
                            fig_cluster = px.scatter(cluster_data_copy, x=cluster_features[0], y=cluster_features[1], 
                                                     color='Cluster', title="K-Means Clustering Results",
                                                     color_continuous_scale=px.colors.qualitative.Plotly) # Use qualitative for distinct clusters
                            st.plotly_chart(fig_cluster, use_container_width=True)
                        elif len(cluster_features) > 2:
                            st.info("More than 2 features selected. Visualizing first 2 principal components.")
                            pca = PCA(n_components=2, random_state=42)
                            principal_components = pca.fit_transform(cluster_data_copy.drop('Cluster', axis=1))
                            pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
                            pca_df['Cluster'] = cluster_data_copy['Cluster'].values # Ensure correct cluster assignment
                            
                            fig_pca_cluster = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                                                         title="K-Means Clustering (PCA Visualization)",
                                                         color_continuous_scale=px.colors.qualitative.Plotly)
                            st.plotly_chart(fig_pca_cluster, use_container_width=True)
                    else:
                        st.warning("Not enough data points for the selected number of clusters after dropping NaNs.")
                else:
                    st.warning("Select at least two numeric features for clustering.")
            else:
                st.info("Clustering requires at least two numeric columns.")

        # NEW FEATURE 17: Data Transformation Tools
        with st.expander("🔄 Data Transformation Tools"):
            st.subheader("Apply Common Transformations")
            if numeric_cols:
                transform_col = st.selectbox("Select Numeric Column to Transform", numeric_cols, key="transform_col_select")
                transform_type = st.selectbox("Transformation Type", ["None", "Log (Natural)", "Square Root", "Standard Scaler", "Min-Max Scaler"], key="transform_type_select")

                if transform_col and transform_type != "None":
                    original_series = df[transform_col].dropna()
                    transformed_series_display = original_series.copy() # For display purposes

                    try:
                        if transform_type == "Log (Natural)":
                            if (original_series <= 0).any():
                                st.warning("Log transform requires all values to be positive. Applying to positive values only or add 1 if 0 is present.")
                                transformed_series_display = np.log(original_series[original_series > 0])
                            else:
                                transformed_series_display = np.log(original_series)
                        elif transform_type == "Square Root":
                            if (original_series < 0).any():
                                st.warning("Square root transform requires all values to be non-negative. Applying to non-negative values only.")
                                transformed_series_display = np.sqrt(original_series[original_series >=0])
                            else:
                                transformed_series_display = np.sqrt(original_series)
                        elif transform_type == "Standard Scaler":
                            scaler = StandardScaler()
                            transformed_series_display = scaler.fit_transform(original_series.values.reshape(-1, 1)).flatten()
                        elif transform_type == "Min-Max Scaler":
                            scaler = MinMaxScaler()
                            transformed_series_display = scaler.fit_transform(original_series.values.reshape(-1, 1)).flatten()
                        
                        col_t1, col_t2 = st.columns(2)
                        with col_t1:
                            fig_before = px.histogram(original_series, title=f"Original: {transform_col}", nbins=30, color_discrete_sequence=[custom_color])
                            st.plotly_chart(fig_before, use_container_width=True)
                        with col_t2:
                            fig_after = px.histogram(transformed_series_display, title=f"Transformed: {transform_col} ({transform_type})", nbins=30)
                            st.plotly_chart(fig_after, use_container_width=True)

                        if st.button(f"Apply '{transform_type}' to '{transform_col}' in DataFrame", key=f"apply_transform_{transform_col}"):
                            # Apply transformation to the actual DataFrame column
                            if transform_type == "Log (Natural)":
                                df[transform_col] = df[transform_col].apply(lambda x: np.log(x) if pd.notna(x) and x > 0 else x)
                            elif transform_type == "Square Root":
                                df[transform_col] = df[transform_col].apply(lambda x: np.sqrt(x) if pd.notna(x) and x >= 0 else x)
                            elif transform_type == "Standard Scaler":
                                scaler_apply = StandardScaler()
                                df[transform_col] = scaler_apply.fit_transform(df[[transform_col]].dropna()) # Apply only on non-NaN
                            elif transform_type == "Min-Max Scaler":
                                scaler_apply = MinMaxScaler()
                                df[transform_col] = scaler_apply.fit_transform(df[[transform_col]].dropna()) # Apply only on non-NaN
                            st.success(f"Transformation '{transform_type}' applied to '{transform_col}'. Rerun other analyses if needed.")
                            st.rerun() 

                    except Exception as e:
                        st.error(f"Error during transformation: {e}")
            else:
                st.info("No numeric columns available for transformation.")

        # NEW FEATURE 18: Interactive Pivot Table Creator
        with st.expander("📋 Interactive Pivot Table Creator"):
            st.subheader("Summarize Data with Pivot Tables")
            if (categorical_cols or date_cols) and numeric_cols:
                pivot_rows_options = categorical_cols + date_cols
                pivot_cols_options = [None] + categorical_cols + date_cols # Allow no column selection
                
                pivot_rows = st.multiselect("Select Row(s)", pivot_rows_options, default=pivot_rows_options[0] if pivot_rows_options else None, key="pivot_rows")
                pivot_cols_selected = st.multiselect("Select Column(s) (Optional)", pivot_cols_options, default=None, key="pivot_cols")
                pivot_values = st.selectbox("Select Value Column (Numeric)", numeric_cols, index=0 if numeric_cols else None, key="pivot_values")
                agg_func = st.selectbox("Aggregation Function", ["mean", "sum", "count", "min", "max", "median", "std", "var"], key="pivot_agg")

                if pivot_rows and pivot_values:
                    try:
                        pivot_table_df = pd.pivot_table(df, index=pivot_rows, 
                                                        columns=pivot_cols_selected if pivot_cols_selected else None, 
                                                        values=pivot_values, aggfunc=agg_func, dropna=False) # Keep dropna=False to see all groups
                        st.dataframe(pivot_table_df)
                    except Exception as e:
                        st.error(f"Could not create pivot table: {e}")
                else:
                    st.info("Select at least Rows and a Value column to create a pivot table.")
            else:
                st.info("Pivot tables require at least one categorical/date column and one numeric column.")

        # NEW FEATURE 19: Simple Time Series Forecasting
        if date_cols and numeric_cols:
            with st.expander("📉 Simple Time Series Forecasting (Exponential Smoothing)"):
                st.subheader("Basic Forecasting")
                forecast_date_col = st.selectbox("Select Date Column for Forecasting", date_cols, key="forecast_date")
                forecast_value_col = st.selectbox("Select Value Column for Forecasting", numeric_cols, key="forecast_value")
                forecast_periods = st.number_input("Number of Periods to Forecast", min_value=1, max_value=365, value=12, key="forecast_periods")

                if forecast_date_col and forecast_value_col:
                    ts_forecast_data = df[[forecast_date_col, forecast_value_col]].copy().dropna()
                    ts_forecast_data = ts_forecast_data.sort_values(forecast_date_col)
                    ts_forecast_data = ts_forecast_data.set_index(forecast_date_col)
                    
                    # Resample to daily frequency, if multiple records per day, take mean. Fill missing with ffill.
                    # This is a common preprocessing step for many time series models.
                    # ts_forecast_data = ts_forecast_data[forecast_value_col].resample('D').mean().fillna(method='ffill')
                    # For simplicity, let's use the data as is, assuming it's somewhat regular or user prepared it.
                    # If data has no inherent frequency, infer_freq might fail.
                    
                    if len(ts_forecast_data) >= 10: # Minimum data for forecasting
                        try:
                            series_to_forecast = ts_forecast_data[forecast_value_col]
                            # Attempt to infer frequency if not set, needed for ExponentialSmoothing forecast index
                            inferred_freq = pd.infer_freq(series_to_forecast.index)
                            if inferred_freq:
                                series_to_forecast = series_to_forecast.asfreq(inferred_freq) # Ensure frequency
                            
                            model = ExponentialSmoothing(series_to_forecast, 
                                                         initialization_method="estimated",
                                                         trend='add', seasonal=None).fit() # Additive trend, no seasonality for simplicity
                            forecast = model.forecast(forecast_periods)
                            
                            fig_forecast = go.Figure()
                            fig_forecast.add_trace(go.Scatter(x=series_to_forecast.index, y=series_to_forecast,
                                                            mode='lines', name='Actual', line=dict(color=custom_color)))
                            fig_forecast.add_trace(go.Scatter(x=model.fittedvalues.index, y=model.fittedvalues,
                                                            mode='lines', name='Fitted', line=dict(dash='dash')))
                            fig_forecast.add_trace(go.Scatter(x=forecast.index, y=forecast.values,
                                                            mode='lines', name='Forecast', line=dict(color='red')))
                            fig_forecast.update_layout(title=f"Forecast for {forecast_value_col}",
                                                     xaxis_title=forecast_date_col, yaxis_title=forecast_value_col)
                            st.plotly_chart(fig_forecast, use_container_width=True)
                            
                            st.write("Forecasted Values:")
                            st.dataframe(forecast.reset_index().rename(columns={'index': forecast_date_col, 0: 'Forecasted Value'}))

                        except Exception as e:
                            st.error(f"Forecasting error: {e}. Ensure data is suitable (e.g., enough points, numeric, regular time index).")
                    else:
                        st.warning("Not enough data points (minimum 10 required) for forecasting after processing.")
                else:
                    st.info("Select a date column and a numeric value column for forecasting.")

        # NEW FEATURE 20: Dedicated PCA Explorer
        with st.expander("🔬 Principal Component Analysis (PCA) Explorer"):
            st.subheader("Dimensionality Reduction & Variance Analysis")
            if len(numeric_cols) >= 2:
                pca_features = st.multiselect(
                    "Select Numeric Features for PCA",
                    numeric_cols,
                    default=numeric_cols[:min(5, len(numeric_cols))], # Default to first 5 or all if less
                    key="pca_features_select"
                )

                if len(pca_features) >= 2:
                    pca_data = df[pca_features].dropna()
                    
                    if not pca_data.empty:
                        # Standardize data before PCA
                        scaler_pca = StandardScaler()
                        scaled_pca_data = scaler_pca.fit_transform(pca_data)
                        
                        n_components_pca = min(len(pca_features), len(pca_data)) # Max components is min(n_samples, n_features)
                        if n_components_pca > 0:
                            pca_model = PCA(n_components=n_components_pca, random_state=42)
                            principal_components = pca_model.fit_transform(scaled_pca_data)
                            
                            # Explained Variance
                            explained_variance_ratio = pca_model.explained_variance_ratio_
                            cumulative_explained_variance = np.cumsum(explained_variance_ratio)
                            
                            pc_labels = [f"PC{i+1}" for i in range(n_components_pca)]
                            
                            fig_explained_var = go.Figure()
                            fig_explained_var.add_trace(go.Bar(x=pc_labels, y=explained_variance_ratio, name='Explained Variance per Component', marker_color=custom_color))
                            fig_explained_var.add_trace(go.Scatter(x=pc_labels, y=cumulative_explained_variance, name='Cumulative Explained Variance', mode='lines+markers'))
                            fig_explained_var.update_layout(title="Explained Variance by Principal Component", xaxis_title="Principal Component", yaxis_title="Explained Variance Ratio")
                            st.plotly_chart(fig_explained_var, use_container_width=True)

                            # Component Loadings
                            st.write("#### Principal Component Loadings")
                            loadings_df = pd.DataFrame(pca_model.components_.T, columns=pc_labels, index=pca_features)
                            st.dataframe(loadings_df.style.background_gradient(cmap='viridis', axis=None))

                            # 2D Scatter Plot of PC1 vs PC2
                            pca_result_df = pd.DataFrame(data=principal_components[:, :2], columns=['PC1', 'PC2'])
                            color_by_pca = st.selectbox("Color Scatter Plot by (Optional Categorical Column)", [None] + categorical_cols, key="pca_scatter_color")
                            
                            if color_by_pca and color_by_pca in df.columns:
                                pca_result_df[color_by_pca] = df.loc[pca_data.index, color_by_pca].values # Align with original df index
                            
                            fig_pca_scatter = px.scatter(pca_result_df, x='PC1', y='PC2', color=color_by_pca if color_by_pca else None, title="Data Projected onto First Two Principal Components")
                            st.plotly_chart(fig_pca_scatter, use_container_width=True)
                        else:
                            st.warning("Not enough features or samples to perform PCA after processing.")
                    else:
                        st.warning("No data available for PCA after dropping NaNs from selected features.")
                else:
                    st.warning("Please select at least two numeric features for PCA.")
            else:
                st.info("PCA requires at least two numeric columns in the dataset.")

        # NEW FEATURE 21: Sentiment Analysis for Text Columns
        with st.expander("💬 Sentiment Analysis for Text Data"):
            st.subheader("Analyze Sentiment in Text Columns")
            
            # Attempt to download vader_lexicon if not found
            try:
                nltk.data.find('sentiment/vader_lexicon.zip')
            except LookupError: # More appropriate exception for nltk.data.find
                st.info("Downloading VADER lexicon for sentiment analysis (one-time download)...")
                try:
                    nltk.download('vader_lexicon')
                    st.success("VADER lexicon downloaded successfully!")
                except Exception as e:
                    st.error(f"Could not download VADER lexicon: {e}. Sentiment analysis might not work.")
            except Exception as e: # Catch other potential errors with nltk.data.find
                st.warning(f"Could not verify VADER lexicon: {e}")

            text_cols = categorical_cols # Text data is often in categorical columns
            if text_cols:
                selected_text_col = st.selectbox(
                    "Select Text Column for Sentiment Analysis",
                    text_cols,
                    key="sentiment_text_col"
                )

                if selected_text_col and df[selected_text_col].dtype == 'object':
                    if st.button("Analyze Sentiment", key="analyze_sentiment_button"):
                        sid = SentimentIntensityAnalyzer()
                        
                        # Ensure the column is string type and handle NaNs
                        texts_to_analyze = df[selected_text_col].astype(str).fillna('')
                        
                        sentiments = texts_to_analyze.apply(lambda text: sid.polarity_scores(text))
                        sentiment_df = pd.DataFrame(list(sentiments))
                        sentiment_df.index = df.index # Align index with original df

                        st.write("#### Sentiment Score Distribution (Compound Score)")
                        fig_sentiment_hist = px.histogram(sentiment_df, x="compound", nbins=30, title="Distribution of Compound Sentiment Scores", color_discrete_sequence=[custom_color])
                        st.plotly_chart(fig_sentiment_hist, use_container_width=True)

                        st.write("#### Average Sentiment Scores")
                        st.json({
                            "Average Compound": f"{sentiment_df['compound'].mean():.3f}",
                            "Average Positive": f"{sentiment_df['pos'].mean():.3f}",
                            "Average Neutral": f"{sentiment_df['neu'].mean():.3f}",
                            "Average Negative": f"{sentiment_df['neg'].mean():.3f}",
                        })

                        if st.checkbox("Add Sentiment Scores to DataFrame?", key="add_sentiment_cols_cb"):
                            df[f'{selected_text_col}_sentiment_compound'] = sentiment_df['compound']
                            df[f'{selected_text_col}_sentiment_pos'] = sentiment_df['pos']
                            df[f'{selected_text_col}_sentiment_neu'] = sentiment_df['neu']
                            df[f'{selected_text_col}_sentiment_neg'] = sentiment_df['neg']
                            st.success(f"Sentiment scores from '{selected_text_col}' added to DataFrame. Rerun other analyses if needed.")
                            st.rerun()
                elif selected_text_col:
                    st.warning(f"Column '{selected_text_col}' does not appear to be a text column (object/string type).")
            else:
                st.info("No categorical (potential text) columns found for sentiment analysis.")

        # NEW FEATURE 22: User-Defined Calculated Fields
        with st.expander("🧮 User-Defined Calculated Fields"):
            st.subheader("Create New Columns from Formulas")
            
            new_col_name = st.text_input("Enter Name for New Calculated Column", key="new_calc_col_name")
            formula = st.text_input("Enter Formula (e.g., 'ColumnA * 2 + ColumnB / ColumnC')", 
                                    placeholder="Use existing column names. Available: " + ", ".join(df.columns),
                                    key="calc_col_formula")
            st.caption("You can use standard arithmetic operators (+, -, *, /) and numpy functions (e.g., np.log(ColumnA)). Ensure column names are exactly as in the dataset.")

            if new_col_name and formula:
                if st.button("Preview & Add Calculated Column", key="add_calc_col_button"):
                    try:
                        # For safety and convenience, df.eval() is good for simple arithmetic.
                        # For more complex operations involving np, a more controlled eval might be needed or provide specific functions.
                        # Here, we'll try df.eval and fall back to a more general eval if np is used.
                        if "np." in formula: # If numpy functions are used
                            calculated_series = df.eval(formula, engine='python', local_dict={'np': np}, global_dict={})
                        else:
                            calculated_series = df.eval(formula, engine='python') # engine='python' allows more complex expressions
                        
                        st.write("#### Preview of Calculated Column (First 5 Rows):")
                        st.dataframe(calculated_series.head())

                        if st.checkbox(f"Confirm and Add '{new_col_name}' to DataFrame?", key="confirm_add_calc_col"):
                            df[new_col_name] = calculated_series
                            st.success(f"Calculated column '{new_col_name}' added to DataFrame. Rerun other analyses if needed.")
                            # Update column lists
                            if pd.api.types.is_numeric_dtype(df[new_col_name]) and new_col_name not in numeric_cols:
                                numeric_cols.append(new_col_name)
                            elif df[new_col_name].dtype == 'object' and new_col_name not in categorical_cols:
                                categorical_cols.append(new_col_name)
                            st.rerun()

                    except Exception as e:
                        st.error(f"Error calculating field '{new_col_name}': {e}")
                        st.info("Please check your formula and ensure column names are correct and operations are valid.")
            elif new_col_name or formula:
                st.warning("Please provide both a name and a formula for the calculated column.")

        # NEW FEATURE 24: Data Deduplication Utility
        with st.expander("🗑️ Data Deduplication Utility"):
            st.subheader("Identify and Remove Duplicate Rows")

            if not df.empty:
                all_columns = df.columns.tolist()
                subset_cols = st.multiselect(
                    "Select columns to consider for identifying duplicates (leave empty for all columns)",
                    all_columns,
                    default=[],
                    key="dedup_subset_cols"
                )

                check_subset = subset_cols if subset_cols else None
                
                duplicates_series = df.duplicated(subset=check_subset, keep=False)
                num_duplicates = duplicates_series.sum()

                st.metric("Number of Duplicate Rows Found (considering all occurrences)", num_duplicates)

                if num_duplicates > 0:
                    st.write("#### Duplicate Rows:")
                    st.dataframe(df[duplicates_series])

                    keep_option = st.radio(
                        "Which duplicates to keep?",
                        ('first', 'last', 'Remove All Duplicates (False)'),
                        index=0,
                        key="dedup_keep_option"
                    )
                    keep_param = keep_option if keep_option != 'Remove All Duplicates (False)' else False

                    if st.button("Remove Duplicates", key="remove_duplicates_button"):
                        df_deduplicated = df.drop_duplicates(subset=check_subset, keep=keep_param)
                        rows_removed = len(df) - len(df_deduplicated)
                        df = df_deduplicated
                        st.success(f"{rows_removed} duplicate rows removed. DataFrame updated.")
                        st.rerun()
                else:
                    st.info("No duplicate rows found based on the selected criteria.")
            else:
                st.info("Upload data to use the deduplication utility.")

        # NEW FEATURE 25: Interactive Data Binning
        with st.expander("📊 Interactive Data Binning"):
            st.subheader("Categorize Numeric Data into Bins")
            if numeric_cols:
                bin_col_select = st.selectbox("Select Numeric Column to Bin", numeric_cols, key="bin_col_select")
                new_binned_col_name = st.text_input("Name for New Binned Column", f"{bin_col_select}_binned", key="new_binned_col_name")
                
                bin_method = st.radio("Binning Method", ["Equal Width (Number of Bins)", "Custom Edges"], key="bin_method_radio")

                binned_series_preview = None

                if bin_method == "Equal Width (Number of Bins)":
                    num_bins_for_width = st.number_input("Number of Bins", min_value=2, max_value=50, value=5, key="num_bins_for_width")
                    if st.button("Preview Binned Column (Equal Width)", key="preview_bin_equal_width"):
                        try:
                            binned_series_preview = pd.cut(df[bin_col_select], bins=num_bins_for_width, include_lowest=True)
                        except Exception as e:
                            st.error(f"Error creating bins: {e}")
                
                elif bin_method == "Custom Edges":
                    custom_edges_str = st.text_input("Enter Custom Bin Edges (comma-separated, e.g., 0,10,20,30)", key="custom_bin_edges")
                    if st.button("Preview Binned Column (Custom Edges)", key="preview_bin_custom_edges"):
                        try:
                            edges = sorted([float(edge.strip()) for edge in custom_edges_str.split(',') if edge.strip()])
                            if len(edges) >= 2:
                                binned_series_preview = pd.cut(df[bin_col_select], bins=edges, include_lowest=True, right=True) # Default right=True
                            else:
                                st.warning("Please provide at least two edges (e.g., min_val, max_val).")
                        except Exception as e:
                            st.error(f"Error creating bins with custom edges: {e}")

                if binned_series_preview is not None:
                    st.write("#### Preview of Binned Column (Value Counts):")
                    st.dataframe(binned_series_preview.value_counts().sort_index().reset_index().rename(columns={'index': 'Bin Range', bin_col_select: 'Count'}))
                    if st.checkbox(f"Confirm and Add '{new_binned_col_name}' to DataFrame?", key="confirm_add_binned_col"):
                        df[new_binned_col_name] = binned_series_preview
                        st.success(f"Binned column '{new_binned_col_name}' added. Rerun other analyses if needed.")
                        if new_binned_col_name not in categorical_cols: # Binned column is categorical
                            categorical_cols.append(new_binned_col_name)
                        st.rerun()
            else:
                st.info("No numeric columns available for binning.")

        # NEW FEATURE 26: Decision Tree Explorer (Was Feature 27)
        with st.expander("🌳 Decision Tree Explorer"):
            st.subheader("Train and Analyze Decision Tree Models")
            if not df.empty:
                all_cols_dt = df.columns.tolist()
                target_col_dt = st.selectbox("Select Target Variable (for Decision Tree)", all_cols_dt, key="dt_target_col")

                if target_col_dt:
                    # Determine task type
                    if df[target_col_dt].dtype in [np.number, 'int64', 'float64'] and df[target_col_dt].nunique() > 10: # Heuristic for regression
                        task_type_dt = "Regression"
                        model_dt = DecisionTreeRegressor(random_state=42)
                        criterion_options_dt = ["squared_error", "friedman_mse", "absolute_error", "poisson"]
                    else:
                        task_type_dt = "Classification"
                        model_dt = DecisionTreeClassifier(random_state=42)
                        criterion_options_dt = ["gini", "entropy", "log_loss"]
                    st.write(f"**Detected Task Type:** {task_type_dt}")

                    feature_cols_dt_options = [col for col in all_cols_dt if col != target_col_dt]
                    feature_cols_dt = st.multiselect("Select Feature Columns", feature_cols_dt_options, default=feature_cols_dt_options[:min(3, len(feature_cols_dt_options))], key="dt_feature_cols")

                    if feature_cols_dt:
                        # Preprocessing for DT
                        df_dt_processed = df[[target_col_dt] + feature_cols_dt].copy().dropna() # Drop NaNs for simplicity here
                        
                        # Encode categorical features
                        categorical_features_dt = df_dt_processed[feature_cols_dt].select_dtypes(include='object').columns.tolist()
                        if categorical_features_dt:
                            df_dt_processed = pd.get_dummies(df_dt_processed, columns=categorical_features_dt, drop_first=True)
                        
                        X_dt = df_dt_processed.drop(target_col_dt, axis=1)
                        y_dt = df_dt_processed[target_col_dt]

                        # Encode target if classification and target is object/string
                        if task_type_dt == "Classification" and y_dt.dtype == 'object':
                            le = LabelEncoder()
                            y_dt = le.fit_transform(y_dt)
                            class_names_dt = le.classes_.astype(str) # For plot_tree
                        else:
                            class_names_dt = None

                        if len(X_dt) > 10 and len(X_dt.columns) > 0:
                            X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X_dt, y_dt, test_size=0.3, random_state=42)

                            st.sidebar.subheader("Decision Tree Hyperparameters")
                            
                            # Define and store initial default values for DT
                            # These are used for widget initialization and for the "Reset All" button
                            st.session_state.INIT_DEFAULT_dt_max_depth = 5
                            st.session_state.INIT_DEFAULT_dt_min_samples_split = 2
                            st.session_state.INIT_DEFAULT_dt_min_samples_leaf = 1
                            
                            default_dt_criterion_val = "squared_error" if task_type_dt == "Regression" else "gini"
                            if default_dt_criterion_val not in criterion_options_dt:
                                default_dt_criterion_val = criterion_options_dt[0] if criterion_options_dt else "gini"
                            st.session_state.INIT_DEFAULT_dt_criterion = default_dt_criterion_val

                            max_depth_dt = st.sidebar.slider("Max Depth (DT)", 2, 30, st.session_state.INIT_DEFAULT_dt_max_depth, 1, key="dt_max_depth")
                            min_samples_split_dt = st.sidebar.slider("Min Samples Split (DT)", 2, 20, st.session_state.INIT_DEFAULT_dt_min_samples_split, key="dt_min_samples_split")
                            min_samples_leaf_dt = st.sidebar.slider("Min Samples Leaf (DT)", 1, 20, st.session_state.INIT_DEFAULT_dt_min_samples_leaf, key="dt_min_samples_leaf")

                            try:
                                current_dt_criterion_val_for_select = st.session_state.get("dt_criterion", st.session_state.INIT_DEFAULT_dt_criterion)
                                if current_dt_criterion_val_for_select not in criterion_options_dt:
                                    current_dt_criterion_val_for_select = st.session_state.INIT_DEFAULT_dt_criterion
                                initial_dt_criterion_index = criterion_options_dt.index(current_dt_criterion_val_for_select)
                            except ValueError:
                                initial_dt_criterion_index = 0
                            except Exception: # Catch if criterion_options_dt is empty
                                initial_dt_criterion_index = 0

                            criterion_dt = st.sidebar.selectbox("Criterion (DT)", criterion_options_dt, index=initial_dt_criterion_index, key="dt_criterion")

                            model_dt.set_params(max_depth=max_depth_dt, min_samples_split=min_samples_split_dt, min_samples_leaf=min_samples_leaf_dt, criterion=criterion_dt)
                            
                            if st.button("Train & Evaluate Decision Tree", key="train_dt_button"):
                                # Ensure model instance is fresh if hyperparameters changed via sidebar
                                model_dt.set_params(max_depth=max_depth_dt, min_samples_split=min_samples_split_dt, min_samples_leaf=min_samples_leaf_dt, criterion=criterion_dt)
                                model_dt.fit(X_train_dt, y_train_dt)
                                y_pred_dt = model_dt.predict(X_test_dt)

                                st.subheader("Model Performance")
                                if task_type_dt == "Regression":
                                    st.metric("R-squared (R²)", f"{r2_score(y_test_dt, y_pred_dt):.3f}")
                                    st.metric("Mean Squared Error (MSE)", f"{mean_squared_error(y_test_dt, y_pred_dt):.3f}")
                                else: # Classification
                                    st.metric("Accuracy", f"{accuracy_score(y_test_dt, y_pred_dt):.3f}")
                                    st.text("Classification Report:")
                                    st.text(classification_report(y_test_dt, y_pred_dt, target_names=class_names_dt, zero_division=0))

                                st.subheader("Feature Importances")
                                importances_dt = pd.DataFrame({'feature': X_train_dt.columns, 'importance': model_dt.feature_importances_}).sort_values('importance', ascending=False)
                                st.dataframe(importances_dt)
                                fig_imp_dt = px.bar(importances_dt, x='importance', y='feature', orientation='h', title="Decision Tree Feature Importances")
                                st.plotly_chart(fig_imp_dt, use_container_width=True)

                                st.subheader("Decision Tree Structure")
                                st.write("#### Text Representation:")
                                st.text(export_text(model_dt, feature_names=list(X_train_dt.columns)))
                                
                                st.write("#### Graphical Plot (Matplotlib):")
                                if max_depth_dt > 7: # Suggest limiting depth for readability
                                    st.warning("Plotting a very deep tree might be slow and hard to read. Consider reducing Max Depth for visualization.")
                                fig_tree, ax_tree = plt.subplots(figsize=(max(15, max_depth_dt*2), max(10, max_depth_dt*1.5))) # Dynamic figsize
                                plot_tree(model_dt, filled=True, feature_names=list(X_train_dt.columns), class_names=class_names_dt, ax=ax_tree, fontsize=8, rounded=True)
                                st.pyplot(fig_tree)
                        else:
                            st.warning("Not enough data or features after preprocessing for Decision Tree training.")
                    else:
                        st.info("Select feature columns to proceed.")
            else:
                st.info("Upload data to use the Decision Tree explorer.")

        # NEW FEATURE: Random Forest Explorer
        with st.expander("🌲 Random Forest Explorer"):
            st.subheader("Train and Analyze Random Forest Models")
            if not df.empty:
                all_cols_rf = df.columns.tolist()
                target_col_rf = st.selectbox("Select Target Variable (for Random Forest)", all_cols_rf, key="rf_target_col")

                if target_col_rf:
                    # Determine task type
                    if df[target_col_rf].dtype in [np.number, 'int64', 'float64'] and df[target_col_rf].nunique() > 10: # Heuristic for regression
                        task_type_rf = "Regression"
                        model_rf_instance = RandomForestRegressor(random_state=42)
                        criterion_options_rf = ["squared_error", "absolute_error", "friedman_mse", "poisson"]
                    else:
                        task_type_rf = "Classification"
                        model_rf_instance = RandomForestClassifier(random_state=42)
                        criterion_options_rf = ["gini", "entropy", "log_loss"]
                    st.write(f"**Detected Task Type:** {task_type_rf}")

                    feature_cols_rf_options = [col for col in all_cols_rf if col != target_col_rf]
                    feature_cols_rf = st.multiselect("Select Feature Columns for RF", feature_cols_rf_options, default=feature_cols_rf_options[:min(3, len(feature_cols_rf_options))], key="rf_feature_cols")

                    if feature_cols_rf:
                        # Preprocessing for RF
                        df_rf_processed = df[[target_col_rf] + feature_cols_rf].copy().dropna()
                        
                        # Encode categorical features
                        categorical_features_rf = df_rf_processed[feature_cols_rf].select_dtypes(include='object').columns.tolist()
                        if categorical_features_rf:
                            df_rf_processed = pd.get_dummies(df_rf_processed, columns=categorical_features_rf, drop_first=True)
                        
                        X_rf_cols = [col for col in df_rf_processed.columns if col != target_col_rf] # Get actual feature columns after dummification
                        X_rf = df_rf_processed[X_rf_cols]
                        y_rf = df_rf_processed[target_col_rf]

                        # Encode target if classification and target is object/string
                        class_names_rf = None
                        if task_type_rf == "Classification" and y_rf.dtype == 'object':
                            le_rf = LabelEncoder()
                            y_rf = le_rf.fit_transform(y_rf)
                            class_names_rf = le_rf.classes_.astype(str)

                        if len(X_rf) > 10 and len(X_rf.columns) > 0:
                            X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.3, random_state=42)

                            st.sidebar.subheader("Random Forest Hyperparameters")
                            
                            # Define and store initial default values for RF
                            # These are used for widget initialization and for the "Reset All" button
                            st.session_state.INIT_DEFAULT_rf_n_estimators = 100
                            st.session_state.INIT_DEFAULT_rf_max_depth = 10
                            st.session_state.INIT_DEFAULT_rf_min_samples_split = 2
                            st.session_state.INIT_DEFAULT_rf_min_samples_leaf = 1
                            
                            if task_type_rf == "Regression":
                                task_specific_default_rf_criterion = "squared_error"
                            else: # Classification
                                task_specific_default_rf_criterion = "gini"
                            
                            # Ensure default_rf_criterion is valid within the current options
                            if task_specific_default_rf_criterion not in criterion_options_rf:
                                task_specific_default_rf_criterion = criterion_options_rf[0] if criterion_options_rf else "gini" # Fallback
                            st.session_state.INIT_DEFAULT_rf_criterion = task_specific_default_rf_criterion

                            # Widgets - their keys link them to st.session_state
                            # The third argument (default value) is used if the key is not in st.session_state yet.
                            n_estimators_rf = st.sidebar.slider("Number of Estimators (Trees)", 10, 500, st.session_state.INIT_DEFAULT_rf_n_estimators, 10, key="rf_n_estimators")
                            max_depth_rf = st.sidebar.slider("Max Depth (RF)", 2, 30, st.session_state.INIT_DEFAULT_rf_max_depth, 1, key="rf_max_depth")
                            min_samples_split_rf = st.sidebar.slider("Min Samples Split (RF)", 2, 20, st.session_state.INIT_DEFAULT_rf_min_samples_split, key="rf_min_samples_split")
                            min_samples_leaf_rf = st.sidebar.slider("Min Samples Leaf (RF)", 1, 20, st.session_state.INIT_DEFAULT_rf_min_samples_leaf, key="rf_min_samples_leaf")
                            
                            # Determine initial index for selectbox
                            # Uses value from session state if available, otherwise uses the calculated default_rf_criterion
                            try:
                                current_rf_criterion_val_for_select = st.session_state.get("rf_criterion", st.session_state.INIT_DEFAULT_rf_criterion)
                                # If the value from session state is no longer valid in current options, revert to default
                                if current_rf_criterion_val_for_select not in criterion_options_rf:
                                    current_rf_criterion_val_for_select = st.session_state.INIT_DEFAULT_rf_criterion
                                initial_rf_criterion_index = criterion_options_rf.index(current_rf_criterion_val_for_select)
                            except ValueError: 
                                # Fallback if default_rf_criterion itself is somehow not in options (e.g. options changed unexpectedly)
                                initial_criterion_index = 0 
                            except Exception: # Catch if criterion_options_rf is empty
                                initial_rf_criterion_index = 0

                            criterion_rf = st.sidebar.selectbox("Criterion (RF)", criterion_options_rf, index=initial_rf_criterion_index, key="rf_criterion")

                            # Unified Reset button for ALL model parameters (DT and RF)
                            st.sidebar.button("Reset All Model Parameters", 
                                              key="reset_all_model_params_button",
                                              on_click=reset_all_model_parameters_callback) # Use the callback

                            model_rf_instance.set_params(
                                n_estimators=n_estimators_rf,
                                max_depth=max_depth_rf, 
                                min_samples_split=min_samples_split_rf, 
                                min_samples_leaf=min_samples_leaf_rf, 
                                criterion=criterion_rf
                            )
                            
                            if st.button("Train & Evaluate Random Forest", key="train_rf_button"):
                                # Ensure model instance is fresh if hyperparameters changed via sidebar
                                model_rf_instance.set_params(n_estimators=n_estimators_rf, max_depth=max_depth_rf, min_samples_split=min_samples_split_rf, min_samples_leaf=min_samples_leaf_rf, criterion=criterion_rf)
                                model_rf_instance.fit(X_train_rf, y_train_rf)
                                y_pred_rf = model_rf_instance.predict(X_test_rf)

                                st.subheader("Model Performance (Random Forest)")
                                if task_type_rf == "Regression":
                                    st.metric("R-squared (R²)", f"{r2_score(y_test_rf, y_pred_rf):.3f}")
                                    st.metric("Mean Squared Error (MSE)", f"{mean_squared_error(y_test_rf, y_pred_rf):.3f}")
                                else: # Classification
                                    st.metric("Accuracy", f"{accuracy_score(y_test_rf, y_pred_rf):.3f}")
                                    st.text("Classification Report:")
                                    st.text(classification_report(y_test_rf, y_pred_rf, target_names=class_names_rf, zero_division=0))

                                st.subheader("Feature Importances (Random Forest)")
                                importances_rf = pd.DataFrame({'feature': X_train_rf.columns, 'importance': model_rf_instance.feature_importances_}).sort_values('importance', ascending=False)
                                st.dataframe(importances_rf)
                                fig_imp_rf = px.bar(importances_rf, x='importance', y='feature', orientation='h', title="Random Forest Feature Importances")
                                st.plotly_chart(fig_imp_rf, use_container_width=True)
                        else:
                            st.warning("Not enough data or features after preprocessing for Random Forest training.")
                    else:
                        st.info("Select feature columns for Random Forest to proceed.")
            else:
                st.info("Upload data to use the Random Forest explorer.")

        # NEW FEATURE 27: Data Grouping & Aggregation
        with st.expander("🧱 Data Grouping & Aggregation"):
            st.subheader("Group Data and Calculate Aggregate Statistics")
            if categorical_cols and numeric_cols:
                group_by_cols = st.multiselect(
                    "Select Column(s) to Group By",
                    categorical_cols + date_cols, # Allow grouping by date cols too
                    key="group_by_cols_select"
                )

                if group_by_cols:
                    agg_col_select = st.selectbox(
                        "Select Numeric Column for Aggregation",
                        numeric_cols,
                        key="agg_col_select_group"
                    )
                    
                    agg_functions_options = {
                        "Mean": "mean", "Sum": "sum", "Count": "count", 
                        "Median": "median", "Min": "min", "Max": "max", 
                        "Standard Deviation": "std", "Variance": "var",
                        "Unique Count": "nunique"
                    }
                    selected_agg_funcs_names = st.multiselect(
                        "Select Aggregation Function(s)",
                        list(agg_functions_options.keys()),
                        default=["Mean", "Count"],
                        key="agg_funcs_select"
                    )

                    if agg_col_select and selected_agg_funcs_names:
                        actual_agg_funcs = [agg_functions_options[name] for name in selected_agg_funcs_names]
                        
                        if st.button("Perform Grouping & Aggregation", key="perform_group_agg_button"):
                            try:
                                # Create a dictionary for multiple aggregations on the same column
                                agg_dict = {agg_col_select: actual_agg_funcs}
                                
                                grouped_df = df.groupby(group_by_cols).agg(agg_dict)
                                
                                # Flatten MultiIndex columns if necessary (e.g., ('Sales', 'mean') -> 'Sales_mean')
                                if isinstance(grouped_df.columns, pd.MultiIndex):
                                    grouped_df.columns = ['_'.join(col).strip() for col in grouped_df.columns.values]
                                
                                grouped_df = grouped_df.reset_index() # Bring group_by_cols back as columns

                                st.write("#### Aggregated Results:")
                                st.dataframe(grouped_df)

                                # Offer a simple bar chart for the first aggregation result if applicable
                                if len(group_by_cols) == 1 and len(actual_agg_funcs) > 0:
                                    first_agg_col_name = grouped_df.columns[-len(actual_agg_funcs)] # Get the first aggregated column
                                    if pd.api.types.is_numeric_dtype(grouped_df[first_agg_col_name]):
                                        st.write(f"#### Quick Plot: {first_agg_col_name} by {group_by_cols[0]}")
                                        fig_agg = px.bar(grouped_df.sort_values(first_agg_col_name, ascending=False).head(20), 
                                                         x=group_by_cols[0], y=first_agg_col_name,
                                                         title=f"{first_agg_col_name} by {group_by_cols[0]} (Top 20)",
                                                         color_discrete_sequence=[custom_color])
                                        st.plotly_chart(fig_agg, use_container_width=True)

                            except Exception as e:
                                st.error(f"Error during grouping and aggregation: {e}")
                else:
                    st.info("Select at least one column to group by.")
            elif not categorical_cols:
                st.info("Grouping requires at least one categorical or date column.")
            elif not numeric_cols:
                st.info("Aggregation requires at least one numeric column.")

        # NEW FEATURE 28: Cross-Tabulation / Contingency Table Creator
        with st.expander("📊 Cross-Tabulation (Contingency Tables)"):
            st.subheader("Explore Relationships Between Categorical Variables")
            if len(categorical_cols) >= 2:
                row_var_ct = st.selectbox("Select Row Variable", categorical_cols, key="ct_row_var")
                col_var_ct = st.selectbox("Select Column Variable", [c for c in categorical_cols if c != row_var_ct], key="ct_col_var")

                if row_var_ct and col_var_ct:
                    normalize_ct = st.selectbox(
                        "Normalize by (Show Percentages)",
                        [False, 'index', 'columns', 'all'],
                        format_func=lambda x: {False: "Absolute Counts", 'index': "Row %", 'columns': "Column %", 'all': "Total %"}.get(x, x),
                        key="ct_normalize"
                    )

                    try:
                        contingency_table = pd.crosstab(df[row_var_ct], df[col_var_ct], normalize=normalize_ct)
                        st.write(f"#### Contingency Table: {row_var_ct} vs {col_var_ct}")
                        st.dataframe(contingency_table.style.format("{:.2%}" if normalize_ct else "{:,}"))

                        # Visualization
                        if not normalize_ct: # Plot counts if not normalized, otherwise normalized plot might be confusing with stacked bar
                            ct_plot_df = pd.crosstab(df[row_var_ct], df[col_var_ct])
                            fig_ct = px.bar(ct_plot_df, barmode='group', title=f"Grouped Bar Chart: {row_var_ct} vs {col_var_ct}", color_discrete_sequence=[custom_color])
                            st.plotly_chart(fig_ct, use_container_width=True)
                        
                        # Chi-squared test
                        if st.checkbox("Perform Chi-squared Test for Independence", key="ct_chi2_test"):
                            chi2, p, dof, expected = stats.chi2_contingency(pd.crosstab(df[row_var_ct], df[col_var_ct]))
                            st.write("##### Chi-squared Test Results:")
                            st.write(f"- Chi-squared Statistic: {chi2:.3f}")
                            st.write(f"- P-value: {p:.4f}")
                            st.write(f"- Degrees of Freedom: {dof}")
                            if p < 0.05:
                                st.success("Result: Significant association between variables (p < 0.05).")
                            else:
                                st.warning("Result: No significant association found (p >= 0.05).")
                    except Exception as e:
                        st.error(f"Error creating cross-tabulation: {e}")
            else:
                st.info("Cross-tabulation requires at least two categorical columns.")

        # NEW FEATURE: Auto Data Merger (Smart Join Suggestion Tool)
        with st.expander("🤝 Auto Data Merger (Smart Join Suggestion Tool)"):
            st.subheader("Merge Two Datasets")
            if len(datasets) >= 2:
                dataset_names = list(datasets.keys())

                col_merge1, col_merge2 = st.columns(2)
                with col_merge1:
                    ds1_name = st.selectbox("Select First Dataset (Left)", dataset_names, key="merge_ds1_name")
                with col_merge2:
                    ds2_options = [name for name in dataset_names if name != ds1_name] if ds1_name else dataset_names
                    ds2_name = st.selectbox("Select Second Dataset (Right)", ds2_options, key="merge_ds2_name", index=0 if ds2_options else -1)

                if ds1_name and ds2_name and ds1_name != ds2_name:
                    df1_merge = datasets[ds1_name]
                    df2_merge = datasets[ds2_name]

                    st.write(f"**Dataset 1 ({ds1_name}) Columns:** {', '.join(df1_merge.columns)}")
                    st.write(f"**Dataset 2 ({ds2_name}) Columns:** {', '.join(df2_merge.columns)}")

                    col_key1, col_key2 = st.columns(2)
                    with col_key1:
                        left_keys = st.multiselect(f"Select Join Key(s) for {ds1_name}", df1_merge.columns.tolist(), key="merge_left_keys")
                    with col_key2:
                        right_keys = st.multiselect(f"Select Join Key(s) for {ds2_name}", df2_merge.columns.tolist(), key="merge_right_keys")

                    join_type = st.selectbox(
                        "Select Join Type",
                        ["inner", "left", "right", "outer"],
                        key="merge_join_type"
                    )

                    default_new_name = f"Merged_{ds1_name.replace('Dataset_','')}_and_{ds2_name.replace('Dataset_','')}"
                    if left_keys and right_keys:
                        default_new_name = f"Merged_{ds1_name.replace('Dataset_','')}_on_{'_'.join(left_keys)}_with_{ds2_name.replace('Dataset_','')}_on_{'_'.join(right_keys)}"
                    new_merged_dataset_name = st.text_input("Name for the New Merged Dataset", value=default_new_name, key="merge_new_name")

                    # Fuzzy match placeholder
                    st.caption("Future enhancement idea: `Suggest Fuzzy Match for Keys` (requires additional libraries and logic).")

                    if left_keys and right_keys:
                        if len(left_keys) != len(right_keys):
                            st.warning("The number of join keys selected for each dataset must be the same.")
                        else:
                            # Prepare suffixes for merge
                            s1_suffix_base = ds1_name.replace("Dataset_", "").replace(" ", "")
                            s2_suffix_base = ds2_name.replace("Dataset_", "").replace(" ", "")
                            suf1 = '_' + re.sub(r'\W+', '', s1_suffix_base) if s1_suffix_base else '_left'
                            suf2 = '_' + re.sub(r'\W+', '', s2_suffix_base) if s2_suffix_base else '_right'
                            if suf1 == suf2: # Avoid identical suffixes
                                suf1 = f"{suf1}1"
                                suf2 = f"{suf2}2"

                            if st.button("Preview Merged Data", key="merge_preview_button"):
                                try:
                                    merged_preview_df = pd.merge(df1_merge, df2_merge, left_on=left_keys, right_on=right_keys, how=join_type, suffixes=(suf1, suf2))
                                    st.write("#### Merged Data Preview (First 5 rows):")
                                    st.dataframe(merged_preview_df.head())
                                    st.write(f"Shape of merged data preview: {merged_preview_df.shape}")
                                except Exception as e:
                                    st.error(f"Error during merge preview: {e}")

                            if st.button("Confirm and Add Merged Dataset", key="merge_confirm_button"):
                                if not new_merged_dataset_name:
                                    st.error("Please provide a name for the new merged dataset.")
                                elif new_merged_dataset_name in datasets:
                                    st.error(f"A dataset named '{new_merged_dataset_name}' already exists. Please choose a different name.")
                                else:
                                    try:
                                        merged_df = pd.merge(df1_merge, df2_merge, left_on=left_keys, right_on=right_keys, how=join_type, suffixes=(suf1, suf2))
                                        datasets[new_merged_dataset_name] = merged_df
                                        st.success(f"Dataset '{new_merged_dataset_name}' added successfully with {merged_df.shape[0]} rows and {merged_df.shape[1]} columns. Select it from the primary dataset dropdown.")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error merging and adding dataset: {e}")
                    elif (left_keys and not right_keys) or (not left_keys and right_keys):
                        st.warning("Please select join keys for both datasets if you intend to join on specific columns.")
                    else:
                        st.info("Select join key(s) for both datasets to proceed with merging.")
                elif ds1_name and ds2_name and ds1_name == ds2_name:
                    st.warning("Please select two different datasets to merge.")

            elif uploaded_files and len(datasets) < 2:
                st.warning("You need to upload at least two datasets to use the merge tool.")
            else:
                st.info("Upload at least two datasets to enable the data merging feature.")

        # NEW FEATURE: Column Summary Cards
        with st.expander("🧾 Column Summary Cards", expanded=False):
            st.subheader("Quick Overview of Each Column")
            if not df.empty:
                num_cards_per_row = 3 # Adjust as needed for layout
                all_df_columns = df.columns.tolist()
                
                for i in range(0, len(all_df_columns), num_cards_per_row):
                    row_cols = st.columns(num_cards_per_row)
                    for j, col_name in enumerate(all_df_columns[i : i + num_cards_per_row]):
                        if j < len(row_cols): # Ensure we don't try to access an out-of-bounds column
                            with row_cols[j]:
                                col_data = df[col_name]
                                col_type = col_data.dtype
                                missing_pct = (col_data.isnull().sum() / len(df)) * 100 if len(df) > 0 else 0
                                unique_count = col_data.nunique()
                                
                                card_html = f"""
                                <div class="column-summary-card">
                                    <h4>{col_name}</h4>
                                    <p><strong>Type:</strong> {str(col_type)}</p>
                                    <p><strong>Missing:</strong> {missing_pct:.1f}%</p>
                                    <p><strong>Unique Values:</strong> {unique_count}</p>
                                """
                                if pd.api.types.is_numeric_dtype(col_data) and not col_data.dropna().empty:
                                    card_html += f"""
                                    <p><strong>Min:</strong> {col_data.min():.2f}<br>
                                    <strong>Max:</strong> {col_data.max():.2f}<br>
                                    <strong>Mean:</strong> {col_data.mean():.2f}</p>
                                    """
                                elif pd.api.types.is_datetime64_any_dtype(col_data) and not col_data.dropna().empty:
                                    try:
                                        card_html += f"""
                                        <p><strong>Min Date:</strong> {col_data.min().strftime('%Y-%m-%d')}<br>
                                        <strong>Max Date:</strong> {col_data.max().strftime('%Y-%m-%d')}</p>
                                        """
                                    except AttributeError: # Handle NaT if min/max results in NaT
                                        card_html += "<p><strong>Min Date:</strong> N/A<br><strong>Max Date:</strong> N/A</p>"
                                card_html += "</div>"
                                st.markdown(card_html, unsafe_allow_html=True)
            else:
                st.info("No data available to display column summaries.")

        # NEW FEATURE 29 (was 23): Advanced Cohort Analysis
        with st.expander("📈 Advanced Cohort Analysis"):
            st.subheader("Track User/Customer Behavior Over Time by Cohort")
            if date_cols and categorical_cols + numeric_cols: # Need a customer/entity ID
                entity_id_col_cohort = st.selectbox("Select Customer/Entity ID Column", categorical_cols + numeric_cols, key="cohort_entity_id")
                cohort_date_col = st.selectbox("Select Acquisition/Event Date Column", date_cols, key="cohort_date_col")
                metric_col_cohort = st.selectbox("Select Metric to Analyze (e.g., Revenue, Activity Count)", numeric_cols, key="cohort_metric_col")
                
                if entity_id_col_cohort and cohort_date_col and metric_col_cohort:
                    if st.button("Analyze Cohorts", key="run_cohort_analysis"):
                        try:
                            cohort_df = df[[entity_id_col_cohort, cohort_date_col, metric_col_cohort]].copy().dropna()
                            cohort_df[cohort_date_col] = pd.to_datetime(cohort_df[cohort_date_col])
                            
                            # Determine acquisition cohort (e.g., first purchase month)
                            cohort_df['AcquisitionMonth'] = cohort_df.groupby(entity_id_col_cohort)[cohort_date_col].transform('min').dt.to_period('M')
                            cohort_df['EventMonth'] = cohort_df[cohort_date_col].dt.to_period('M')
                            
                            # Calculate cohort period
                            cohort_df['CohortPeriod'] = (cohort_df['EventMonth'].dt.year - cohort_df['AcquisitionMonth'].dt.year) * 12 + \
                                                      (cohort_df['EventMonth'].dt.month - cohort_df['AcquisitionMonth'].dt.month)

                            # Retention: Number of unique active customers per cohort period
                            cohort_pivot_retention = cohort_df.groupby(['AcquisitionMonth', 'CohortPeriod'])[entity_id_col_cohort].nunique().reset_index()
                            cohort_pivot_retention = cohort_pivot_retention.pivot_table(index='AcquisitionMonth', columns='CohortPeriod', values=entity_id_col_cohort)
                            
                            cohort_sizes = cohort_pivot_retention.iloc[:, 0]
                            retention_matrix = cohort_pivot_retention.divide(cohort_sizes, axis=0) * 100

                            st.write("#### Monthly Cohort Retention Rate (%)")
                            fig_retention, ax_retention = plt.subplots(figsize=(12, max(6, len(retention_matrix)*0.4)))
                            sns.heatmap(retention_matrix, annot=True, fmt='.1f', cmap='viridis', ax=ax_retention)
                            ax_retention.set_title('Monthly Cohort Retention (%)')
                            ax_retention.set_ylabel('Acquisition Month')
                            ax_retention.set_xlabel('Months Since Acquisition')
                            st.pyplot(fig_retention)

                            # Behavior: Average metric value per cohort period
                            cohort_pivot_behavior = cohort_df.groupby(['AcquisitionMonth', 'CohortPeriod'])[metric_col_cohort].mean().reset_index()
                            cohort_pivot_behavior = cohort_pivot_behavior.pivot_table(index='AcquisitionMonth', columns='CohortPeriod', values=metric_col_cohort)
                            
                            st.write(f"#### Average '{metric_col_cohort}' by Cohort Period")
                            fig_behavior, ax_behavior = plt.subplots(figsize=(12, max(6, len(cohort_pivot_behavior)*0.4)))
                            sns.heatmap(cohort_pivot_behavior, annot=True, fmt='.2f', cmap='coolwarm', ax=ax_behavior)
                            ax_behavior.set_title(f"Average '{metric_col_cohort}' by Cohort")
                            ax_behavior.set_ylabel('Acquisition Month')
                            ax_behavior.set_xlabel('Months Since Acquisition')
                            st.pyplot(fig_behavior)

                        except Exception as e:
                            st.error(f"Error in Cohort Analysis: {e}")
                else:
                    st.info("Select Entity ID, Date, and Metric columns for cohort analysis.")
            else:
                st.info("Cohort analysis requires date columns and categorical/numeric columns for entity ID and metrics.")

        # NEW FEATURE 30 (was 24): Market Basket Analysis
        with st.expander("🧺 Market Basket Analysis (Association Rules)"):
            st.subheader("Discover Frequently Co-purchased Items")
            if categorical_cols: # Need at least two: transaction ID and item ID
                transaction_id_col_mba = st.selectbox("Select Transaction ID Column", categorical_cols + numeric_cols, key="mba_transaction_id")
                item_id_col_mba = st.selectbox("Select Item ID Column", categorical_cols + numeric_cols, key="mba_item_id")

                if transaction_id_col_mba and item_id_col_mba:
                    min_support_mba = st.slider("Minimum Support", 0.001, 0.1, 0.01, 0.001, format="%.3f", key="mba_min_support")
                    min_confidence_mba = st.slider("Minimum Confidence", 0.1, 1.0, 0.5, 0.05, key="mba_min_confidence")

                    if st.button("Run Market Basket Analysis", key="run_mba"):
                        try:
                            mba_df_prep = df[[transaction_id_col_mba, item_id_col_mba]].copy().dropna()
                            # Create list of lists for transactions
                            transactions_mba = mba_df_prep.groupby(transaction_id_col_mba)[item_id_col_mba].apply(lambda x: list(set(x))).tolist()
                            transactions_mba = [t for t in transactions_mba if len(t) > 1] # Only transactions with >1 item

                            if not transactions_mba:
                                st.warning("No transactions with multiple items found for analysis.")
                            else:
                                te = TransactionEncoder()
                                te_ary = te.fit(transactions_mba).transform(transactions_mba)
                                basket_df_mba = pd.DataFrame(te_ary, columns=te.columns_)

                                frequent_itemsets_mba = apriori(basket_df_mba, min_support=min_support_mba, use_colnames=True, max_len=5)
                                if frequent_itemsets_mba.empty:
                                    st.info(f"No frequent itemsets found with support >= {min_support_mba}. Try lowering support.")
                                else:
                                    st.write("#### Frequent Itemsets (Top 20)")
                                    st.dataframe(frequent_itemsets_mba.sort_values("support", ascending=False).head(20))

                                    rules_mba = association_rules(frequent_itemsets_mba, metric="confidence", min_threshold=min_confidence_mba)
                                    if rules_mba.empty:
                                        st.info(f"No association rules found with confidence >= {min_confidence_mba}. Try lowering confidence or adjusting support.")
                                    else:
                                        st.write("#### Association Rules (Top 30 by Lift)")
                                        st.dataframe(rules_mba.sort_values(["lift", "confidence"], ascending=[False, False]).head(30))
                        except Exception as e:
                            st.error(f"Error in Market Basket Analysis: {e}")
                else:
                    st.info("Select Transaction ID and Item ID columns.")
            else:
                st.info("Market Basket Analysis requires categorical columns for transaction and item identifiers.")

        # NEW FEATURE 31 (was 25): Geospatial Heatmap/Density Analysis
        with st.expander("🔥 Geospatial Heatmap/Density Analysis"):
            st.subheader("Visualize Data Concentration on a Map")
            potential_lat_cols_hm = [col for col in numeric_cols if 'lat' in col.lower()]
            potential_lon_cols_hm = [col for col in numeric_cols if 'lon' in col.lower() or 'lng' in col.lower()]

            lat_col_hm_default = potential_lat_cols_hm[0] if potential_lat_cols_hm else None
            lon_col_hm_default = potential_lon_cols_hm[0] if potential_lon_cols_hm else None

            lat_col_hm = st.selectbox("Select Latitude Column for Heatmap", numeric_cols, index=numeric_cols.index(lat_col_hm_default) if lat_col_hm_default and lat_col_hm_default in numeric_cols else 0, key="geo_hm_lat")
            lon_col_hm = st.selectbox("Select Longitude Column for Heatmap", numeric_cols, index=numeric_cols.index(lon_col_hm_default) if lon_col_hm_default and lon_col_hm_default in numeric_cols else (1 if len(numeric_cols) > 1 else 0), key="geo_hm_lon")
            weight_col_hm = st.selectbox("Optional: Select Weight/Intensity Column", [None] + numeric_cols, key="geo_hm_weight")

            if lat_col_hm and lon_col_hm:
                if st.button("Generate Heatmap", key="run_heatmap"):
                    heatmap_data = df[[lat_col_hm, lon_col_hm]].copy().dropna()
                    if weight_col_hm:
                        heatmap_data['weight'] = df[weight_col_hm]
                        heatmap_data = heatmap_data.dropna(subset=['weight'])
                    
                    heatmap_data.columns = ['lat', 'lon'] + (['weight'] if weight_col_hm else [])

                    if not heatmap_data.empty and (-90 <= heatmap_data['lat'].min() and heatmap_data['lat'].max() <= 90) and \
                       (-180 <= heatmap_data['lon'].min() and heatmap_data['lon'].max() <= 180):
                        
                        m_heatmap = folium.Map(location=[heatmap_data['lat'].mean(), heatmap_data['lon'].mean()], zoom_start=6)
                        from folium.plugins import HeatMap
                        heat_data_points = [[row['lat'], row['lon'], row.get('weight', 1)] for index, row in heatmap_data.iterrows()]
                        HeatMap(heat_data_points).add_to(m_heatmap)
                        folium_static(m_heatmap)
                    else:
                        st.warning("No valid data points for heatmap or lat/lon values are out of range.")
            else:
                st.info("Select Latitude and Longitude columns to generate a heatmap.")

        # NEW FEATURE: Network Analysis of Categorical Co-occurrence
        with st.expander("🕸️ Network Analysis of Categorical Co-occurrence"):
            st.subheader("Visualize Relationships Between Categorical Values")
            if len(categorical_cols) >= 2:
                col1_net = st.selectbox("Select First Categorical Column", categorical_cols, key="net_col1")
                col2_net_options = [c for c in categorical_cols if c != col1_net]
                if col2_net_options:
                    col2_net = st.selectbox("Select Second Categorical Column", col2_net_options, key="net_col2")

                    if col1_net and col2_net:
                        if st.button("Generate Co-occurrence Network", key="run_network_analysis"):
                            try:
                                net_df = df[[col1_net, col2_net]].copy().dropna()
                                if net_df.empty:
                                    st.warning("No data available for network analysis after dropping NaNs.")
                                else:
                                    st.write(f"#### Co-occurrence Network: '{col1_net}' vs '{col2_net}'")

                                    # Create edges based on co-occurrence in the same row
                                    edges = net_df.apply(lambda row: tuple(sorted((row[col1_net], row[col2_net]))), axis=1)
                                    edge_counts = edges.value_counts().reset_index()
                                    edge_counts.columns = ['Pair', 'Frequency']

                                    # Filter by minimum frequency
                                    min_freq_net = st.slider("Minimum Co-occurrence Frequency to show edge", 1, int(edge_counts['Frequency'].max()), max(1, int(edge_counts['Frequency'].quantile(0.8))), key="net_min_freq")
                                    filtered_edges = edge_counts[edge_counts['Frequency'] >= min_freq_net]

                                    if filtered_edges.empty:
                                        st.info(f"No pairs found with co-occurrence frequency >= {min_freq_net}. Try lowering the threshold.")
                                    else:
                                        st.write(f"Top Co-occurring Pairs (Frequency >= {min_freq_net}):")
                                        st.dataframe(filtered_edges.head(20))

                                        # Build NetworkX graph
                                        G = nx.Graph()
                                        for index, row in filtered_edges.iterrows():
                                            node1, node2 = row['Pair']
                                            freq = row['Frequency']
                                            G.add_edge(node1, node2, weight=freq)

                                        if G.number_of_nodes() > 0:
                                            # Draw the network
                                            fig_net, ax_net = plt.subplots(figsize=(10, 8))
                                            pos = nx.spring_layout(G, k=0.5, iterations=50) # Layout algorithm
                                            
                                            # Draw nodes
                                            node_size = [G.degree(node) * 50 + 100 for node in G.nodes()] # Size based on degree
                                            nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='skyblue', alpha=0.9, ax=ax_net)

                                            # Draw edges
                                            edge_width = [d['weight'] / edge_counts['Frequency'].max() * 5 for (u, v, d) in G.edges(data=True)] # Width based on frequency
                                            nx.draw_networkx_edges(G, pos, width=edge_width, edge_color='gray', alpha=0.6, ax=ax_net)

                                            # Draw labels
                                            nx.draw_networkx_labels(G, pos, font_size=10, ax=ax_net)

                                            ax_net.set_title(f"Co-occurrence Network: {col1_net} vs {col2_net}")
                                            plt.axis('off')
                                            st.pyplot(fig_net)
                                        else:
                                            st.info("Network graph could not be generated (likely due to filtering resulting in no connected nodes).")

                            except Exception as e:
                                st.error(f"Error during network analysis: {e}")
                else:
                    st.warning("Need at least two distinct categorical columns.")
            else:
                st.info("Network analysis requires at least two categorical columns.")

        # NEW FEATURE: Automated Feature Engineering Suggestions
        with st.expander("🛠️ Automated Feature Engineering Suggestions"):
            st.subheader("Get Suggestions for New Features")
            st.info("This tool suggests potential new features based on existing columns. Select suggestions to add them to your DataFrame.")

            suggestions_list = []

            # Date Feature Extraction
            if date_cols:
                for d_col in date_cols:
                    suggestions_list.append({'Type': 'Date Part', 'Columns': [d_col], 'New Feature': f'{d_col}_year', 'Description': f'Year from {d_col}'})
                    suggestions_list.append({'Type': 'Date Part', 'Columns': [d_col], 'New Feature': f'{d_col}_month', 'Description': f'Month from {d_col}'})
                    suggestions_list.append({'Type': 'Date Part', 'Columns': [d_col], 'New Feature': f'{d_col}_day', 'Description': f'Day of month from {d_col}'})
                    suggestions_list.append({'Type': 'Date Part', 'Columns': [d_col], 'New Feature': f'{d_col}_dayofweek', 'Description': f'Day of week from {d_col}'})
                    suggestions_list.append({'Type': 'Date Part', 'Columns': [d_col], 'New Feature': f'{d_col}_dayofyear', 'Description': f'Day of year from {d_col}'})
                    suggestions_list.append({'Type': 'Date Part', 'Columns': [d_col], 'New Feature': f'{d_col}_weekofyear', 'Description': f'Week of year from {d_col}'})
                    suggestions_list.append({'Type': 'Date Part', 'Columns': [d_col], 'New Feature': f'{d_col}_quarter', 'Description': f'Quarter from {d_col}'})

            # Numeric Interaction Terms (simple pairs)
            if len(numeric_cols) >= 2:
                for i in range(len(numeric_cols)):
                    for j in range(i + 1, len(numeric_cols)):
                        col1, col2 = numeric_cols[i], numeric_cols[j]
                        suggestions_list.append({'Type': 'Interaction', 'Columns': [col1, col2], 'New Feature': f'{col1}_x_{col2}', 'Description': f'Interaction term: {col1} * {col2}'})
                        # Add ratio if denominator is likely non-zero
                        if df[col2].fillna(0).min() >= 0 and df[col2].fillna(0).sum() > 0: # Simple check for potential division by zero
                             suggestions_list.append({'Type': 'Ratio', 'Columns': [col1, col2], 'New Feature': f'{col1}_div_{col2}', 'Description': f'Ratio: {col1} / {col2}'})

            # Simple Polynomial Features (Square)
            if numeric_cols:
                for n_col in numeric_cols:
                     suggestions_list.append({'Type': 'Polynomial', 'Columns': [n_col], 'New Feature': f'{n_col}_squared', 'Description': f'Square of {n_col}'})

            if suggestions_list:
                suggestions_df = pd.DataFrame(suggestions_list)
                st.write("#### Suggested Features:")
                st.dataframe(suggestions_df)

                selected_suggestions_indices = st.multiselect(
                    "Select suggestions to add to DataFrame (select by index from table above)",
                    suggestions_df.index.tolist(),
                    key="selected_fe_suggestions"
                )

                if st.button("Add Selected Features", key="add_fe_button"):
                    if selected_suggestions_indices:
                        for idx in selected_suggestions_indices:
                            suggestion = suggestions_df.loc[idx]
                            new_col_name = suggestion['New Feature']
                            col_type = suggestion['Type']
                            source_cols = suggestion['Columns']

                            if new_col_name in df.columns:
                                st.warning(f"Column '{new_col_name}' already exists. Skipping.")
                                continue

                            try:
                                if col_type == 'Date Part':
                                    source_col = source_cols[0]
                                    if suggestion['New Feature'].endswith('_year'):
                                        df[new_col_name] = df[source_col].dt.year
                                    elif suggestion['New Feature'].endswith('_month'):
                                        df[new_col_name] = df[source_col].dt.month
                                    elif suggestion['New Feature'].endswith('_day'):
                                        df[new_col_name] = df[source_col].dt.day
                                    elif suggestion['New Feature'].endswith('_dayofweek'):
                                        df[new_col_name] = df[source_col].dt.dayofweek # Monday=0, Sunday=6
                                    elif suggestion['New Feature'].endswith('_dayofyear'):
                                        df[new_col_name] = df[source_col].dt.dayofyear
                                    elif suggestion['New Feature'].endswith('_weekofyear'):
                                        df[new_col_name] = df[source_col].dt.isocalendar().week.astype(int) # Use isocalendar for week
                                    elif suggestion['New Feature'].endswith('_quarter'):
                                        df[new_col_name] = df[source_col].dt.quarter
                                elif col_type == 'Interaction':
                                    col1, col2 = source_cols
                                    df[new_col_name] = df[col1] * df[col2]
                                elif col_type == 'Ratio':
                                     col1, col2 = source_cols
                                     df[new_col_name] = df[col1] / df[col2].replace(0, np.nan) # Replace 0 with NaN to avoid inf
                                elif col_type == 'Polynomial':
                                     source_col = source_cols[0]
                                     df[new_col_name] = df[source_col] ** 2

                                st.success(f"Added new feature: '{new_col_name}' ({col_type})")
                                # Update column lists and rerun
                                if pd.api.types.is_numeric_dtype(df[new_col_name]) and new_col_name not in numeric_cols:
                                    numeric_cols.append(new_col_name)
                                elif df[new_col_name].dtype == 'object' and new_col_name not in categorical_cols:
                                    categorical_cols.append(new_col_name)
                                elif pd.api.types.is_datetime64_any_dtype(df[new_col_name]) and new_col_name not in date_cols:
                                     date_cols.append(new_col_name)
                                # Rerun to update selectboxes and analysis options
                                st.rerun()

                            except Exception as e:
                                st.error(f"Error creating feature '{new_col_name}': {e}")
                    else:
                        st.info("Select at least one suggestion to add features.")
            else:
                st.info("No feature engineering suggestions could be generated based on available column types.")

        # NEW FEATURE: Simple What-If Scenario Builder
        with st.expander("💡 Simple What-If Scenario Builder"):
            st.subheader("Predict Outcome Based on Input Change")
            st.info("Uses a simple linear regression model to estimate the impact of changing one numeric variable on another.")
            if len(numeric_cols) >= 2:
                input_col_wi = st.selectbox("Select Input Variable", numeric_cols, key="wi_input_col")
                output_col_wi = st.selectbox("Select Output Variable", [c for c in numeric_cols if c != input_col_wi], key="wi_output_col")

                if input_col_wi and output_col_wi:
                    wi_df = df[[input_col_wi, output_col_wi]].dropna()

                    if len(wi_df) >= 2:
                        try:
                            # Train a simple linear regression model
                            X_wi = wi_df[[input_col_wi]]
                            y_wi = wi_df[output_col_wi]
                            model_wi = LinearRegression()
                            model_wi.fit(X_wi, y_wi)

                            st.write(f"Model: `{output_col_wi}` ≈ {model_wi.coef_[0]:.2f} * `{input_col_wi}` + {model_wi.intercept_:.2f}")
                            st.write(f"R-squared of model: {r2_score(y_wi, model_wi.predict(X_wi)):.2f}")

                            st.write("#### Scenario Input")
                            current_avg_input = wi_df[input_col_wi].mean()
                            st.write(f"Current Average `{input_col_wi}`: {current_avg_input:.2f}")

                            scenario_type = st.radio("Change Input By:", ["New Value", "Percentage Change"], key="wi_scenario_type")

                            if scenario_type == "New Value":
                                new_input_value = st.number_input(f"Enter New Value for '{input_col_wi}'", value=current_avg_input, key="wi_new_value")
                                scenario_input = np.array([[new_input_value]])
                            else: # Percentage Change
                                percentage_change = st.slider(f"Percentage Change in '{input_col_wi}'", -100, 100, 10, key="wi_pct_change")
                                new_input_value = current_avg_input * (1 + percentage_change / 100.0)
                                scenario_input = np.array([[new_input_value]])

                            if st.button("Predict Outcome", key="run_what_if"):
                                predicted_output = model_wi.predict(scenario_input)[0]
                                st.write("#### Predicted Outcome")
                                st.metric(f"Predicted '{output_col_wi}'", f"{predicted_output:.2f}")
                                st.caption("Note: This is a simple linear model prediction. Actual outcomes may vary.")

                        except Exception as e:
                            st.error(f"Error during What-If analysis: {e}")
                    else:
                        st.warning("Not enough data points (minimum 2) for the selected columns after dropping NaNs to build a model.")
            else:
                st.info("What-If scenario builder requires at least two numeric columns.")

        # NEW FEATURE: Text Column Profiler & Keyword Extractor
        with st.expander("📝 Text Column Profiler & Keyword Extractor"):
            st.subheader("Analyze Text Content")
            text_cols_profiler = df.select_dtypes(include='object').columns.tolist() # Only object type for text
            if text_cols_profiler:
                selected_text_col_profiler = st.selectbox(
                    "Select Text Column to Profile",
                    text_cols_profiler,
                    key="profiler_text_col"
                )

                if selected_text_col_profiler:
                    if st.button("Profile Text Column", key="run_text_profiler"):
                        try:
                            text_series = df[selected_text_col_profiler].astype(str).dropna()
                            if text_series.empty:
                                st.warning("Selected text column is empty after dropping NaNs.")
                            else:
                                st.write(f"#### Profile for '{selected_text_col_profiler}'")
                                
                                # Basic Stats
                                total_texts = len(text_series)
                                total_words = text_series.apply(lambda x: len(x.split())).sum()
                                avg_words_per_text = total_words / total_texts if total_texts > 0 else 0
                                
                                st.write(f"- **Total Entries:** {total_texts}")
                                st.write(f"- **Total Words:** {total_words}")
                                st.write(f"- **Average Words per Entry:** {avg_words_per_text:.2f}")

                                # Top Keywords (TF-IDF)
                                st.write("#### Top Keywords (TF-IDF)")
                                try:
                                    from sklearn.feature_extraction.text import TfidfVectorizer
                                    tfidf = TfidfVectorizer(max_features=50, stop_words='english') # Limit features for performance
                                    tfidf_matrix = tfidf.fit_transform(text_series)
                                    feature_names = tfidf.get_feature_names_out()
                                    
                                    # Sum TF-IDF scores for each word across all documents
                                    tfidf_scores = tfidf_matrix.sum(axis=0)
                                    tfidf_df = pd.DataFrame(tfidf_scores, columns=['score'], index=feature_names)
                                    tfidf_df = tfidf_df.sort_values('score', ascending=False)

                                    st.dataframe(tfidf_df.head(20))

                                    # Word Cloud
                                    st.write("#### Word Cloud of Top Keywords")
                                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(tfidf_df['score'].to_dict())
                                    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                                    ax_wc.imshow(wordcloud, interpolation='bilinear')
                                    ax_wc.axis('off')
                                    st.pyplot(fig_wc)

                                except ImportError:
                                    st.warning("Scikit-learn not installed. Cannot perform TF-IDF keyword extraction.")
                                except Exception as e:
                                    st.error(f"Error during TF-IDF or Word Cloud generation: {e}")

                        except Exception as e:
                            st.error(f"Error profiling text column: {e}")
            else:
                st.info("No text (object/string) columns found for profiling.")

        # NEW FEATURE: Distribution Fitting & Goodness-of-Fit Test
        with st.expander("🧬 Distribution Fitting & Goodness-of-Fit Test"):
            st.subheader("Fit Statistical Distributions to Numeric Data")
            if numeric_cols:
                dist_col = st.selectbox("Select Numeric Column to Fit Distribution", numeric_cols, key="dist_fit_col")

                if dist_col:
                    if st.button("Fit Distributions", key="run_dist_fitting"):
                        try:
                            data_to_fit = df[dist_col].dropna()
                            if data_to_fit.empty or len(data_to_fit) < 20: # Need reasonable data points
                                st.warning("Not enough data points (minimum 20 recommended) after dropping NaNs for distribution fitting.")
                            else:
                                st.write(f"#### Distribution Fitting Results for '{dist_col}'")

                                # Distributions to test (common ones)
                                distributions = [norm, lognorm, expon, weibull_min]
                                results_dist = []

                                for distribution in distributions:
                                    try:
                                        # Fit distribution to data
                                        params = distribution.fit(data_to_fit)

                                        # Perform Kolmogorov-Smirnov test for goodness-of-fit
                                        # Compare data distribution to the fitted distribution
                                        ks_statistic, p_value_ks = stats.kstest(data_to_fit, distribution.cdf, args=params)

                                        results_dist.append({
                                            'Distribution': distribution.name,
                                            'Parameters': params,
                                            'KS Statistic': ks_statistic,
                                            'P-value (KS Test)': p_value_ks
                                        })
                                    except Exception as e:
                                        st.warning(f"Could not fit {distribution.name}: {e}")

                                if results_dist:
                                    results_df_dist = pd.DataFrame(results_dist).sort_values('P-value (KS Test)', ascending=False)
                                    st.dataframe(results_df_dist)

                                    # Find the best fitting distribution (highest p-value)
                                    best_fit_row = results_df_dist.iloc[0]
                                    best_distribution_name = best_fit_row['Distribution']
                                    best_distribution = getattr(stats, best_distribution_name)
                                    best_params = best_fit_row['Parameters']

                                    st.info(f"**Best Fitting Distribution (based on highest KS p-value):** {best_distribution_name}")
                                    st.write(f"Parameters: {best_params}")
                                    if best_fit_row['P-value (KS Test)'] >= 0.05:
                                        st.success("The data does not significantly differ from this distribution (at alpha=0.05).")
                                    else:
                                        st.warning("The data significantly differs from this distribution (at alpha=0.05).")

                                    # Plot histogram with fitted PDF
                                    fig_dist, ax_dist = plt.subplots()
                                    ax_dist.hist(data_to_fit, bins=30, density=True, alpha=0.6, color=custom_color, label='Data Histogram')

                                    # Plot PDF of the best-fitting distribution
                                    xmin, xmax = plt.xlim()
                                    x = np.linspace(xmin, xmax, 100)
                                    p = best_distribution.pdf(x, *best_params)
                                    ax_dist.plot(x, p, 'k', linewidth=2, label=f'Fitted {best_distribution_name} PDF')

                                    ax_dist.set_title(f"Distribution Fit for '{dist_col}'")
                                    ax_dist.set_xlabel(dist_col)
                                    ax_dist.set_ylabel("Density")
                                    ax_dist.legend()
                                    st.pyplot(fig_dist)

                                else:
                                    st.warning("No distributions could be fitted successfully.")

                        except Exception as e:
                            st.error(f"Error during distribution fitting: {e}")
            else:
                st.info("Distribution fitting requires a numeric column.")

        # NEW FEATURE 29: Custom Theme Builder
        with st.expander("🖌️ Custom Theme Designer"):
            st.subheader("Create Your Custom Theme")
            
            col1, col2 = st.columns(2)
            with col1:
                primary_color = st.color_picker("Primary Color", "#38B2AC")      # Teal
                secondary_color = st.color_picker("Secondary Color", "#805AD5")  # Purple
                text_color = st.color_picker("Text Color", "#E2E8F0")            # Light Gray
            with col2:
                bg_color = st.color_picker("Background Color", "#1A202C")        # Very Dark Blue/Gray
                accent_color = st.color_picker("Accent Color", "#ED8936")        # Orange
                
            theme_name = st.text_input("Theme Name", "My Custom Theme")

            if st.button("Apply Custom Theme"):
                custom_css = f"""
                <style>
                .stApp {{
                    background-color: {bg_color};
                    color: {text_color};
                }}
                .metric-card {{
                    background: linear-gradient(135deg, {primary_color} 0%, {secondary_color} 100%);
                    color: {text_color}; /* Ensure text color contrasts with new gradient */
                }}
                .insight-box {{
                    background: #2D3748; /* Or a slightly lighter shade of bg_color */
                    border-left-color: {accent_color};
                    color: {text_color};
                }}
                .stButton > button {{
                    background-color: {accent_color};
                    color: {bg_color}; /* Text color for button, ensure contrast */
                    border: 1px solid {accent_color};
                }}
                div[data-testid="stExpander"] > div:first-child summary {{
                    color: {text_color};
                }}
                </style>
                """
                st.markdown(custom_css, unsafe_allow_html=True)
                st.success(f"Applied theme: {theme_name}")
                
                # Save theme
                theme_config = {
                    "name": theme_name,
                    "primary": primary_color,
                    "secondary": secondary_color,
                    "text": text_color,
                    "background": bg_color,
                    "accent": accent_color
                }
                st.download_button("Download Theme", 
                                 json.dumps(theme_config, indent=2),
                                 f"{theme_name.lower().replace(' ', '_')}_theme.json")

        # NEW FEATURE 32 (was 26): Automated Narrative Report Generation (Gemini-Enhanced)
        with st.expander("✍️ Automated Narrative Report Generation (Gemini-Enhanced)"):
            st.subheader("Generate Text Summaries of Your Findings")
            if gemini_api_key:
                report_elements_options = ["Overall Data Summary", "Key Trends (if time series analyzed)", "Top Correlations", "Anomaly Insights", "Cluster Profiles (if K-Means run)"]
                selected_report_elements = st.multiselect("Select Elements for Narrative Report", report_elements_options, default=report_elements_options[:2], key="narrative_elements")
                report_tone = st.selectbox("Select Report Tone", ["Formal", "Informal", "Technical"], key="narrative_tone")
                report_length = st.select_slider("Desired Report Length", options=["Brief", "Standard", "Detailed"], value="Standard", key="narrative_length")

                if st.button("Generate Narrative Report with AI", key="run_narrative_report"):
                    with st.spinner("AI is drafting your report..."):
                        # Construct prompt based on selected elements and available data/analyses
                        narrative_prompt = f"You are an expert data analyst. Based on the following dataset characteristics and potential analysis results, generate a {report_length}, {report_tone} narrative report. Focus on the following selected elements: {', '.join(selected_report_elements)}.\n\n"
                        narrative_prompt += f"Dataset Overview: {df.shape[0]} rows, {df.shape[1]} columns. Columns are: {', '.join(df.columns.tolist())}.\n"
                        if "Overall Data Summary" in selected_report_elements:
                            narrative_prompt += f"Key descriptive stats (first 3 numeric columns):\n{df[numeric_cols[:3]].describe().to_string()}\n\n"
                        # Add more context from other analyses if they were run (this part would need session_state or similar to track results)
                        # For now, this is a simplified version.
                        
                        try:
                            model_narrative = genai.GenerativeModel("gemini-2.0-flash") # or your preferred model
                            response_narrative = model_narrative.generate_content(narrative_prompt)
                            st.markdown("#### AI-Generated Narrative Report:")
                            st.markdown(response_narrative.text)
                            st.download_button("Download Narrative Report", response_narrative.text, file_name="ai_narrative_report.txt")
                        except Exception as e:
                            st.error(f"Gemini API Error for Narrative Report: {str(e)}")
            else:
                st.info("Enter your Gemini API key in the sidebar to enable AI-generated narrative reports.")

        # NEW FEATURE 33 (was 27): Interactive Outlier Explanation
        with st.expander("🕵️ Interactive Outlier Explanation"):
            st.subheader("Understand Why a Data Point is an Outlier")
            if 'anomalies_detected_df' in st.session_state and not st.session_state.anomalies_detected_df.empty: # Check if anomalies were detected
                outlier_df = st.session_state.anomalies_detected_df
                st.write("Previously detected outliers (from Anomaly Detection Dashboard):")
                st.dataframe(outlier_df.head())

                if not outlier_df.empty:
                    selected_outlier_index = st.selectbox("Select an Outlier Index to Explain", outlier_df.index.tolist(), key="select_outlier_explain")
                    if selected_outlier_index is not None:
                        outlier_data_point = df.loc[selected_outlier_index]
                        st.write(f"#### Explaining Outlier at Index: {selected_outlier_index}")
                        st.write(outlier_data_point)
                        
                        # Placeholder for explanation logic (could use SHAP/LIME if a model was involved, or simple feature comparison)
                        st.info("Explanation Feature (Conceptual): This outlier might be unusual due to...")
                        for col in numeric_cols: # Example: compare to mean
                            if col in outlier_data_point and pd.notna(outlier_data_point[col]):
                                if outlier_data_point[col] > df[col].mean() + 2 * df[col].std() or outlier_data_point[col] < df[col].mean() - 2 * df[col].std():
                                    st.write(f"- Its value for '{col}' ({outlier_data_point[col]:.2f}) is significantly different from the column mean ({df[col].mean():.2f}).")
            else:
                st.info("Run the 'Anomaly Detection Dashboard' first to identify outliers that can be explained.")

        # Auto-refresh functionality
        if refresh_interval > 0:
            time.sleep(refresh_interval)
            st.rerun()

        # Footer with session info
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"🕒 Session: {datetime.now().strftime('%H:%M:%S')}")
        with col2:
            st.info(f"🎨 Theme: {selected_theme}")
        with col3:
            st.info(f"📚 Datasets: {len(datasets)}")

else:
    st.info("📂 Please upload your data files to begin exploring! 🚀")
    
    # Sample data generator for testing
    if st.button("🎲 Generate Sample Data & Explore Features"):
        sample_data = {
            'Date': pd.date_range('2023-01-01', periods=100),
            'Sales': np.random.randint(100, 1000, 100),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
            'Temperature': np.random.normal(20, 5, 100),
            'Revenue': np.random.uniform(1000, 5000, 100).round(2)
        }
        df = pd.DataFrame(sample_data)
        st.success("Sample data generated! Use this to explore features.")
        st.dataframe(df.head())


with st.sidebar:
    st.image("d1.jpg",caption="Decode with design", use_container_width=True)


st.image("d4.jpg", caption="Data Analysis meets AI meets Elegance.", use_container_width=True)
