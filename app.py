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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import folium
from streamlit_folium import folium_static
import google.generativeai as genai
import io
import base64
import re
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(layout="wide", page_title="Advanced Dashboard Creator", page_icon="📊")

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 10px 0;
}
.insight-box {
    background: #f0f2f6;
    padding: 15px;
    border-left: 5px solid #667eea;
    border-radius: 5px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("📊 Dashboard Options")
selected_theme = st.sidebar.selectbox("🎨 Theme", options=['light', 'dark', 'cyberpunk'], index=0)
color_scheme = st.sidebar.selectbox("🌈 Color Scheme", options=['viridis', 'plasma', 'inferno', 'magma'], index=0)
custom_color = st.sidebar.color_picker("🎨 Custom Color", "#1f77b4")

# NEW FEATURE 1: Real-time data refresh
refresh_interval = st.sidebar.slider("⚡ Auto-refresh (seconds)", 0, 60, 0)
if refresh_interval > 0:
    st.sidebar.info(f"Dashboard will refresh every {refresh_interval} seconds")

# Gemini API Integration
st.sidebar.header("🧠 AI Insights")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)
    except Exception as e:
        st.sidebar.error(f"API Error: {str(e)}")

# Main title with metrics
st.title("🚀 Enhanced Dashboard Creator")
st.write("Upload a file to generate an advanced dashboard with 10+ new features")

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
        df = datasets[selected_dataset]
        
        # NEW FEATURE 4: Advanced data profiling
        with st.expander("📈 Advanced Data Profiling", expanded=True):
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
            
            # Data quality score
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            quality_score = max(0, 100 - missing_pct - (duplicates/df.shape[0]*10))
            st.markdown(f'<div class="insight-box"><strong>Data Quality Score: {quality_score:.1f}/100</strong><br>Based on missing values and duplicate records</div>', unsafe_allow_html=True)

        # Identify column types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # NEW FEATURE 5: Smart data type detection and conversion
        with st.expander("🔍 Smart Data Type Detection"):
            st.subheader("Suggested Data Type Conversions")
            suggestions = []
            
            for col in df.columns:
                if col in categorical_cols:
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
                    if st.checkbox(f"Convert '{col}' to {suggested_type} ({reason})"):
                        try:
                            if suggested_type == "numeric":
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                                numeric_cols.append(col)
                                categorical_cols.remove(col)
                            elif suggested_type == "datetime":
                                df[col] = pd.to_datetime(df[col], errors='coerce')
                            st.success(f"Converted {col} to {suggested_type}")
                        except Exception as e:
                            st.error(f"Conversion failed: {str(e)}")
            else:
                st.info("No conversion suggestions found")

        # NEW FEATURE 6: Advanced filtering system
        with st.expander("🔧 Advanced Data Filtering"):
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
            
            st.info(f"Filtered dataset: {df.shape[0]} rows × {df.shape[1]} columns")

        # NEW FEATURE 7: Automated statistical testing
        with st.expander("📊 Statistical Analysis Suite"):
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
        with st.expander("🤖 AutoML Pipeline"):
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
                        
                        # Model selection
                        models = {
                            "Linear Regression": LinearRegression(),
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
            with st.expander("⚖️ Dataset Comparison Dashboard"):
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
        with st.expander("📋 Advanced Export & Reporting"):
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

        # Interactive visualization builder
        with st.expander("📊 Quick Visualization Builder"):
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

        # Auto-refresh functionality
        if refresh_interval > 0:
            time.sleep(refresh_interval)
            st.rerun()

        # Footer with session info
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"Session: {datetime.now().strftime('%H:%M:%S')}")
        with col2:
            st.info(f"Theme: {selected_theme}")
        with col3:
            st.info(f"Datasets: {len(datasets)}")

else:
    st.info("👆 Upload files to start creating your dashboard")
    
    # Sample data generator for testing
    if st.button("🎲 Generate Sample Data"):
        sample_data = {
            'Date': pd.date_range('2023-01-01', periods=100),
            'Sales': np.random.randint(100, 1000, 100),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
            'Temperature': np.random.normal(20, 5, 100),
            'Revenue': np.random.randint(1000, 5000, 100)
        }
        df = pd.DataFrame(sample_data)
        st.success("Sample data generated! Use this to explore features.")
        st.dataframe(df.head())
