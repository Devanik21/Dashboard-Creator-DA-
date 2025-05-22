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
import time # Import the time module
import warnings # Import locally to keep dependencies clear
from scipy import stats
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

        # AI-Powered Insights with Gemini API
        with st.expander("🧠 AI-Powered Insights", expanded=True):
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

        # NEW FEATURE 11: Anomaly Detection Dashboard
        with st.expander("🎯 Anomaly Detection Dashboard"):
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
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_cols.append(col)
                except: pass
        
        if date_cols and numeric_cols:
            with st.expander("📊 Time Series Analysis"):
                st.subheader("Trend Analysis & Forecasting")
                date_col = st.selectbox("Date Column", date_cols)
                value_col = st.selectbox("Value Column", numeric_cols)
                
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
        with st.expander("🔗 Data Relationship Mapper"):
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
            with st.expander("📈 A/B Testing Suite", expanded=False): # Keep it collapsed by default
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
                    
        # NEW FEATURE 15: Custom Theme Builder
        with st.expander("🎨 Custom Theme Builder"):
            st.subheader("Create Your Custom Theme")
            
            col1, col2 = st.columns(2)
            with col1:
                primary_color = st.color_picker("Primary Color", "#667eea")
                secondary_color = st.color_picker("Secondary Color", "#764ba2") 
                text_color = st.color_picker("Text Color", "#262730")
            with col2:
                bg_color = st.color_picker("Background Color", "#ffffff")
                accent_color = st.color_picker("Accent Color", "#f39c12")
                
            theme_name = st.text_input("Theme Name", "My Custom Theme")
            
            if st.button("Apply Custom Theme"):
                custom_css = f"""
                <style>
                .stApp {{
                    background-color: {bg_color};
                    color: {text_color};
                }}
                .metric-card {{
                    background: linear-gradient(45deg, {primary_color} 0%, {secondary_color} 100%);
                }}
                .stButton > button {{
                    background-color: {accent_color};
                    color: white;
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
