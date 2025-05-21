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
from sklearn.preprocessing import OneHotEncoder
import folium
from streamlit_folium import folium_static
import google.generativeai as genai
import io
import base64
import re
from datetime import datetime

# Page configuration
st.set_page_config(layout="wide", page_title="Advanced Dashboard Creator")

# Theme configuration
themes = {
    "light": {"bg": "#FFFFFF", "text": "#262730", "secondary": "#F0F2F6"},
    "dark": {"bg": "#0E1117", "text": "#FAFAFA", "secondary": "#1E1E1E"},
    "cyberpunk": {"bg": "#120458", "text": "#F7EE7F", "secondary": "#6320EE"}
}

# Sidebar: Global options
st.sidebar.header("📊 Dashboard Options")
selected_theme = st.sidebar.selectbox("🎨 Theme", options=list(themes.keys()), index=0)
color_scheme = st.sidebar.selectbox("🌈 Color Scheme", options=['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo', 'Blues'], index=0)
custom_color = st.sidebar.color_picker("🎨 Custom Color", "#1f77b4")

# Gemini API Integration
st.sidebar.header("🧠 AI Insights")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
gemini_model = "gemini-2.0-flash"
if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)
    except Exception as e:
        st.sidebar.error(f"API Error: {str(e)}")

# Main title
st.title("🚀 Advanced Automatic Dashboard Creator")
st.write("Upload a CSV or Excel file to generate an advanced interactive dashboard with AI-powered insights.")

# File uploader for CSV and Excel files
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error("Unsupported file format.")
            df = None
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        df = None

    if df is not None:
        # Data Preparation Section
        with st.expander("📥 Data Preparation", expanded=True):
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Missing data handling
            st.subheader("Missing Data Handling")
            cols_with_missing = df.columns[df.isna().any()].tolist()
            if cols_with_missing:
                for col in cols_with_missing:
                    method = st.selectbox(f"Handle missing values in '{col}'", 
                                         ["Keep as is", "Drop rows", "Fill with mean", "Fill with median", "Fill with zero", "Fill with custom value"])
                    if method == "Drop rows":
                        df = df.dropna(subset=[col])
                    elif method == "Fill with mean" and pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].mean())
                    elif method == "Fill with median" and pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].median())
                    elif method == "Fill with zero":
                        df[col] = df[col].fillna(0)
                    elif method == "Fill with custom value":
                        custom_val = st.text_input(f"Custom value for '{col}'")
                        if custom_val:
                            df[col] = df[col].fillna(custom_val)
            else:
                st.write("No missing values found in the dataset.")
            
            # Column type converter
            st.subheader("Column Type Converter")
            column_to_convert = st.selectbox("Select column to convert", options=df.columns)
            target_type = st.selectbox("Convert to type", options=["No conversion", "Numeric", "Text/Category", "Datetime"])
            
            if target_type != "No conversion":
                try:
                    if target_type == "Numeric":
                        df[column_to_convert] = pd.to_numeric(df[column_to_convert])
                    elif target_type == "Text/Category":
                        df[column_to_convert] = df[column_to_convert].astype(str)
                    elif target_type == "Datetime":
                        df[column_to_convert] = pd.to_datetime(df[column_to_convert])
                    st.success(f"Converted '{column_to_convert}' to {target_type}")
                except Exception as e:
                    st.error(f"Conversion error: {str(e)}")
            
            # Export cleaned data
            cleaned_csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Cleaned Dataset", data=cleaned_csv, file_name="cleaned_data.csv", mime="text/csv")

        # Identify column types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Detect date columns (attempt conversion)
        for col in df.columns:
            if col not in date_cols:
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_cols.append(col)
                    # If it was in categorical_cols, remove it
                    if col in categorical_cols:
                        categorical_cols.remove(col)
                except Exception:
                    continue

        # Check for potential geo data
        lat_cols = [col for col in df.columns if re.search(r'lat|latitude', col.lower())]
        lon_cols = [col for col in df.columns if re.search(r'lon|lng|longitude', col.lower())]
        has_geo_data = len(lat_cols) > 0 and len(lon_cols) > 0
        
        # Interactive chart builder
        with st.expander("📊 Interactive Chart Builder", expanded=True):
            st.subheader("Build Your Custom Chart")
            
            chart_type = st.selectbox("Select Chart Type", options=[
                "Bar Chart", "Line Chart", "Scatter Plot", "Histogram", 
                "Box Plot", "Pie Chart", "Treemap", "Heatmap","Area Chart", "Bubble Chart", "Radar Chart", "Donut Chart", "Violin Plot", "Waterfall Chart", "Sunburst Chart", "Sankey Diagram", "Parallel Coordinates Plot", "Candlestick Chart"

            ])
            
            col1, col2 = st.columns(2)
            
            with col1:
                if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot", "Box Plot", "Area Chart", "Bubble Chart", "Radar Chart"]:
                    x_axis = st.selectbox("X-Axis", options=df.columns)
                    y_axis = st.selectbox("Y-Axis", options=numeric_cols if numeric_cols else df.columns)
                    color_by = st.selectbox("Color By (Optional)", options=["None"] + categorical_cols)
                elif chart_type in ["Histogram", "Violin Plot"]:
                    x_axis = st.selectbox("Select Column", options=numeric_cols if numeric_cols else df.columns)
                    bins = st.slider("Number of Bins", min_value=5, max_value=100, value=20)
                elif chart_type in ["Pie Chart", "Treemap", "Donut Chart", "Sunburst Chart"]:
                    labels = st.selectbox("Labels", options=categorical_cols if categorical_cols else df.columns)
                    values = st.selectbox("Values", options=numeric_cols if numeric_cols else df.columns)
                elif chart_type == "Heatmap":
                    corr_method = st.selectbox("Correlation Method", options=["pearson", "kendall", "spearman"])
                elif chart_type == "Waterfall Chart":
                    categories = st.selectbox("Categories", options=categorical_cols if categorical_cols else df.columns)
                    values = st.selectbox("Values", options=numeric_cols if numeric_cols else df.columns)
                elif chart_type == "Sankey Diagram":
                    source = st.selectbox("Source", options=categorical_cols if categorical_cols else df.columns)
                    target = st.selectbox("Target", options=categorical_cols if categorical_cols else df.columns)
                    value = st.selectbox("Value", options=numeric_cols if numeric_cols else df.columns)
                elif chart_type == "Parallel Coordinates Plot":
                    dimensions = st.multiselect("Select Dimensions", options=numeric_cols if numeric_cols else df.columns)
                elif chart_type == "Candlestick Chart":
                    date_col = st.selectbox("Date Column", options=df.columns)
                    open_col = st.selectbox("Open", options=numeric_cols if numeric_cols else df.columns)
                    high_col = st.selectbox("High", options=numeric_cols if numeric_cols else df.columns)
                    low_col = st.selectbox("Low", options=numeric_cols if numeric_cols else df.columns)
                    close_col = st.selectbox("Close", options=numeric_cols if numeric_cols else df.columns)

            
            with col2:
                width = st.slider("Chart Width", min_value=400, max_value=1200, value=700)
                height = st.slider("Chart Height", min_value=300, max_value=800, value=500)
                log_scale = st.checkbox("Log Scale (Y-axis)")
            
            # Generate selected chart
            if chart_type == "Bar Chart":
                if color_by != "None":
                    fig = px.bar(df, x=x_axis, y=y_axis, color=color_by, width=width, height=height, 
                                log_y=log_scale, title=f"Bar Chart: {y_axis} by {x_axis}")
                else:
                    fig = px.bar(df, x=x_axis, y=y_axis, width=width, height=height, 
                                log_y=log_scale, title=f"Bar Chart: {y_axis} by {x_axis}")
                st.plotly_chart(fig)
            
            elif chart_type == "Line Chart":
                if color_by != "None":
                    fig = px.line(df, x=x_axis, y=y_axis, color=color_by, width=width, height=height, 
                                 log_y=log_scale, title=f"Line Chart: {y_axis} by {x_axis}")
                else:
                    fig = px.line(df, x=x_axis, y=y_axis, width=width, height=height, 
                                 log_y=log_scale, title=f"Line Chart: {y_axis} by {x_axis}")
                st.plotly_chart(fig)
            
            elif chart_type == "Scatter Plot":
                if color_by != "None":
                    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, width=width, height=height,
                                    log_y=log_scale, title=f"Scatter Plot: {y_axis} vs {x_axis}")
                else:
                    fig = px.scatter(df, x=x_axis, y=y_axis, width=width, height=height,
                                    log_y=log_scale, title=f"Scatter Plot: {y_axis} vs {x_axis}")
                st.plotly_chart(fig)
            
            elif chart_type == "Histogram":
                fig = px.histogram(df, x=x_axis, nbins=bins, width=width, height=height,
                                  title=f"Histogram of {x_axis}")
                st.plotly_chart(fig)
            
            elif chart_type == "Box Plot":
                fig = px.box(df, x=x_axis, y=y_axis, width=width, height=height,
                            log_y=log_scale, title=f"Box Plot: {y_axis} by {x_axis}")
                st.plotly_chart(fig)
            
            elif chart_type == "Pie Chart":
                fig = px.pie(df, names=labels, values=values, width=width, height=height,
                            title=f"Pie Chart: {values} by {labels}")
                st.plotly_chart(fig)
            
            elif chart_type == "Treemap":
                fig = px.treemap(df, path=[labels], values=values, width=width, height=height,
                                title=f"Treemap: {values} by {labels}")
                st.plotly_chart(fig)
                





            elif chart_type == "Area Chart":
                x_axis = st.selectbox("X-Axis", options=df.columns)
                y_axis = st.selectbox("Y-Axis", options=numeric_cols if numeric_cols else df.columns)
                color_by = st.selectbox("Color By (Optional)", options=["None"] + categorical_cols)

            elif chart_type == "Bubble Chart":
                x_axis = st.selectbox("X-Axis", options=numeric_cols)
                y_axis = st.selectbox("Y-Axis", options=numeric_cols)
                size = st.selectbox("Bubble Size", options=numeric_cols)
                color_by = st.selectbox("Color By (Optional)", options=["None"] + categorical_cols)

            elif chart_type == "Radar Chart":
                category_col = st.selectbox("Category Column", options=categorical_cols)
                value_cols = st.multiselect("Value Columns", options=numeric_cols)
                if len(value_cols) < 3:
                    st.warning("Select at least 3 value columns for a meaningful radar chart.")

            elif chart_type == "Donut Chart":
                labels = st.selectbox("Labels", options=categorical_cols if categorical_cols else df.columns)
                values = st.selectbox("Values", options=numeric_cols if numeric_cols else df.columns)
                hole_size = st.slider("Donut Hole Size", 0.1, 0.9, 0.5)

            elif chart_type == "Violin Plot":
                y_axis = st.selectbox("Y-Axis", options=numeric_cols)
                x_axis = st.selectbox("X-Axis (Optional)", options=["None"] + categorical_cols)
                color_by = st.selectbox("Color By (Optional)", options=["None"] + categorical_cols)

            elif chart_type == "Waterfall Chart":
                measure = st.selectbox("Measure Column", options=categorical_cols)
                values = st.selectbox("Values", options=numeric_cols)
                base = st.number_input("Initial Value", value=0)

            elif chart_type == "Sunburst Chart":
                path_cols = st.multiselect("Hierarchy Path (2+ Columns)", options=categorical_cols)
                values = st.selectbox("Values", options=numeric_cols if numeric_cols else df.columns)

            elif chart_type == "Sankey Diagram":
                source = st.selectbox("Source", options=categorical_cols)
                target = st.selectbox("Target", options=[col for col in categorical_cols if col != source])
                value = st.selectbox("Value", options=numeric_cols)

            elif chart_type == "Parallel Coordinates Plot":
                dimensions = st.multiselect("Select Dimensions", options=numeric_cols)
                color_by = st.selectbox("Color By (Optional)", options=["None"] + categorical_cols)

            elif chart_type == "Candlestick Chart":
                date_col = st.selectbox("Date Column", options=df.columns)
                open_col = st.selectbox("Open", options=numeric_cols)
                high_col = st.selectbox("High", options=numeric_cols)
                low_col = st.selectbox("Low", options=numeric_cols)
                close_col = st.selectbox("Close", options=numeric_cols)

    
            elif chart_type == "Heatmap":
                if numeric_cols and len(numeric_cols) > 1:
                    corr = df[numeric_cols].corr(method=corr_method)
                    fig = px.imshow(corr, text_auto=True, aspect="auto", width=width, height=height,
                                   title=f"Correlation Heatmap ({corr_method})")
                    st.plotly_chart(fig)
                else:
                    st.error("Need at least 2 numeric columns for correlation heatmap")

        # Geospatial visualization
        if has_geo_data:
            with st.expander("🗺️ Geospatial Visualization", expanded=True):
                st.subheader("Geo Map")
                lat_col = st.selectbox("Latitude Column", options=lat_cols)
                lon_col = st.selectbox("Longitude Column", options=lon_cols)
                
                # Filter out rows with missing lat/lon
                geo_df = df.dropna(subset=[lat_col, lon_col])
                
                # Get center of map
                center_lat = geo_df[lat_col].mean()
                center_lon = geo_df[lon_col].mean()
                
                m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
                
                # Add points
                color_col = st.selectbox("Color By", options=["None"] + categorical_cols)
                size_col = st.selectbox("Size By", options=["Constant"] + numeric_cols)
                
                if color_col != "None" and size_col != "Constant":
                    for _, row in geo_df.iterrows():
                        folium.Circle(
                            location=[row[lat_col], row[lon_col]],
                            radius=float(row[size_col]) if pd.notna(row[size_col]) else 10,
                            color=custom_color,
                            fill=True,
                            popup=f"{color_col}: {row[color_col]}, {size_col}: {row[size_col]}"
                        ).add_to(m)
                else:
                    for _, row in geo_df.iterrows():
                        folium.Marker(
                            location=[row[lat_col], row[lon_col]],
                            popup=f"Lat: {row[lat_col]}, Lon: {row[lon_col]}"
                        ).add_to(m)
                
                folium_static(m)

        # ML & Analytics Integration
        with st.expander("🧠 Machine Learning & Analytics", expanded=True):
            ml_analysis = st.selectbox("Select Analysis Type", 
                                      ["Clustering (K-Means)", "Anomaly Detection", "One-Hot Encoding"])
            
            if ml_analysis == "Clustering (K-Means)" and len(numeric_cols) >= 2:
                st.subheader("K-Means Clustering")
                col1, col2 = st.columns(2)
                
                with col1:
                    feature1 = st.selectbox("Feature 1", options=numeric_cols, index=0)
                    feature2 = st.selectbox("Feature 2", options=numeric_cols, index=min(1, len(numeric_cols)-1))
                    n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)
                
                # Perform clustering
                features = df[[feature1, feature2]].dropna()
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                df_cluster = features.copy()
                df_cluster['cluster'] = kmeans.fit_predict(features)
                
                # Visualize clusters
                fig = px.scatter(df_cluster, x=feature1, y=feature2, color='cluster', 
                                title=f"K-Means Clustering ({n_clusters} clusters)")
                st.plotly_chart(fig)
                
                # Show cluster centers
                centers = pd.DataFrame(kmeans.cluster_centers_, columns=[feature1, feature2])
                centers['cluster'] = range(n_clusters)
                st.write("Cluster Centers:")
                st.dataframe(centers)
            

            
            elif ml_analysis == "Anomaly Detection" and numeric_cols:
                st.subheader("Anomaly Detection (Isolation Forest)")
                
                # Select features for anomaly detection
                features_for_anomaly = st.multiselect("Select Features for Anomaly Detection", 
                                                     options=numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))])
                
                contamination = st.slider("Contamination (expected % of outliers)", 
                                         min_value=0.01, max_value=0.5, value=0.05, step=0.01)
                
                if features_for_anomaly:
                    # Prepare data
                    anomaly_data = df[features_for_anomaly].dropna()
                    
                    # Train isolation forest
                    iso_forest = IsolationForest(contamination=contamination, random_state=42)
                    anomalies = iso_forest.fit_predict(anomaly_data)
                    anomaly_data['anomaly'] = anomalies
                    anomaly_data['is_anomaly'] = anomaly_data['anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
                    
                    # Display results
                    if len(features_for_anomaly) >= 2:
                        # Select 2 features for visualization
                        feat1, feat2 = features_for_anomaly[:2]
                        fig = px.scatter(anomaly_data, x=feat1, y=feat2, color='is_anomaly',
                                        color_discrete_map={'Anomaly': 'red', 'Normal': 'blue'},
                                        title=f"Anomaly Detection: {feat1} vs {feat2}")
                        st.plotly_chart(fig)
                    
                    # Display anomaly summary
                    anomaly_count = (anomalies == -1).sum()
                    st.write(f"Detected {anomaly_count} anomalies out of {len(anomaly_data)} records ({anomaly_count/len(anomaly_data)*100:.2f}%)")
                    
                    # Show anomalies
                    if st.checkbox("Show Anomaly Records"):
                        st.dataframe(df.iloc[np.where(anomalies == -1)[0]])
            
            elif ml_analysis == "One-Hot Encoding" and categorical_cols:
                st.subheader("One-Hot Encoding Preview")
                categorical_for_encoding = st.multiselect("Select Categorical Columns for Encoding", 
                                                         options=categorical_cols, 
                                                         default=categorical_cols[:min(2, len(categorical_cols))])
                
                if categorical_for_encoding:
                    # Apply one-hot encoding
                    encoder = OneHotEncoder(sparse_output=False, drop='first')
                    encoded_data = encoder.fit_transform(df[categorical_for_encoding])
                    
                    # Create DataFrame with encoded columns
                    feature_names = []
                    for i, col in enumerate(categorical_for_encoding):
                        categories = encoder.categories_[i][1:]  # Drop first category
                        feature_names.extend([f"{col}_{cat}" for cat in categories])
                    
                    encoded_df = pd.DataFrame(encoded_data, columns=feature_names)
                    
                    # Display encoded data
                    st.write("One-Hot Encoded Preview:")
                    st.dataframe(encoded_df.head(10))
                    
                    # Download option
                    csv_encoded = encoded_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Encoded Data", data=csv_encoded, 
                                      file_name="one_hot_encoded.csv", mime="text/csv")
            else:
                st.write("Not enough appropriate columns for the selected analysis.")

        # Explainability & Insights with Gemini API
        with st.expander("🔍 AI-Powered Insights", expanded=True):
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
                        model = genai.GenerativeModel(gemini_model)
                        response = model.generate_content(prompt)
                        st.write("AI Response:")
                        st.write(response.text)
                    except Exception as e:
                        st.error(f"Gemini API Error: {str(e)}")
            else:
                st.info("Enter your Gemini API key in the sidebar to enable AI insights.")
                
            # Generate automated correlation insights
            if numeric_cols and len(numeric_cols) > 1:
                st.subheader("Automated Correlation Insights")
                corr_matrix = df[numeric_cols].corr()
                
                # Find strongest correlations
                corr_pairs = []
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        corr_value = corr_matrix.iloc[i, j]
                        corr_pairs.append((numeric_cols[i], numeric_cols[j], corr_value))
                
                # Sort by absolute correlation value
                corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                
                # Display top correlations
                if corr_pairs:
                    st.write("Top correlations found:")
                    for col1, col2, corr_val in corr_pairs[:5]:  # Show top 5
                        correlation_type = "positive" if corr_val > 0 else "negative"
                        strength = "strong" if abs(corr_val) > 0.7 else "moderate" if abs(corr_val) > 0.3 else "weak"
                        st.write(f"- {strength.title()} {correlation_type} correlation between '{col1}' and '{col2}': {corr_val:.2f}")
                else:
                    st.write("No correlations to display.")

        # Save dashboard configuration
        config = {
            "theme": selected_theme,
            "color_scheme": color_scheme,
            "custom_color": custom_color,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "date_columns": date_cols,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Generate session ID for saving/loading
        session_id = base64.b64encode(json.dumps(config).encode()).decode()
        
        st.sidebar.markdown("### 💾 Save/Load Session")
        st.sidebar.download_button(
            "Download Configuration", 
            data=json.dumps(config, indent=4), 
            file_name="dashboard_config.json", 
            mime="application/json"
        )
        
        st.sidebar.markdown(f"Session ID: `{session_id[:10]}...`")
        st.sidebar.markdown("Copy this ID to restore your session later")
        
        # Restore session option
        restore_id = st.sidebar.text_input("Restore Session ID")
        if restore_id and st.sidebar.button("Restore Session"):
            try:
                restored_config = json.loads(base64.b64decode(restore_id))
                st.success("Session restored successfully!")
                st.write(restored_config)
            except Exception as e:
                st.error(f"Error restoring session: {str(e)}")

        # Footer
        st.markdown("---")
        st.markdown("Advanced Dashboard Creator | Created with Streamlit")
        st.markdown(f"Using theme: {selected_theme} | Color scheme: {color_scheme}")
