import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import IsolationForest
import json
import io
import base64
import datetime
import requests
from wordcloud import WordCloud
import missingno as msno
import networkx as nx

st.set_page_config(layout="wide")
st.title("Advanced Automatic Dashboard Creator")
st.write("Upload a CSV or Excel file to automatically generate an advanced dashboard with multiple visualization options.")

# Sidebar: Global options
st.sidebar.header("Dashboard Options")
color_scheme = st.sidebar.selectbox("Select Color Scheme", options=['viridis', 'plasma', 'inferno', 'magma', 'cividis'], index=0)
theme = st.sidebar.selectbox("UI Theme", ["Light", "Dark"], index=0)
if theme == "Dark":
    st.markdown("""
    <style>
    .stApp {background-color: #121212; color: white;}
    .stMarkdown {color: white;}
    </style>
    """, unsafe_allow_html=True)

# Dashboard sections toggle
sections = {
    "Data Preview": True,
    "Data Summary": True,
    "Missing Data Analysis": True,
    "Univariate Analysis": True,
    "Bivariate Analysis": True, 
    "Multivariate Analysis": True,
    "Time Series Analysis": True,
    "Advanced Analytics": True,
    "Reporting": True
}
with st.sidebar.expander("Show/Hide Sections"):
    for section, default in sections.items():
        sections[section] = st.checkbox(section, value=default)

# Helper functions
def get_download_link(object_to_download, download_filename, download_link_text):
    """Generate a download link for a file"""
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def plot_network(df, source_col, target_col, weight_col=None):
    """Create a network graph from dataframe columns"""
    G = nx.Graph()
    for _, row in df.iterrows():
        source = str(row[source_col])
        target = str(row[target_col])
        if source and target:  # Skip rows with empty values
            weight = float(row[weight_col]) if weight_col else 1.0
            G.add_edge(source, target, weight=weight)
    
    # Create a plotly figure
    pos = nx.spring_layout(G)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'), hoverinfo='none'
    )
    
    node_x, node_y = [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()),
        marker=dict(showscale=True, colorscale='YlGnBu', size=10, color=[])
    )
    
    # Color nodes by degree
    node_adjacencies = []
    for node in G.nodes():
        node_adjacencies.append(len(list(G.neighbors(node))))
    node_trace.marker.color = node_adjacencies
    
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    showlegend=False, hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=0),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

# File uploader for CSV and Excel files
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Add support for more file import options
        file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": f"{uploaded_file.size / 1024:.2f} KB"}
        st.write(file_details)
        
        import_options = st.expander("Import Options")
        with import_options:
            sep = st.text_input("Separator (for CSV)", ",")
            encoding = st.selectbox("Encoding", ["utf-8", "latin-1", "iso-8859-1", "cp1252"])
            na_values = st.text_input("NA Values (comma separated)", "NA,N/A,null")
            
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, sep=sep, encoding=encoding, na_values=na_values.split(','))
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            else:
                st.error("Unsupported file format.")
                df = None
        except Exception as e:
            st.error(f"Error reading file: {e}")
            df = None

        if df is not None:
            # Save a copy for comparison after transformations
            original_df = df.copy()
            
            # Data Preprocessing Options
            with st.expander("Data Preprocessing"):
                # Handle missing values
                if df.isna().any().any():
                    missing_strategy = st.selectbox("Handle Missing Values", 
                                                  ["Keep as is", "Drop rows", "Drop columns", "Fill with mean/mode", "Fill with median", "Fill with zero"])
                    if missing_strategy == "Drop rows":
                        df = df.dropna()
                        st.info(f"Dropped {len(original_df) - len(df)} rows with missing values")
                    elif missing_strategy == "Drop columns":
                        df = df.dropna(axis=1)
                        st.info(f"Dropped {len(original_df.columns) - len(df.columns)} columns with missing values")
                    elif missing_strategy == "Fill with mean/mode":
                        for col in df.columns:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                df[col] = df[col].fillna(df[col].mean())
                            else:
                                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
                    elif missing_strategy == "Fill with median":
                        for col in df.columns:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                df[col] = df[col].fillna(df[col].median())
                            else:
                                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
                    elif missing_strategy == "Fill with zero":
                        df = df.fillna(0)
                
                # Feature engineering options
                if st.checkbox("Enable Feature Engineering"):
                    # Date extraction
                    date_cols_fe = st.multiselect("Extract components from date columns", 
                                               df.select_dtypes(include=['datetime64']).columns.tolist())
                    for date_col in date_cols_fe:
                        df[f"{date_col}_year"] = df[date_col].dt.year
                        df[f"{date_col}_month"] = df[date_col].dt.month
                        df[f"{date_col}_day"] = df[date_col].dt.day
                        df[f"{date_col}_dayofweek"] = df[date_col].dt.dayofweek
                    
                    # Text length
                    text_cols = st.multiselect("Calculate text length for columns", 
                                             df.select_dtypes(include=['object']).columns.tolist())
                    for text_col in text_cols:
                        df[f"{text_col}_length"] = df[text_col].astype(str).apply(len)
                    
                    # Binning numeric columns
                    num_cols_bin = st.multiselect("Bin numeric columns", 
                                                df.select_dtypes(include=['number']).columns.tolist())
                    for num_col in num_cols_bin:
                        num_bins = st.slider(f"Number of bins for {num_col}", 2, 10, 5)
                        df[f"{num_col}_binned"] = pd.qcut(df[num_col], q=num_bins, duplicates='drop', labels=False)

            # Identify column types
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Detect date columns
            date_cols = []
            for col in df.columns:
                if df[col].dtype == 'datetime64[ns]':
                    date_cols.append(col)
                else:
                    try:
                        pd.to_datetime(df[col])
                        date_cols.append(col)
                        df[col] = pd.to_datetime(df[col])
                    except Exception:
                        continue
            
            # DATA PREVIEW SECTION
            if sections["Data Preview"]:
                st.markdown("---")
                st.subheader("Data Preview")
                
                preview_tabs = st.tabs(["Head", "Sample", "Columns", "Data Types"])
                with preview_tabs[0]:
                    st.write(df.head())
                with preview_tabs[1]:
                    st.write(df.sample(min(5, len(df))))
                with preview_tabs[2]:
                    st.write(pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes,
                        'Non-Null Count': df.count(),
                        'Null Count': df.isna().sum(),
                        'Unique Values': [df[col].nunique() for col in df.columns]
                    }))
                with preview_tabs[3]:
                    st.json({col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)})
            
            # DATA SUMMARY SECTION
            if sections["Data Summary"]:
                st.markdown("---")
                st.subheader("Data Summary")
                
                summary_tabs = st.tabs(["Statistics", "Correlation", "Metadata"])
                with summary_tabs[0]:
                    st.write(df.describe(include='all'))
                with summary_tabs[1]:
                    if numeric_cols:
                        corr = df[numeric_cols].corr()
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(corr, annot=True, cmap=color_scheme, ax=ax)
                        st.pyplot(fig)
                with summary_tabs[2]:
                    meta = {
                        "Rows": len(df),
                        "Columns": len(df.columns),
                        "Numeric Columns": len(numeric_cols),
                        "Categorical Columns": len(categorical_cols),
                        "Date Columns": len(date_cols),
                        "Missing Values": df.isna().sum().sum(),
                        "Memory Usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
                    }
                    st.json(meta)
            
            # MISSING DATA ANALYSIS
            if sections["Missing Data Analysis"] and df.isna().any().any():
                st.markdown("---")
                st.subheader("Missing Data Analysis")
                
                # Missing heatmap using missingno
                plt.figure(figsize=(10, 6))
                msno_matrix = msno.matrix(df)
                st.pyplot(msno_matrix.figure)
                
                # Missing bar chart
                missing_df = pd.DataFrame({
                    'Column': df.columns,
                    'Missing Count': df.isna().sum(),
                    'Missing Percentage': df.isna().sum() / len(df) * 100
                }).sort_values('Missing Percentage', ascending=False)
                
                fig = px.bar(missing_df, x='Column', y='Missing Percentage',
                          text=missing_df['Missing Count'],
                          title="Missing Values by Column")
                st.plotly_chart(fig)
            
            # UNIVARIATE ANALYSIS SECTION
            if sections["Univariate Analysis"]:
                st.markdown("---") 
                st.subheader("Univariate Analysis")
                
                # Enhanced visualizations with tabs for each type
                if numeric_cols:
                    st.write("### Numeric Columns")
                    selected_num_col = st.selectbox("Select Numeric Column", numeric_cols)
                    
                    num_tabs = st.tabs(["Histogram", "Box Plot", "Violin Plot", "KDE"])
                    with num_tabs[0]:
                        fig = px.histogram(df, x=selected_num_col, marginal="rug", 
                                        color_discrete_sequence=px.colors.sequential.Viridis)
                        st.plotly_chart(fig)
                    with num_tabs[1]:
                        fig = px.box(df, y=selected_num_col)
                        st.plotly_chart(fig)
                    with num_tabs[2]:
                        fig = px.violin(df, y=selected_num_col, box=True, points="all")
                        st.plotly_chart(fig)
                    with num_tabs[3]:
                        fig, ax = plt.subplots()
                        sns.kdeplot(df[selected_num_col].dropna(), fill=True, ax=ax)
                        st.pyplot(fig)
                
                if categorical_cols:
                    st.write("### Categorical Columns")
                    selected_cat_col = st.selectbox("Select Categorical Column", categorical_cols)
                    
                    # For large cardinality, only show top values
                    n_unique = df[selected_cat_col].nunique()
                    top_n = st.slider("Show top N categories", 5, min(50, n_unique), min(20, n_unique))
                    top_cats = df[selected_cat_col].value_counts().nlargest(top_n).index
                    filtered_df = df[df[selected_cat_col].isin(top_cats)]
                    
                    cat_tabs = st.tabs(["Bar Chart", "Pie Chart", "Word Cloud"])
                    with cat_tabs[0]:
                        fig = px.bar(filtered_df[selected_cat_col].value_counts().reset_index(),
                                  x='index', y=selected_cat_col, title=f"Distribution of {selected_cat_col}")
                        st.plotly_chart(fig)
                    with cat_tabs[1]:
                        fig = px.pie(filtered_df, names=selected_cat_col)
                        st.plotly_chart(fig)
                    with cat_tabs[2]:
                        # Generate wordcloud from categorical column
                        text = " ".join(df[selected_cat_col].dropna().astype(str))
                        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
            
            # BIVARIATE ANALYSIS SECTION
            if sections["Bivariate Analysis"]:
                st.markdown("---")
                st.subheader("Bivariate Analysis")
                
                bi_vis_type = st.selectbox("Select Plot Type", 
                                        ["Scatter Plot", "Grouped Bar Chart", "Heatmap", "Line Plot"])
                
                if bi_vis_type == "Scatter Plot" and len(numeric_cols) >= 2:
                    x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
                    y_col = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
                    color_col = st.selectbox("Color by", ["None"] + categorical_cols, key="scatter_color")
                    
                    if color_col != "None":
                        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{x_col} vs {y_col} by {color_col}")
                    else:
                        fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                    
                    fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
                    st.plotly_chart(fig)
                    
                    # Add regression line option
                    if st.checkbox("Add regression line", key="add_regression"):
                        fig = px.scatter(df, x=x_col, y=y_col, trendline="ols", 
                                      title=f"{x_col} vs {y_col} with trend line")
                        st.plotly_chart(fig)
                
                elif bi_vis_type == "Grouped Bar Chart" and categorical_cols:
                    bar_x = st.selectbox("X-axis (Category)", categorical_cols, key="bar_x")
                    bar_y = st.selectbox("Y-axis (Value)", numeric_cols, key="bar_y")
                    bar_color = st.selectbox("Group by", ["None"] + categorical_cols, key="bar_color")
                    
                    # Ensure we don't try to group by the same column
                    if bar_color != "None" and bar_color != bar_x:
                        agg_df = df.groupby([bar_x, bar_color])[bar_y].mean().reset_index()
                        fig = px.bar(agg_df, x=bar_x, y=bar_y, color=bar_color, 
                                  title=f"Average {bar_y} by {bar_x} and {bar_color}", barmode="group")
                    else:
                        agg_df = df.groupby(bar_x)[bar_y].mean().reset_index()
                        fig = px.bar(agg_df, x=bar_x, y=bar_y, title=f"Average {bar_y} by {bar_x}")
                    
                    st.plotly_chart(fig)
                
                elif bi_vis_type == "Heatmap" and categorical_cols and len(categorical_cols) >= 2:
                    heat_x = st.selectbox("X-axis", categorical_cols, key="heat_x")
                    heat_y = st.selectbox("Y-axis", categorical_cols, key="heat_y")
                    heat_val = st.selectbox("Value", ["Count"] + numeric_cols, key="heat_val")
                    
                    if heat_x != heat_y:
                        if heat_val == "Count":
                            pivot_table = pd.crosstab(df[heat_y], df[heat_x])
                        else:
                            pivot_table = pd.pivot_table(df, values=heat_val, index=heat_y, 
                                                      columns=heat_x, aggfunc='mean')
                        
                        fig = px.imshow(pivot_table, title=f"Heatmap of {heat_y} vs {heat_x}")
                        st.plotly_chart(fig)
                
                elif bi_vis_type == "Line Plot" and date_cols:
                    line_x = st.selectbox("X-axis (Date)", date_cols, key="line_x")
                    line_y = st.selectbox("Y-axis (Value)", numeric_cols, key="line_y")
                    line_color = st.selectbox("Group by", ["None"] + categorical_cols, key="line_color")
                    
                    if line_color != "None":
                        df_grouped = df.groupby([pd.Grouper(key=line_x, freq='D'), line_color])[line_y].mean().reset_index()
                        fig = px.line(df_grouped, x=line_x, y=line_y, color=line_color, 
                                   title=f"{line_y} over time by {line_color}")
                    else:
                        df_grouped = df.groupby(pd.Grouper(key=line_x, freq='D'))[line_y].mean().reset_index()
                        fig = px.line(df_grouped, x=line_x, y=line_y, title=f"{line_y} over time")
                    
                    st.plotly_chart(fig)
            
            # MULTIVARIATE ANALYSIS SECTION
            if sections["Multivariate Analysis"] and len(numeric_cols) >= 3:
                st.markdown("---")
                st.subheader("Multivariate Analysis")
                
                multi_tabs = st.tabs(["3D Scatter", "Parallel Coordinates", "Radar Chart", "Network Graph"])
                
                with multi_tabs[0]:
                    if len(numeric_cols) >= 3:
                        x3d = st.selectbox("X-axis", numeric_cols, key="3d_x")
                        y3d = st.selectbox("Y-axis", numeric_cols, key="3d_y", index=min(1, len(numeric_cols)-1))
                        z3d = st.selectbox("Z-axis", numeric_cols, key="3d_z", index=min(2, len(numeric_cols)-1))
                        color3d = st.selectbox("Color by", ["None"] + categorical_cols + numeric_cols, key="3d_color")
                        
                        if color3d != "None":
                            fig = px.scatter_3d(df, x=x3d, y=y3d, z=z3d, color=color3d)
                        else:
                            fig = px.scatter_3d(df, x=x3d, y=y3d, z=z3d)
                        
                        st.plotly_chart(fig)
                
                with multi_tabs[1]:
                    # Parallel coordinates
                    para_cols = st.multiselect("Select columns for parallel coordinates", 
                                            numeric_cols, default=numeric_cols[:5])
                    para_color = st.selectbox("Color by", ["None"] + categorical_cols + numeric_cols, key="para_color")
                    
                    if para_cols:
                        if para_color != "None":
                            fig = px.parallel_coordinates(df, dimensions=para_cols, color=para_color)
                        else:
                            fig = px.parallel_coordinates(df, dimensions=para_cols)
                        st.plotly_chart(fig)
                
                with multi_tabs[2]:
                    # Radar chart
                    if categorical_cols:
                        radar_cat = st.selectbox("Category for radar chart", categorical_cols)
                        radar_vals = st.multiselect("Values for radar chart", numeric_cols, default=numeric_cols[:5])
                        
                        if radar_vals:
                            # Get the mean values for each category
                            radar_df = df.groupby(radar_cat)[radar_vals].mean().reset_index()
                            
                            # Prepare data for radar chart
                            fig = go.Figure()
                            
                            for i, category in enumerate(radar_df[radar_cat].unique()):
                                cat_data = radar_df[radar_df[radar_cat] == category]
                                fig.add_trace(go.Scatterpolar(
                                    r=cat_data[radar_vals].values.flatten(),
                                    theta=radar_vals,
                                    fill='toself',
                                    name=str(category)
                                ))
                            
                            fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                                           showlegend=True)
                            st.plotly_chart(fig)
                
                with multi_tabs[3]:
                    # Network graph between two categorical variables
                    if len(categorical_cols) >= 2:
                        source_col = st.selectbox("Source column", categorical_cols, key="net_source")
                        target_col = st.selectbox("Target column", categorical_cols, key="net_target", 
                                               index=min(1, len(categorical_cols)-1))
                        weight_col = st.selectbox("Weight column (optional)", ["None"] + numeric_cols, key="net_weight")
                        
                        if source_col != target_col:
                            weight = None if weight_col == "None" else weight_col
                            network_fig = plot_network(df, source_col, target_col, weight)
                            st.plotly_chart(network_fig)
            
            # TIME SERIES ANALYSIS SECTION
            if sections["Time Series Analysis"] and date_cols:
                st.markdown("---")
                st.subheader("Time Series Analysis")
                
                # Enhanced time series options
                ts_date = st.selectbox("Select Date Column", date_cols, key="ts_date")
                ts_value = st.selectbox("Select Value Column", numeric_cols, key="ts_value")
                
                # Resample options
                resample_options = {
                    "None": None, "Daily": "D", "Weekly": "W", "Monthly": "M", 
                    "Quarterly": "Q", "Yearly": "Y"
                }
                resample_freq = st.selectbox("Resample Frequency", list(resample_options.keys()), key="resample")
                
                if resample_freq != "None":
                    # Ensure df is sorted by date
                    df_ts = df.sort_values(by=ts_date)
                    df_ts = df_ts.set_index(ts_date)
                    
                    # Resample and plot
                    df_resampled = df_ts[ts_value].resample(resample_options[resample_freq]).mean().reset_index()
                    
                    fig = px.line(df_resampled, x=ts_date, y=ts_value, 
                               title=f"{ts_value} over time ({resample_freq} frequency)")
                    st.plotly_chart(fig)
                    
                    # Time series decomposition
                    if st.checkbox("Show Time Series Decomposition"):
                        try:
                            # Fill any missing values for decomposition
                            ts_filled = df_ts[ts_value].fillna(method='ffill').fillna(method='bfill')
                            
                            # Make sure we have enough data points
                            if len(ts_filled) >= 4:  # Minimal requirement for decomposition
                                result = seasonal_decompose(ts_filled, model='additive', 
                                                         period=min(4, len(ts_filled)//2))
                                
                                # Plot the decomposition
                                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
                                result.observed.plot(ax=ax1)
                                ax1.set_title('Observed')
                                result.trend.plot(ax=ax2)
                                ax2.set_title('Trend')
                                result.seasonal.plot(ax=ax3)
                                ax3.set_title('Seasonal')
                                result.resid.plot(ax=ax4)
                                ax4.set_title('Residual')
                                fig.tight_layout()
                                st.pyplot(fig)
                            else:
                                st.error("Not enough data points for time series decomposition")
                        except Exception as e:
                            st.error(f"Error in time series decomposition: {e}")
            
            # ADVANCED ANALYTICS SECTION
            if sections["Advanced Analytics"] and len(numeric_cols) >= 2:
                st.markdown("---")
                st.subheader("Advanced Analytics")
                
                adv_tabs = st.tabs(["Clustering", "PCA", "Anomaly Detection", "What-If Analysis"])
                
                with adv_tabs[0]:
                    # K-means clustering
                    cluster_cols = st.multiselect("Select columns for clustering", 
                                               numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))])
                    n_clusters = st.slider("Number of clusters", 2, 10, 3)
                    
                    if cluster_cols and len(cluster_cols) >= 2:
                        # Scale the data
                        scaler = StandardScaler()
                        df_scaled = scaler.fit_transform(df[cluster_cols])
                        
                        # Apply KMeans
                        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                        df['cluster'] = kmeans.fit_predict(df_scaled)
                        
                        # Visualize the clusters (using the first 2 columns)
                        fig = px.scatter(df, x=cluster_cols[0], y=cluster_cols[1], color='cluster', 
                                      title=f"KMeans Clustering with {n_clusters} clusters")
                        st.plotly_chart(fig)
                        
                        # Show cluster statistics
                        st.write("Cluster Statistics")
                        cluster_stats = df.groupby('cluster')[cluster_cols].mean()
                        st.write(cluster_stats)
                
                with adv_tabs[1]:
                    # Principal Component Analysis
                    pca_cols = st.multiselect("Select columns for PCA", 
                                           numeric_cols, default=numeric_cols)
                    n_components = st.slider("Number of components", 2, min(len(pca_cols), 5), 2)
                    
                    if pca_cols and len(pca_cols) >= n_components:
                        # Scale the data
                        scaler = StandardScaler()
                        df_scaled = scaler.fit_transform(df[pca_cols])
                        
                        # Apply PCA
                        pca = PCA(n_components=n_components)
                        principal_components = pca.fit_transform(df_scaled)
                        
                        # Create a DataFrame with the principal components
                        pca_df = pd.DataFrame(data=principal_components, 
                                            columns=[f'PC{i+1}' for i in range(n_components)])
                        
                        # Explained variance ratio
                        st.write("Explained Variance Ratio per Component")
                        st.write(pd.DataFrame({
                            'Component': [f'PC{i+1}' for i in range(n_components)],
                            'Explained Variance Ratio': pca.explained_variance_ratio_
                        }))
                        
                        # Visualize first two components
                        fig = px.scatter(pca_df, x='PC1', y='PC2', 
                                      title='PCA: First Two Principal Components')
                        st.plotly_chart(fig)
                
                with adv_tabs[2]:
                    # Anomaly Detection using Isolation Forest
                    anomaly_cols = st.multiselect("Select columns for anomaly detection", 
                                               numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))])
                    contamination = st.slider("Contamination (proportion of outliers)", 0.01, 0.5, 0.1, 0.01)
                    
                    if anomaly_cols:
                        # Prepare data
                        df_anomaly = df[anomaly_cols].dropna()
                        iso_forest = IsolationForest(contamination=contamination, random_state=42)
                        anomalies = iso_forest.fit_predict(df_anomaly)
                        
                        # Add anomaly labels to original dataframe
                        df.loc[df_anomaly.index, 'anomaly'] = anomalies
                        df['anomaly'] = df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})
                        
                        # Visualize (using first two columns if more than 2 selected)
                        if len(anomaly_cols) >= 2:
                            fig = px.scatter(df, x=anomaly_cols[0], y=anomaly_cols[1], color='anomaly',
                                          title='Anomaly Detection Results')
                            st.plotly_chart(fig)
                        
                        # Show anomaly statistics
                        st.write("Anomaly Statistics")
                        st.write(df['anomaly'].value_counts())
                
                with adv_tabs[3]:
                    # What-If Analysis
                    what_if_col = st.selectbox("Select column to modify", numeric_cols)
                    what_if_value = st.number_input(f"New value for {what_if_col}", 
                                                 value=float(df[what_if_col].mean()))
                    
                    # Create a copy of the dataframe with modified values
                    df_what_if = df.copy()
                    df_what_if[what_if_col] = what_if_value
                    
                    # Show comparison
                    if st.button("Run What-If Analysis"):
                        st.write("Original vs Modified Statistics")
                        comparison = pd.concat([
                            df[what_if_col].describe().rename("Original"),
                            df_what_if[what_if_col].describe().rename("Modified")
                        ], axis=1)
                        st.write(comparison)
                        
                        # Visualize comparison
                        if len(numeric_cols) > 1:
                            compare_col = st.selectbox("Compare with", 
                                                    [col for col in numeric_cols if col != what_if_col])
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df[compare_col], y=df[what_if_col], 
                                                mode='markers', name='Original'))
                            fig.add_trace(go.Scatter(x=df_what_if[compare_col], y=df_what_if[what_if_col], 
                                                mode='markers', name='Modified'))
                            fig.update_layout(title=f'{what_if_col} vs {compare_col}: Original vs Modified')
                            st.plotly_chart(fig)
            
            # REPORTING SECTION
            if sections["Reporting"]:
                st.markdown("---")
                st.subheader("Reporting")
                
                report_format = st.selectbox("Report Format", ["HTML", "CSV", "JSON"])
                
                if st.button("Generate Report"):
                    if report_format == "HTML":
                        html_report = "<html><body>"
                        html_report += f"<h1>Data Analysis Report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h1>"
                        html_report += "<h2>Data Summary</h2>"
                        html_report += df.describe().to_html()
                        html_report += "</body></html>"
                        st.download_button("Download HTML Report", html_report, "report.html", "text/html")
                    
                    elif report_format == "CSV":
                        csv_report = df.describe().to_csv()
                        st.download_button("Download CSV Report", csv_report, "report.csv", "text/csv")
                    
                    elif report_format == "JSON":
                        json_report = df.describe().to_json()
                        st.download_button("Download JSON Report", json_report, "report.json", "application/json")
                
                # Custom report options
                with st.expander("Custom Report Options"):
                    report_cols = st.multiselect("Select columns for custom report", df.columns.tolist())
                    if report_cols and st.button("Generate Custom Report"):
                        custom_report = df[report_cols].describe().to_csv()
                        st.download_button("Download Custom Report", custom_report, "custom_report.csv", "text/csv")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
