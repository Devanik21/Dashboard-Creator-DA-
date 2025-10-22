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
from sklearn.cluster import DBSCAN # For Geospatial Clustering # For ROC AUC
import google.generativeai as genai
import io
import base64
import os
import re
from datetime import datetime, timedelta
import time # Import the time module
import warnings # Import locally to keep dependencies clear
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL # For Time Series Anomaly Detection
from mlxtend.frequent_patterns import apriori, association_rules # For Market Basket
from mlxtend.preprocessing import TransactionEncoder # For Market Basket
import nltk # For Sentiment Analysis
import networkx as nx # For Network Analysis
from wordcloud import WordCloud # For Text Profiler
from scipy.stats import norm, lognorm, expon, weibull_min # For Distribution Fitting
import matplotlib.cm as cm # For Distribution Fitting plot colors
from statsmodels.tsa.stattools import ccf # For Time-Lagged Cross-Correlation
from sklearn.linear_model import LogisticRegression # For Propensity Scoring & Treatment Effect
from lifelines import KaplanMeierFitter # For Survival Analysis
from sklearn.preprocessing import LabelEncoder # For Decision Tree target encoding # For ROC curve
from nltk.sentiment.vader import SentimentIntensityAnalyzer # For Sentiment Analysis
from lifelines import CoxPHFitter # For Survival Regression
# New imports for added tools
import duckdb # For SQL Query Workbench
from scipy.cluster.hierarchy import dendrogram, linkage # For Hierarchical Clustering
from sklearn.feature_extraction.text import CountVectorizer # For LDA
from sklearn.decomposition import LatentDirichletAllocation # For LDA
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.inspection import PartialDependenceDisplay # For PDP/ICE plots

warnings.filterwarnings('ignore') 
# New imports for SHAP and Prophet
import shap # For SHAP
from prophet import Prophet # For Prophet forecasting
from prophet.plot import plot_plotly as prophet_plot_plotly, plot_components_plotly as prophet_plot_components_plotly # For Prophet plots

# Page configuration
from sklearn.ensemble import GradientBoostingClassifier
st.set_page_config(layout="wide", page_title="Advanced Dashboard Creator", page_icon="📊")
 
# --- FUNCTIONS ---
def get_base64_of_bin_file(bin_file):
    """Encodes a binary file to a base64 string."""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
 
def set_page_background_and_style(file_path):
    """Sets the background image and applies custom CSS styles."""
    if not os.path.exists(file_path):
        st.error(f"Error: Background image not found at '{file_path}'.")
        return
    
    base64_img = get_base64_of_bin_file(file_path)
    
    css_text = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{base64_img}");
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    /* 100% Pure Transparency - No boxes, no borders */
    [data-testid="stHeader"],
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebarContent"],
    [data-testid="stBottomBlockContainer"],
    [data-testid="stChatInputContainer"],
    [data-testid="stFileUploader"],
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzoneInstructions"],
    .stTextArea,
    .stTextInput,
    .stChatMessage,
    [data-testid="stChatMessageContent"],
    .element-container,
    .stMarkdown,
    section[data-testid="stSidebar"],
    .stSelectbox,
    div[data-baseweb="select"],
    .stExpander,
    [data-testid="stSidebar"]::before {{
        background: transparent !important;
        backdrop-filter: none !important;
        border: none !important;
        box-shadow: none !important;
    }}
    
    /* Remove all borders from sidebar */
    [data-testid="stSidebar"] {{
        border-right: none !important;
    }}
    
    /* Pure transparent inputs - no borders */
    textarea, input {{
        background: transparent !important;
        border: none !important;
        color: #E2E8F0 !important;
        transition: all 0.3s ease !important;
        box-shadow: none !important;
    }}
    
    textarea:hover, input:hover,
    textarea:focus, input:focus {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #FFFFFF !important; /* Brighter on focus/hover */
    }}
    
    /* Pure transparent buttons - no borders */
    button {{
        background: transparent !important;
        border: none !important;
        color: #E2E8F0 !important;
        transition: all 0.3s ease !important;
        box-shadow: none !important;
    }}
    
    button:hover {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #FFFFFF !important; /* Brighter on hover */
        transform: none !important;
    }}
    
    /* App-wide text styling */
    /* The div selector is modified with :not() to exclude icon containers, fixing a bug where icons were replaced by text. */
    body, h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {{
        color: #E2E8F0 !important;
        font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
        font-size: 1.02rem; /* Slightly larger base font size for readability */
        line-height: 1.6;
    }}
    
    h1, h2, h3 {{
        font-weight: 300 !important;
        text-align: center;
        letter-spacing: 2px;
    }}
    
    h1 {{
        font-size: 3rem !important;
        color: #FFFFFF !important; /* Make main title stand out */
    }}
    
    .subtitle {{
        color: rgba(226, 232, 240, 0.8); /* Subtler version of main text color */
        font-size: 1.1rem; /* Keep subtitle size */
        margin-top: -10px;
        letter-spacing: 3px;
        font-weight: 300;
    }}
    
    /* Transparent chat messages - no borders */
    .stChatMessage {{
        background: transparent !important;
        border: none !important;
        padding-left: 0px !important;
        margin: 8px 0 !important;
    }}
    
    .stChatMessage:hover {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}
    
    /* File badges - pure transparent */
    .file-badge {{
        display: inline-block;
        background: transparent;
        border: none;
        padding: 4px 10px;
        margin: 3px;
        font-size: 0.9rem; /* Slightly larger badge text */
        color: rgba(226, 232, 240, 0.8);
        transition: all 0.3s ease;
    }}
    
    .file-badge:hover {{
        background: transparent;
        border: none;
        color: #FFFFFF;
    }}
    
    /* Data tool buttons - no special styling */
    .data-tool-button button {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}

    .data-tool-button button:hover {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}
    
    /* Minimal scrollbar */
    ::-webkit-scrollbar {{
        width: 6px;
        background: transparent;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: rgba(150, 150, 150, 0.3);
        border-radius: 3px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: rgba(180, 180, 180, 0.5);
    }}
    
    /* Placeholder text */
    ::placeholder {{
        color: rgba(226, 232, 240, 0.5) !important;
    }}
    
    /* Selectbox - transparent */
    div[data-baseweb="select"] > div {{
        background: transparent !important;
        border: none !important;
        color: #E2E8F0 !important; /* Ensure selectbox text is correct color */
    }}
    
    div[data-baseweb="select"]:hover > div {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}
    
    /* Footer */
    .footer {{
        font-size: 0.9rem;
        color: rgba(226, 232, 240, 0.7);
        text-align: center;
        font-weight: 300;
        letter-spacing: 1px;
    }}
    
    hr {{
        opacity: 0.1;
        border-color: rgba(200, 200, 200, 0.15);
        box-shadow: none;
    }}
    
    /* Expander - transparent */
    .stExpander {{
        background: transparent !important;
        border: none !important;
    }}
    
    .stExpander:hover {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}
    
    /* Caption text */
    .stCaptionContainer, small {{
        color: rgba(226, 232, 240, 0.7) !important;
        font-weight: 300;
    }}
    
    /* File uploader */
    .stFileUploader label {{
        color: #E2E8F0 !important;
    }}
    
    .stFileUploader section {{
        background: transparent !important;
        border: none !important;
    }}
    
    .stFileUploader section:hover {{
        background: transparent !important;
        border: none !important;
    }}
    </style>
    '''
    st.markdown(css_text, unsafe_allow_html=True)

# --- APP LAYOUT ---
set_page_background_and_style('Gemini_Generated_Image_phsbymphsbymphsb (1).png')

# --- PASSWORD PROTECTION ---
def check_password():
    """Returns `True` if the user has entered the correct password."""

    # Initialize session state if it doesn't exist
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
        st.session_state.password_attempts = 0

    # If password is correct, we're done.
    if st.session_state.password_correct:
        return True

    # If max attempts reached, lock the app
    if st.session_state.password_attempts >= 3:
        st.warning("🚨 Too many incorrect attempts. Access denied. Please refresh the page to try again.")
        st.stop()

    # Show password input form
    st.title("🔐 Secure Access")
    st.info("Please enter the password to access the Advanced Data Explorer.")
    with st.form("password_form"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter")

        if submitted:
            if password == st.secrets["app_password"]:
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.session_state.password_attempts += 1
                st.error(f"Incorrect password. You have {3 - st.session_state.password_attempts} attempts left.")
                st.rerun()
    st.stop()

check_password()

#with st.sidebar:
 #   st.image("d2.jpg", caption="From rows to revelations.", use_container_width=True)

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
# Helper function to get common columns
def get_common_columns(datasets_dict, col_type='all'):
    if not datasets_dict or len(datasets_dict) < 2: # Needs at least two datasets to find common columns
        return []

    dfs_to_compare = list(datasets_dict.values())
    if not dfs_to_compare:
        return []
        
    common_cols_set = set(dfs_to_compare[0].columns)
    for i in range(1, len(dfs_to_compare)):
        common_cols_set.intersection_update(dfs_to_compare[i].columns)
    
    common_cols_list = sorted(list(common_cols_set))

    if col_type == 'all':
        return common_cols_list
    
    first_df = dfs_to_compare[0]
    if col_type == 'numeric':
        return [col for col in common_cols_list if pd.api.types.is_numeric_dtype(first_df[col])]
    elif col_type == 'categorical':
        return [col for col in common_cols_list if pd.api.types.is_object_dtype(first_df[col]) or pd.api.types.is_categorical_dtype(first_df[col])]
    elif col_type == 'datetime':
        return [col for col in common_cols_list if pd.api.types.is_datetime64_any_dtype(first_df[col])]
    return []

st.sidebar.header("🪄 AI-Powered Assistance")
gemini_api_key = None # Initialize to handle cases where it's not found
try:
    # Load Gemini API key from Streamlit secrets
    gemini_api_key = st.secrets["gemini_api_key"]
    genai.configure(api_key=gemini_api_key)
    st.sidebar.success("Gemini API key loaded from secrets.")
except KeyError:
    st.sidebar.warning("Gemini API key not found in secrets. AI features will be disabled.")
except Exception as e:
    st.sidebar.error(f"API Error: {str(e)}")

# Main title with metrics
st.title("Advanced Data Explorer & Visualizer")
#st.image("d3.jpg", caption="Made for Analysts. Loved by Scientists.", use_container_width=True)
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
            with st.expander("🆚 Dataset Comparison Dashboard", expanded=True): # Keep expanded if conditions met
                # Ensure default selection is valid and doesn't exceed available datasets
                num_available_datasets = len(datasets.keys())
                default_selection_count = min(2, num_available_datasets)
                
                compare_datasets_selected = st.multiselect(
                    "Select datasets to compare",
                                                list(datasets.keys()), 
                                                default=list(datasets.keys())[:default_selection_count] if num_available_datasets > 0 else [],
                                                key="compare_multiselect"
                )
                
                if len(compare_datasets_selected) >= 2:
                    selected_dfs_dict = {name: datasets[name] for name in compare_datasets_selected}
                    
                    tab_overview, tab_schema, tab_numeric, tab_categorical, tab_quality = st.tabs([
                        "📊 Overview", "🧬 Schema Comparison", "🔢 Numeric Deep Dive", 
                        "🔠 Categorical Deep Dive", "📉 Data Quality"
                    ])

                    with tab_overview:
                        st.subheader("Basic Statistics Overview")
                        overview_data = []
                        for name, ds in selected_dfs_dict.items():
                            missing_pct_val = (ds.isnull().sum().sum() / (ds.shape[0] * ds.shape[1]) * 100) if ds.shape[0] > 0 and ds.shape[1] > 0 else 0
                            overview_data.append({
                                'Dataset': name,
                                'Rows': ds.shape[0],
                                'Columns': ds.shape[1],
                                'Numeric Cols': len(ds.select_dtypes(include=['number']).columns),
                                'Categorical Cols': len(ds.select_dtypes(include=['object', 'category']).columns),
                                'Datetime Cols': len(ds.select_dtypes(include=['datetime']).columns),
                                'Total Missing %': f"{missing_pct_val:.2f}%",
                                'Memory Usage (MB)': f"{ds.memory_usage(deep=True).sum() / (1024*1024):.2f}", # Feature 1
                                'Duplicate Rows': ds.duplicated().sum() # Feature 2
                            })
                        overview_df = pd.DataFrame(overview_data)
                        st.dataframe(overview_df)

                        if not overview_df.empty:
                            fig_rows_comp = px.bar(overview_df, x='Dataset', y='Rows', title="Dataset Size Comparison (Rows)", color_discrete_sequence=[custom_color])
                            st.plotly_chart(fig_rows_comp, use_container_width=True)
                            fig_cols_comp = px.bar(overview_df, x='Dataset', y='Columns', title="Dataset Size Comparison (Columns)", color_discrete_sequence=[px.colors.qualitative.Plotly[1]])
                            st.plotly_chart(fig_cols_comp, use_container_width=True)

                    with tab_schema:
                        st.subheader("Schema & Structure Comparison")
                        
                        # Feature 3: Common Columns
                        common_cols_schema = get_common_columns(selected_dfs_dict, 'all')
                        st.markdown("#### Common Columns Across All Selected Datasets")
                        if common_cols_schema:
                            st.write(", ".join(common_cols_schema) if common_cols_schema else "None")
                        else:
                            st.info("No columns are common across all selected datasets.")

                        # Feature 4: Unique Columns
                        st.markdown("#### Unique Columns per Dataset")
                        all_cols_in_selection = set()
                        for ds_name, ds_df in selected_dfs_dict.items():
                            all_cols_in_selection.update(ds_df.columns)
                        
                        for ds_name, ds_df in selected_dfs_dict.items():
                            other_cols = set()
                            for other_name, other_df in selected_dfs_dict.items():
                                if ds_name != other_name:
                                    other_cols.update(other_df.columns)
                            unique_to_ds = set(ds_df.columns) - other_cols
                            if unique_to_ds:
                                st.write(f"**{ds_name}:** {', '.join(sorted(list(unique_to_ds)))}")
                            else:
                                st.write(f"**{ds_name}:** No columns unique to this dataset compared to others in selection.")
                        
                        # Feature 5: Data Type Mismatches for Common Columns
                        if common_cols_schema:
                            st.markdown("#### Data Type Mismatches for Common Columns")
                            mismatch_info = []
                            first_ds_name = list(selected_dfs_dict.keys())[0]
                            first_ds_df = selected_dfs_dict[first_ds_name]
                            
                            for col_name in common_cols_schema:
                                base_dtype = str(first_ds_df[col_name].dtype)
                                for ds_name_comp, ds_df_comp in selected_dfs_dict.items():
                                    if ds_name_comp == first_ds_name:
                                        continue
                                    current_dtype = str(ds_df_comp[col_name].dtype)
                                    if base_dtype != current_dtype:
                                        mismatch_info.append({
                                            'Column': col_name,
                                            first_ds_name: base_dtype,
                                            ds_name_comp: current_dtype,
                                            'Status': 'Mismatch'
                                        })
                            if mismatch_info:
                                st.dataframe(pd.DataFrame(mismatch_info))
                            else:
                                st.info("No data type mismatches found for common columns across the selected datasets.")

                    with tab_numeric:
                        st.subheader("Numeric Column Deep Dive")
                        common_numeric_cols = get_common_columns(selected_dfs_dict, 'numeric')
                        if not common_numeric_cols:
                            st.info("No common numeric columns found across selected datasets.")
                        else:
                            selected_num_col_compare = st.selectbox("Select a common numeric column to analyze:", common_numeric_cols, key="num_col_compare_select")
                            if selected_num_col_compare:
                                # Feature 6: Side-by-side descriptive statistics
                                st.markdown(f"#### Descriptive Statistics for '{selected_num_col_compare}'")
                                desc_stats_list = []
                                for ds_name, ds_df in selected_dfs_dict.items():
                                    if selected_num_col_compare in ds_df.columns:
                                        desc_stats_list.append(ds_df[selected_num_col_compare].describe().rename(ds_name))
                                if desc_stats_list:
                                    st.dataframe(pd.concat(desc_stats_list, axis=1))
                                
                                # Feature 7: Visual comparison (overlaid histograms or box plots)
                                st.markdown(f"#### Visual Comparison for '{selected_num_col_compare}'")
                                plot_type_num = st.radio("Plot Type:", ["Overlaid Histograms", "Box Plots"], key="num_plot_type_radio")
                                
                                combined_num_data = pd.DataFrame()
                                for ds_name, ds_df in selected_dfs_dict.items():
                                    if selected_num_col_compare in ds_df.columns:
                                        temp_df = pd.DataFrame({selected_num_col_compare: ds_df[selected_num_col_compare], 'Dataset': ds_name})
                                        combined_num_data = pd.concat([combined_num_data, temp_df], ignore_index=True)

                                if not combined_num_data.empty:
                                    if plot_type_num == "Overlaid Histograms":
                                        fig_num_hist = px.histogram(combined_num_data, x=selected_num_col_compare, color='Dataset', 
                                                                    barmode='overlay', marginal='rug', opacity=0.7,
                                                                    title=f"Distribution of '{selected_num_col_compare}' by Dataset")
                                        st.plotly_chart(fig_num_hist, use_container_width=True)
                                    else: # Box Plots
                                        fig_num_box = px.box(combined_num_data, y=selected_num_col_compare, color='Dataset',
                                                             title=f"Box Plot of '{selected_num_col_compare}' by Dataset")
                                        st.plotly_chart(fig_num_box, use_container_width=True)

                                # Feature 8: Statistical test (T-test/Mann-Whitney U) - for first two selected datasets
                                if len(selected_dfs_dict) >= 2:
                                    st.markdown(f"#### Statistical Test for Difference in '{selected_num_col_compare}'")
                                    ds_names_for_test = list(selected_dfs_dict.keys())[:2]
                                    data1_test = selected_dfs_dict[ds_names_for_test[0]][selected_num_col_compare].dropna()
                                    data2_test = selected_dfs_dict[ds_names_for_test[1]][selected_num_col_compare].dropna()

                                    if len(data1_test) > 1 and len(data2_test) > 1: # Need at least 2 samples for these tests
                                        test_choice = st.radio("Choose Test:", ["T-test (parametric)", "Mann-Whitney U (non-parametric)"], key="num_stat_test_choice")
                                        st.caption(f"Comparing '{ds_names_for_test[0]}' and '{ds_names_for_test[1]}'. For more than 2 datasets, only the first two are compared here.")
                                        
                                        alpha_stat_test = 0.05 # Significance level
                                        if test_choice == "T-test (parametric)":
                                            stat, p_value = stats.ttest_ind(data1_test, data2_test, equal_var=False) # Welch's t-test
                                            st.write(f"**T-test results:** Statistic = {stat:.3f}, P-value = {p_value:.4f}")
                                        else: # Mann-Whitney U
                                            stat, p_value = stats.mannwhitneyu(data1_test, data2_test, alternative='two-sided')
                                            st.write(f"**Mann-Whitney U test results:** Statistic = {stat:.3f}, P-value = {p_value:.4f}")
                                        
                                        if p_value < alpha_stat_test:
                                            st.success(f"Significant difference found (p < {alpha_stat_test}).")
                                        else:
                                            st.info(f"No significant difference found (p >= {alpha_stat_test}).")
                                    else:
                                        st.warning("Not enough data in one or both datasets for statistical testing on this column.")

                    with tab_categorical:
                        st.subheader("Categorical Column Deep Dive")
                        common_categorical_cols = get_common_columns(selected_dfs_dict, 'categorical')
                        if not common_categorical_cols:
                            st.info("No common categorical columns found across selected datasets.")
                        else:
                            selected_cat_col_compare = st.selectbox("Select a common categorical column to analyze:", common_categorical_cols, key="cat_col_compare_select")
                            if selected_cat_col_compare:
                                # Feature 9: Side-by-side value counts & visual
                                st.markdown(f"#### Value Distribution for '{selected_cat_col_compare}'")
                                all_cat_data_for_plot = pd.DataFrame()
                                for ds_name, ds_df in selected_dfs_dict.items():
                                    if selected_cat_col_compare in ds_df.columns:
                                        counts = ds_df[selected_cat_col_compare].value_counts(normalize=True).mul(100).rename('Percentage').reset_index()
                                        counts.columns = [selected_cat_col_compare, 'Percentage']
                                        counts['Dataset'] = ds_name
                                        all_cat_data_for_plot = pd.concat([all_cat_data_for_plot, counts])
                                
                                if not all_cat_data_for_plot.empty:
                                    fig_cat_dist = px.bar(all_cat_data_for_plot, x=selected_cat_col_compare, y='Percentage', color='Dataset', 
                                                          barmode='group', title=f"Distribution of '{selected_cat_col_compare}' by Dataset")
                                    st.plotly_chart(fig_cat_dist, use_container_width=True)
                                    
                                    st.markdown("##### Value Counts (Top 10 per Dataset)")
                                    for ds_name, ds_df in selected_dfs_dict.items():
                                        if selected_cat_col_compare in ds_df.columns:
                                            st.write(f"**{ds_name}:**")
                                            st.dataframe(ds_df[selected_cat_col_compare].value_counts().head(10).rename("Count"))

                    with tab_quality:
                        st.subheader("Data Quality Comparison")
                        # Feature 10: Detailed Missing Values per common column
                        st.markdown("#### Missing Values per Common Column")
                        common_cols_quality = get_common_columns(selected_dfs_dict, 'all')
                        if not common_cols_quality:
                            st.info("No common columns to compare missing values.")
                        else:
                            missing_data_list = []
                            for ds_name, ds_df in selected_dfs_dict.items():
                                for col_name in common_cols_quality:
                                    if col_name in ds_df.columns:
                                        missing_count = ds_df[col_name].isnull().sum()
                                        missing_pct = (missing_count / len(ds_df)) * 100 if len(ds_df) > 0 else 0
                                        missing_data_list.append({'Dataset': ds_name, 'Column': col_name, 'MissingPercentage': missing_pct})
                            
                            if missing_data_list:
                                missing_df_plot = pd.DataFrame(missing_data_list)
                                fig_missing_comp = px.bar(missing_df_plot, x='Column', y='MissingPercentage', color='Dataset',
                                                          barmode='group', title="Missing Value Percentage per Common Column by Dataset")
                                st.plotly_chart(fig_missing_comp, use_container_width=True)
                            else:
                                st.info("No data to plot for missing values comparison.")

                elif len(compare_datasets_selected) < 2: # If mode is on, datasets > 1, but not enough selected in multiselect
                    st.info("Please select at least two datasets from the dropdown above to compare.")
        elif comparison_mode and len(datasets) <= 1: # If mode is on, but not enough datasets uploaded
            st.info("📊 Data Comparison Mode is enabled. Please upload at least two datasets to use this feature.")

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
                anomaly_col = st.selectbox("Select Column for Anomaly Detection", numeric_cols, key="f11_anomaly_col_selector")
                method = st.selectbox("Detection Method", ["IQR", "Z-Score", "Isolation Forest"], key="f11_method_selector")
                
                anomalies_mask = pd.Series(False, index=df.index) # Initialize a boolean mask aligned with the DataFrame

                if method == "IQR":
                    if pd.api.types.is_numeric_dtype(df[anomaly_col]) and not df[anomaly_col].isnull().all():
                        Q1, Q3 = df[anomaly_col].quantile([0.25, 0.75])
                        IQR_value = Q3 - Q1 # Renamed to avoid conflict if a module named IQR is imported
                        lower_bound = Q1 - 1.5 * IQR_value
                        upper_bound = Q3 + 1.5 * IQR_value
                        anomalies_mask = (df[anomaly_col] < lower_bound) | (df[anomaly_col] > upper_bound)
                    else:
                        st.warning(f"Column '{anomaly_col}' is not suitable for IQR method (e.g., not numeric or all NaN).")
                
                elif method == "Z-Score":
                    if pd.api.types.is_numeric_dtype(df[anomaly_col]) and not df[anomaly_col].isnull().all() and df[anomaly_col].std() != 0:
                        z_scores = np.abs((df[anomaly_col] - df[anomaly_col].mean()) / df[anomaly_col].std())
                        anomalies_mask = z_scores > 3
                    else:
                        st.warning(f"Column '{anomaly_col}' is not suitable for Z-Score method (e.g., not numeric, all NaN, or zero standard deviation).")
                
                else:  # Isolation Forest
                    data_for_iso = df[[anomaly_col]].dropna()
                    if not data_for_iso.empty and len(data_for_iso) > 1: # Isolation Forest needs at least 2 samples
                        contamination_iso = st.slider("Isolation Forest Contamination", 0.01, 0.5, 0.1, 0.01, key="iso_contamination_f11")
                        iso_forest = IsolationForest(contamination=contamination_iso, random_state=42)
                        predictions = iso_forest.fit_predict(data_for_iso) # numpy array of -1s (outlier) and 1s (inlier)
                        
                        # Create a boolean Series on data_for_iso's index indicating anomalies
                        is_anomaly_on_subset = pd.Series(predictions == -1, index=data_for_iso.index)
                        
                        # Update the main anomalies_mask for the original DataFrame
                        anomalies_mask.loc[is_anomaly_on_subset[is_anomaly_on_subset].index] = True
                    elif not data_for_iso.empty and len(data_for_iso) <=1 :
                        st.warning(f"Not enough data points ({len(data_for_iso)}) in '{anomaly_col}' after dropping NaNs for Isolation Forest. Need at least 2.")
                    else:
                        st.warning(f"Column '{anomaly_col}' is empty after dropping NaNs. Cannot use Isolation Forest.")
                
                # Store the detected anomaly rows in session state
                st.session_state.anomalies_detected_df = df[anomalies_mask].copy()
                
                anomaly_count = anomalies_mask.sum()
                st.metric("Anomalies Found", anomaly_count)
                
                # Visualization
                fig = go.Figure()
                # Plot all points, highlighting anomalies based on the mask
                fig.add_trace(go.Scatter(x=df.index, y=df[anomaly_col], mode='markers', name='Data Points',
                                       marker=dict(color=np.where(anomalies_mask, 'red', custom_color), 
                                                   size=np.where(anomalies_mask, 8, 5))))
                
                fig.update_layout(title=f"Anomaly Detection in '{anomaly_col}' using {method} method",
                                  xaxis_title="Index", yaxis_title=anomaly_col)
                st.plotly_chart(fig, use_container_width=True)

                if anomaly_count > 0:
                    st.write("Detected Anomalies (first 100 rows):")
                    st.dataframe(st.session_state.anomalies_detected_df.head(100))
            else:
                st.info("Anomaly detection requires numeric columns.")

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
                        
                        # Add the 'Cluster' column back to the main DataFrame
                        # Ensure alignment by using the index from cluster_data (which came from df.dropna())
                        df.loc[cluster_data.index, 'Cluster'] = cluster_data_copy['Cluster']
                        
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
                                    tfidf_df = pd.DataFrame(tfidf_scores.A1, columns=['score'], index=feature_names) # Corrected line
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

        # NEW FEATURE 34: Automated Data Cleaning Suggestions
        with st.expander("🧹 Automated Data Cleaning Suggestions"):
            st.subheader("Get Smart Suggestions for Data Cleaning")
            if not df.empty:
                cleaning_suggestions = []
                for col in df.columns:
                    # Missing value suggestions
                    missing_pct = (df[col].isnull().sum() / len(df)) * 100
                    if missing_pct > 0:
                        suggestion = f"Column '{col}' has {missing_pct:.1f}% missing values. "
                        if pd.api.types.is_numeric_dtype(df[col]):
                            suggestion += "Consider imputation with mean, median, or a constant (e.g., 0)."
                        elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
                            suggestion += "Consider imputation with mode or a placeholder like 'Unknown'."
                        else:
                            suggestion += "Consider imputation based on data type."
                        cleaning_suggestions.append({'Column': col, 'Issue': 'Missing Values', 'Suggestion': suggestion, 'Severity': 'High' if missing_pct > 20 else 'Medium'})

                    # Outlier suggestions for numeric columns (simple IQR based)
                    if pd.api.types.is_numeric_dtype(df[col]) and not df[col].isnull().all():
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR_val = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR_val
                        upper_bound = Q3 + 1.5 * IQR_val
                        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                        if not outliers.empty:
                            cleaning_suggestions.append({'Column': col, 'Issue': 'Potential Outliers', 
                                                         'Suggestion': f"Found {len(outliers)} potential outliers (values outside {lower_bound:.2f} - {upper_bound:.2f}). Consider capping, removing, or transforming.", 
                                                         'Severity': 'Medium'})

                    # High cardinality categorical columns
                    if (df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col])) and df[col].nunique() > 50 and df[col].nunique() < len(df) * 0.8:
                        cleaning_suggestions.append({'Column': col, 'Issue': 'High Cardinality', 
                                                     'Suggestion': f"Column '{col}' has {df[col].nunique()} unique values. Consider grouping rare categories, target encoding, or dimensionality reduction if used as a feature.", 
                                                     'Severity': 'Low'})
                
                if cleaning_suggestions:
                    suggestions_df_clean = pd.DataFrame(cleaning_suggestions)
                    st.write("#### Data Cleaning Suggestions:")
                    st.dataframe(suggestions_df_clean)
                else:
                    st.info("No immediate data cleaning suggestions found based on basic checks.")
            else:
                st.info("Upload data to get cleaning suggestions.")

        # NEW FEATURE 35: Advanced Customer Segmentation Profiler
        with st.expander("🧩 Advanced Customer Segmentation Profiler"):
            st.subheader("Profile and Understand Your Customer Segments")
            if 'Cluster' in df.columns and pd.api.types.is_numeric_dtype(df['Cluster']): # Check if K-Means or similar was run
                st.info("This tool profiles segments based on an existing 'Cluster' column (assumed to be generated by a clustering algorithm like K-Means).")
                
                profile_features_num = st.multiselect("Select Numeric Features for Profiling Segments", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))], key="seg_profile_num_feats")
                profile_features_cat = st.multiselect("Select Categorical Features for Profiling Segments", categorical_cols, default=categorical_cols[:min(2, len(categorical_cols))], key="seg_profile_cat_feats")

                if profile_features_num or profile_features_cat:
                    st.write("#### Segment Profiles:")
                    segment_summary = df.groupby('Cluster').agg(
                        **{f'{col}_mean': (col, 'mean') for col in profile_features_num},
                        **{f'{col}_median': (col, 'median') for col in profile_features_num},
                        **{f'{col}_mode': (col, lambda x: x.mode()[0] if not x.mode().empty else 'N/A') for col in profile_features_cat},
                        SegmentSize=('Cluster', 'size')
                    ).reset_index()
                    st.dataframe(segment_summary)

                    for col_prof in profile_features_num:
                        fig_prof_num = px.box(df, x='Cluster', y=col_prof, color='Cluster', title=f"Distribution of '{col_prof}' by Segment", color_discrete_sequence=px.colors.qualitative.Plotly)
                        st.plotly_chart(fig_prof_num, use_container_width=True)
                    
                    for col_prof_cat in profile_features_cat:
                        cat_summary_df = df.groupby('Cluster')[col_prof_cat].value_counts(normalize=True).mul(100).rename('Percentage').reset_index()
                        fig_prof_cat = px.bar(cat_summary_df, x='Cluster', y='Percentage', color=col_prof_cat, title=f"Distribution of '{col_prof_cat}' by Segment", barmode='group')
                        st.plotly_chart(fig_prof_cat, use_container_width=True)
                else:
                    st.info("Select features to profile the segments.")
            else:
                st.warning("No 'Cluster' column found. Please run K-Means Clustering Analysis first to generate segments.")

        # NEW FEATURE 36: Geospatial Clustering (DBSCAN)
        with st.expander("🌍 Geospatial Clustering (DBSCAN)"):
            st.subheader("Identify Geographic Clusters of Data Points")
            potential_lat_cols_dbscan = [col for col in numeric_cols if 'lat' in col.lower()]
            potential_lon_cols_dbscan = [col for col in numeric_cols if 'lon' in col.lower() or 'lng' in col.lower()]

            lat_col_dbscan_default = potential_lat_cols_dbscan[0] if potential_lat_cols_dbscan else None
            lon_col_dbscan_default = potential_lon_cols_dbscan[0] if potential_lon_cols_dbscan else None

            lat_col_dbscan = st.selectbox("Select Latitude Column for DBSCAN", numeric_cols, index=numeric_cols.index(lat_col_dbscan_default) if lat_col_dbscan_default and lat_col_dbscan_default in numeric_cols else 0, key="dbscan_lat")
            lon_col_dbscan = st.selectbox("Select Longitude Column for DBSCAN", numeric_cols, index=numeric_cols.index(lon_col_dbscan_default) if lon_col_dbscan_default and lon_col_dbscan_default in numeric_cols else (1 if len(numeric_cols) > 1 else 0), key="dbscan_lon")
            
            if lat_col_dbscan and lon_col_dbscan:
                eps_dbscan = st.slider("DBSCAN Epsilon (max distance between samples for one to be considered as in the neighborhood of the other)", 0.01, 5.0, 0.5, 0.01, key="dbscan_eps", help="Adjust based on the scale of your coordinates and desired cluster density.")
                min_samples_dbscan = st.slider("DBSCAN Min Samples (number of samples in a neighborhood for a point to be considered as a core point)", 2, 50, 5, key="dbscan_min_samples")

                if st.button("Run Geospatial Clustering (DBSCAN)", key="run_dbscan"):
                    geo_data_dbscan = df[[lat_col_dbscan, lon_col_dbscan]].copy().dropna()
                    geo_data_dbscan.columns = ['lat', 'lon'] # Standardize names for st.map

                    if not geo_data_dbscan.empty and len(geo_data_dbscan) >= min_samples_dbscan:
                        # Scale coordinates if they are in degrees (DBSCAN uses Euclidean distance)
                        # A simple scaling might be needed, or convert to radians for haversine if using a custom metric.
                        # For simplicity, we'll use scaled Euclidean distance on lat/lon.
                        scaler_dbscan = StandardScaler()
                        scaled_coords = scaler_dbscan.fit_transform(geo_data_dbscan[['lat', 'lon']])
                        
                        dbscan = DBSCAN(eps=eps_dbscan, min_samples=min_samples_dbscan)
                        geo_data_dbscan['GeoCluster'] = dbscan.fit_predict(scaled_coords)
                        
                        num_geo_clusters = len(set(geo_data_dbscan['GeoCluster'])) - (1 if -1 in geo_data_dbscan['GeoCluster'] else 0)
                        st.metric("Number of Geographic Clusters Found", num_geo_clusters)
                        st.metric("Number of Noise Points (Outliers)", (geo_data_dbscan['GeoCluster'] == -1).sum())

                        # Create a color map for clusters, ensuring -1 (noise) is distinct
                        unique_clusters_dbscan = sorted(geo_data_dbscan['GeoCluster'].unique())
                        colors_dbscan = px.colors.qualitative.Plotly + px.colors.qualitative.Light24
                        cluster_color_map_dbscan = {
                            cluster_id: colors_dbscan[i % len(colors_dbscan)] 
                            for i, cluster_id in enumerate(unique_clusters_dbscan) if cluster_id != -1
                        }
                        cluster_color_map_dbscan[-1] = 'grey' # Noise points

                        geo_data_dbscan['Color'] = geo_data_dbscan['GeoCluster'].map(cluster_color_map_dbscan)

                        fig_dbscan_map = px.scatter_mapbox(geo_data_dbscan, lat="lat", lon="lon", 
                                                           color="GeoCluster", # Use the cluster ID for legend
                                                           color_discrete_map=cluster_color_map_dbscan, # Apply custom colors
                                                           hover_name=geo_data_dbscan.index,
                                                           title="Geospatial Clusters (DBSCAN)",
                                                           zoom=3, height=600)
                        fig_dbscan_map.update_layout(mapbox_style="open-street-map")
                        fig_dbscan_map.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
                        st.plotly_chart(fig_dbscan_map, use_container_width=True)
                        
                        st.write("Data with Geospatial Clusters (first 100 rows):")
                        st.dataframe(geo_data_dbscan.head(100))
                    else:
                        st.warning("Not enough data points for DBSCAN after dropping NaNs or fewer points than min_samples.")
            else:
                st.info("Select Latitude and Longitude columns for geospatial clustering.")

        # NEW FEATURE 37: Automated Time Series Anomaly Detection (STL)
        with st.expander("📈 Automated Time Series Anomaly Detection (STL)"):
            st.subheader("Identify Anomalies in Time Series using STL Decomposition")
            if date_cols and numeric_cols:
                ts_anomaly_date_col = st.selectbox("Select Date Column for Time Series Anomaly", date_cols, key="ts_anomaly_date")
                ts_anomaly_value_col = st.selectbox("Select Value Column for Time Series Anomaly", numeric_cols, key="ts_anomaly_value")
                stl_period = st.number_input("Seasonality Period (e.g., 7 for daily data with weekly seasonality, 12 for monthly with yearly)", min_value=2, value=7, key="stl_period")
                anomaly_threshold_factor = st.slider("Anomaly Threshold Factor (for residuals std dev)", 1.0, 5.0, 2.5, 0.1, key="stl_anomaly_thresh")

                if ts_anomaly_date_col and ts_anomaly_value_col:
                    if st.button("Detect Time Series Anomalies", key="run_ts_anomaly_stl"):
                        try:
                            ts_anomaly_df = df[[ts_anomaly_date_col, ts_anomaly_value_col]].copy().dropna()
                            ts_anomaly_df = ts_anomaly_df.sort_values(ts_anomaly_date_col)
                            ts_anomaly_df = ts_anomaly_df.set_index(ts_anomaly_date_col)[ts_anomaly_value_col]
                            
                            # Resample to daily frequency, taking mean for duplicates, then ffill
                            ts_anomaly_df_resampled = ts_anomaly_df.resample('D').mean().fillna(method='ffill')

                            if len(ts_anomaly_df_resampled) < 2 * stl_period: # STL needs at least 2 full periods
                                st.warning(f"Not enough data for STL decomposition. Need at least {2*stl_period} data points after resampling. Found {len(ts_anomaly_df_resampled)}.")
                            else:
                                stl = STL(ts_anomaly_df_resampled, period=stl_period, robust=True)
                                res_stl = stl.fit()
                                
                                residuals = res_stl.resid
                                residual_std = residuals.std()
                                lower_bound_stl = residuals.mean() - anomaly_threshold_factor * residual_std
                                upper_bound_stl = residuals.mean() + anomaly_threshold_factor * residual_std
                                
                                anomalies_stl = ts_anomaly_df_resampled[(residuals < lower_bound_stl) | (residuals > upper_bound_stl)]

                                fig_stl = go.Figure()
                                fig_stl.add_trace(go.Scatter(x=ts_anomaly_df_resampled.index, y=ts_anomaly_df_resampled, mode='lines', name='Original Series', line=dict(color=custom_color)))
                                fig_stl.add_trace(go.Scatter(x=anomalies_stl.index, y=anomalies_stl, mode='markers', name='Anomalies', marker=dict(color='red', size=8)))
                                fig_stl.update_layout(title=f"Time Series Anomaly Detection for '{ts_anomaly_value_col}'", xaxis_title="Date", yaxis_title="Value")
                                st.plotly_chart(fig_stl, use_container_width=True)

                                st.metric("Number of Anomalies Detected", len(anomalies_stl))
                                if not anomalies_stl.empty:
                                    st.write("Detected Anomalies:")
                                    st.dataframe(anomalies_stl.reset_index())
                        except Exception as e:
                            st.error(f"Error during Time Series Anomaly Detection: {e}")
            else:
                st.info("Time Series Anomaly Detection requires a date column and a numeric value column.")

        # NEW FEATURE 38: Comparative Product Performance (Top vs. Bottom N%)
        with st.expander("🆚 Comparative Product Performance (Top vs. Bottom N%)"):
            st.subheader("Profile Top and Bottom Performing Products")
            if categorical_cols and numeric_cols: # Need a product ID and a performance metric
                product_id_col_comp = st.selectbox("Select Product ID Column", categorical_cols + numeric_cols, key="comp_prod_id")
                performance_metric_col = st.selectbox("Select Performance Metric (e.g., Sales, Revenue)", numeric_cols, key="comp_perf_metric")
                top_n_percent = st.slider("Select N% for Top/Bottom Performers", 1, 50, 10, key="comp_top_n_pct")
                
                profiling_attrs_comp = st.multiselect("Select Attributes for Profiling", categorical_cols + numeric_cols, 
                                                      default=[c for c in categorical_cols[:2] + numeric_cols[:1] if c not in [product_id_col_comp, performance_metric_col]], 
                                                      key="comp_profile_attrs")

                if product_id_col_comp and performance_metric_col and profiling_attrs_comp:
                    if st.button("Compare Product Performance", key="run_comp_perf"):
                        try:
                            product_performance = df.groupby(product_id_col_comp)[performance_metric_col].sum().reset_index()
                            
                            if product_performance.empty:
                                st.warning("No product performance data to rank.")
                            else:
                                n_products = len(product_performance)
                                num_to_select = max(1, int(n_products * (top_n_percent / 100.0)))

                                top_performers_ids = product_performance.nlargest(num_to_select, performance_metric_col)[product_id_col_comp]
                                bottom_performers_ids = product_performance.nsmallest(num_to_select, performance_metric_col)[product_id_col_comp]

                                top_df_comp = df[df[product_id_col_comp].isin(top_performers_ids)]
                                bottom_df_comp = df[df[product_id_col_comp].isin(bottom_performers_ids)]

                                st.write(f"#### Profiling Top {top_n_percent}% vs. Bottom {top_n_percent}% Products")
                                st.write(f"(Top {num_to_select} products vs. Bottom {num_to_select} products)")

                                for attr in profiling_attrs_comp:
                                    st.markdown(f"##### Attribute: {attr}")
                                    if pd.api.types.is_numeric_dtype(df[attr]):
                                        top_mean = top_df_comp[attr].mean()
                                        bottom_mean = bottom_df_comp[attr].mean()
                                        st.metric(f"Avg '{attr}' (Top)", f"{top_mean:.2f}" if pd.notna(top_mean) else "N/A")
                                        st.metric(f"Avg '{attr}' (Bottom)", f"{bottom_mean:.2f}" if pd.notna(bottom_mean) else "N/A", delta=f"{(top_mean - bottom_mean):.2f}" if pd.notna(top_mean) and pd.notna(bottom_mean) else None)
                                    else: # Categorical
                                        col_top, col_bottom = st.columns(2)
                                        with col_top:
                                            st.write(f"Distribution for Top Performers:")
                                            st.dataframe(top_df_comp[attr].value_counts(normalize=True).mul(100).round(1).rename("Top %"))
                                        with col_bottom:
                                            st.write(f"Distribution for Bottom Performers:")
                                            st.dataframe(bottom_df_comp[attr].value_counts(normalize=True).mul(100).round(1).rename("Bottom %"))
                        except Exception as e:
                            st.error(f"Error during comparative product performance analysis: {e}")
                else:
                    st.info("Select Product ID, Performance Metric, and at least one Profiling Attribute.")
            else:
                st.info("Comparative Product Performance requires categorical columns (for Product ID) and numeric columns (for metrics and profiling).")

        # NEW TOOL 1: Time-Lagged Cross-Correlation Analysis
        with st.expander("🕰️ Time-Lagged Cross-Correlation Analysis"):
            st.subheader("Analyze Leading/Lagging Relationships Between Time Series")
            st.info("This tool calculates the cross-correlation between two time series to identify if one series leads or lags the other. Requires two numeric columns and one date column.")
            if len(date_cols) > 0 and len(numeric_cols) >= 2:
                tlcc_date_col = st.selectbox("Select Date Column for Time Series", date_cols, key="tlcc_date")
                tlcc_col1 = st.selectbox("Select First Numeric Column (Series 1)", numeric_cols, key="tlcc_col1")
                tlcc_col2_options = [col for col in numeric_cols if col != tlcc_col1]
                if tlcc_col2_options:
                    tlcc_col2 = st.selectbox("Select Second Numeric Column (Series 2)", tlcc_col2_options, key="tlcc_col2")
                    max_lag = st.slider("Maximum Lag to Consider", 1, 100, 20, key="tlcc_max_lag")

                    if tlcc_date_col and tlcc_col1 and tlcc_col2:
                        if st.button("Run Time-Lagged Cross-Correlation", key="run_tlcc"):
                            try:
                                tlcc_df_prep = df[[tlcc_date_col, tlcc_col1, tlcc_col2]].copy().dropna()
                                tlcc_df_prep = tlcc_df_prep.sort_values(tlcc_date_col)
                                
                                # Resample to a consistent frequency (e.g., daily mean) to handle irregular timestamps
                                series1 = tlcc_df_prep.set_index(tlcc_date_col)[tlcc_col1].resample('D').mean().fillna(method='ffill').dropna()
                                series2 = tlcc_df_prep.set_index(tlcc_date_col)[tlcc_col2].resample('D').mean().fillna(method='ffill').dropna()

                                # Align series by index
                                common_index = series1.index.intersection(series2.index)
                                series1_aligned = series1.loc[common_index]
                                series2_aligned = series2.loc[common_index]

                                if len(series1_aligned) < max_lag + 5 or len(series2_aligned) < max_lag + 5: # Need enough data
                                    st.warning(f"Not enough overlapping data points after resampling for the chosen lag ({max_lag}). Need at least {max_lag+5} points.")
                                else:
                                    cross_corr_values = ccf(series1_aligned, series2_aligned, adjusted=False, fft=True)[:max_lag+1] # Calculate for positive lags
                                    lags = np.arange(0, max_lag + 1)

                                    fig_tlcc = go.Figure()
                                    fig_tlcc.add_trace(go.Bar(x=lags, y=cross_corr_values, name=f'Cross-correlation ({tlcc_col1} vs {tlcc_col2})', marker_color=custom_color))
                                    fig_tlcc.update_layout(
                                        title=f"Time-Lagged Cross-Correlation: {tlcc_col1} leads/lags {tlcc_col2}",
                                        xaxis_title=f"Lag (Positive lag means {tlcc_col1} leads {tlcc_col2})",
                                        yaxis_title="Cross-correlation Coefficient"
                                    )
                                    st.plotly_chart(fig_tlcc, use_container_width=True)

                                    peak_corr_lag = lags[np.argmax(np.abs(cross_corr_values))]
                                    peak_corr_val = cross_corr_values[np.argmax(np.abs(cross_corr_values))]
                                    st.info(f"Peak absolute correlation of {peak_corr_val:.3f} occurs at lag {peak_corr_lag}. This suggests the strongest relationship when '{tlcc_col1}' is shifted by {peak_corr_lag} periods relative to '{tlcc_col2}'.")

                            except Exception as e:
                                st.error(f"Error during Time-Lagged Cross-Correlation: {e}")
                else:
                    st.info("Select a date column and two distinct numeric columns.")
            else:
                st.info("Time-Lagged Cross-Correlation requires at least one date column and two numeric columns.")

        # NEW TOOL 2: Interactive Segment Profiler (Rule-Based)
        with st.expander("📐 Interactive Segment Profiler (Rule-Based)"):
            st.subheader("Define and Profile Custom Data Segments")
            st.info("Create segments by defining rules on one or more columns, then analyze their characteristics.")

            if not df.empty:
                num_rules = st.number_input("Number of Rules to Define Segment", 1, 5, 1, key="isp_num_rules")
                rules = []
                for i in range(num_rules):
                    st.markdown(f"--- **Rule {i+1}** ---")
                    rule_col = st.selectbox(f"Select Column for Rule {i+1}", df.columns, key=f"isp_rule_col_{i}")
                    
                    if rule_col:
                        if pd.api.types.is_numeric_dtype(df[rule_col]):
                            operator = st.selectbox(f"Operator for Rule {i+1}", [">", ">=", "<", "<=", "==", "!="], key=f"isp_op_num_{i}")
                            value = st.number_input(f"Value for Rule {i+1} ({df[rule_col].min()} to {df[rule_col].max()})", value=df[rule_col].median(), key=f"isp_val_num_{i}")
                            rules.append({'column': rule_col, 'operator': operator, 'value': value})
                        else: # Categorical/Object/Date
                            operator = st.selectbox(f"Operator for Rule {i+1}", ["is", "is not", "contains", "does not contain"], key=f"isp_op_cat_{i}")
                            if operator in ["is", "is not"] and df[rule_col].nunique() < 50: # Use selectbox for low cardinality
                                value = st.selectbox(f"Value for Rule {i+1}", df[rule_col].dropna().unique(), key=f"isp_val_cat_sel_{i}")
                            else: # Use text input for high cardinality or contains/does not contain
                                value = st.text_input(f"Value for Rule {i+1} (case-sensitive for 'contains')", key=f"isp_val_cat_txt_{i}")
                            rules.append({'column': rule_col, 'operator': operator, 'value': value})
                
                if rules and st.button("Create and Profile Segment", key="isp_run_profile"):
                    try:
                        segment_mask = pd.Series(True, index=df.index)
                        for rule in rules:
                            col, op, val = rule['column'], rule['operator'], rule['value']
                            if pd.api.types.is_numeric_dtype(df[col]):
                                if op == ">": segment_mask &= (df[col] > val)
                                elif op == ">=": segment_mask &= (df[col] >= val)
                                elif op == "<": segment_mask &= (df[col] < val)
                                elif op == "<=": segment_mask &= (df[col] <= val)
                                elif op == "==": segment_mask &= (df[col] == val)
                                elif op == "!=": segment_mask &= (df[col] != val)
                            else: # Categorical/Object/Date (treat date as string for contains)
                                if op == "is": segment_mask &= (df[col] == val)
                                elif op == "is not": segment_mask &= (df[col] != val)
                                elif op == "contains": segment_mask &= df[col].astype(str).str.contains(str(val), case=False, na=False)
                                elif op == "does not contain": segment_mask &= ~df[col].astype(str).str.contains(str(val), case=False, na=False)
                        
                        segment_df = df[segment_mask]
                        st.write(f"#### Segment Profile (Segment Size: {len(segment_df)} rows)")
                        if not segment_df.empty:
                            st.dataframe(segment_df.describe(include='all').T)
                            st.write("Segment Data Preview (First 100 rows):")
                            st.dataframe(segment_df.head(100))
                        else:
                            st.warning("The defined rules result in an empty segment.")
                    except Exception as e:
                        st.error(f"Error creating or profiling segment: {e}")
            else:
                st.info("Upload data to define segments.")

        # NEW TOOL 3: Survival Analysis (Kaplan-Meier)
        with st.expander("⏳ Survival Analysis (Kaplan-Meier)"):
            st.subheader("Analyze Time-to-Event Data")
            st.info("This tool performs survival analysis using the Kaplan-Meier estimator. You need a 'duration' column (time until event or censoring) and an 'event observed' column (binary: 1 if event occurred, 0 if censored).")

            if numeric_cols and (categorical_cols or numeric_cols): # Need duration and event columns
                duration_col_sa = st.selectbox("Select Duration Column (Numeric)", numeric_cols, key="sa_duration")
                event_col_sa = st.selectbox("Select Event Observed Column (Binary: 0 or 1)", numeric_cols + categorical_cols, key="sa_event")
                group_col_sa = st.selectbox("Optional: Select Grouping Column (Categorical)", [None] + categorical_cols, key="sa_group")

                if duration_col_sa and event_col_sa:
                    if st.button("Run Survival Analysis", key="run_sa"):
                        try:
                            sa_df_prep = df[[duration_col_sa, event_col_sa]].copy().dropna()
                            # Ensure event column is binary 0/1
                            if sa_df_prep[event_col_sa].nunique() == 2:
                                unique_event_vals = sorted(sa_df_prep[event_col_sa].unique())
                                # Ensure there are two unique values before trying to map
                                if len(unique_event_vals) == 2:
                                    sa_df_prep[event_col_sa] = sa_df_prep[event_col_sa].map({unique_event_vals[0]: 0, unique_event_vals[1]: 1})
                                # If nunique is 2 but unique_event_vals isn't (e.g. due to NaNs), this will be caught by the elif
                            elif not sa_df_prep[event_col_sa].isin([0,1]).all():
                                st.error(f"Event column '{event_col_sa}' must be binary (0 or 1, or two distinct values that can be mapped to 0/1).")
                                st.stop()

                            if sa_df_prep.empty:
                                st.warning("No data available for survival analysis after filtering.")
                            else:
                                kmf = KaplanMeierFitter()
                                fig_sa, ax_sa = plt.subplots()

                                if group_col_sa and group_col_sa in df.columns:
                                    sa_df_prep[group_col_sa] = df.loc[sa_df_prep.index, group_col_sa] # Add group column
                                    for name, grouped_df in sa_df_prep.groupby(group_col_sa):
                                        if not grouped_df.empty:
                                            kmf.fit(grouped_df[duration_col_sa], event_observed=grouped_df[event_col_sa], label=str(name))
                                            kmf.plot_survival_function(ax=ax_sa)
                                    ax_sa.set_title(f"Kaplan-Meier Survival Curves by '{group_col_sa}'")
                                else:
                                    kmf.fit(sa_df_prep[duration_col_sa], event_observed=sa_df_prep[event_col_sa])
                                    kmf.plot_survival_function(ax=ax_sa)
                                    ax_sa.set_title("Kaplan-Meier Survival Curve")
                                
                                ax_sa.set_xlabel("Time (Duration)")
                                ax_sa.set_ylabel("Survival Probability")
                                plt.tight_layout()
                                st.pyplot(fig_sa)

                                st.write("Median Survival Time(s):")
                                if group_col_sa and group_col_sa in df.columns:
                                    median_survival_times = sa_df_prep.groupby(group_col_sa).apply(
                                        lambda x: KaplanMeierFitter().fit(x[duration_col_sa], event_observed=x[event_col_sa]).median_survival_time_
                                    )
                                    st.dataframe(median_survival_times.rename("Median Survival Time"))
                                else:
                                    st.metric("Overall Median Survival Time", f"{kmf.median_survival_time_:.2f}" if pd.notna(kmf.median_survival_time_) else "Not Reached")

                        except ImportError:
                            st.error("The 'lifelines' library is required for Survival Analysis. Please install it (`pip install lifelines`).")
                        except Exception as e:
                            st.error(f"Error during Survival Analysis: {e}")
                else:
                    st.info("Select Duration and Event Observed columns.")
            else:
                st.info("Survival Analysis requires numeric columns for duration and a binary column for event observation.")

        # NEW TOOL 4: AI-Powered Chart-to-Text Summarizer (Gemini)
        with st.expander("🪄 AI Chart-to-Text Summarizer (Gemini)"):
            st.subheader("Get AI-Generated Summaries of Your Charts")
            st.info("Select columns to generate a basic chart, then ask AI to summarize its insights.")
            if gemini_api_key:
                if numeric_cols or categorical_cols:
                    chart_type_ai = st.selectbox("Chart Type for AI Summary", ["Bar Chart (Categorical vs Numeric)", "Scatter Plot (Numeric vs Numeric)"], key="ai_chart_type")
                    
                    fig_for_ai = None
                    chart_description_for_ai = ""

                    if chart_type_ai == "Bar Chart (Categorical vs Numeric)" and categorical_cols and numeric_cols:
                        cat_col_ai = st.selectbox("Select Categorical Column for Bar Chart", categorical_cols, key="ai_bar_cat")
                        num_col_ai = st.selectbox("Select Numeric Column for Bar Chart", numeric_cols, key="ai_bar_num")
                        agg_func_ai = st.selectbox("Aggregation for Bar Chart", ["mean", "sum", "count"], key="ai_bar_agg")
                        if cat_col_ai and num_col_ai:
                            bar_data = df.groupby(cat_col_ai)[num_col_ai].agg(agg_func_ai).reset_index()
                            fig_for_ai = px.bar(bar_data, x=cat_col_ai, y=num_col_ai, title=f"{agg_func_ai.capitalize()} of {num_col_ai} by {cat_col_ai}")
                            st.plotly_chart(fig_for_ai, use_container_width=True)
                            chart_description_for_ai = f"This is a bar chart showing the {agg_func_ai} of '{num_col_ai}' for each category in '{cat_col_ai}'. The x-axis is '{cat_col_ai}' and the y-axis is '{num_col_ai}'. Data: {bar_data.head().to_string()}"

                    elif chart_type_ai == "Scatter Plot (Numeric vs Numeric)" and len(numeric_cols) >= 2:
                        x_col_ai = st.selectbox("Select X-axis for Scatter Plot", numeric_cols, key="ai_scatter_x")
                        y_col_ai_options = [col for col in numeric_cols if col != x_col_ai]
                        if y_col_ai_options:
                            y_col_ai = st.selectbox("Select Y-axis for Scatter Plot", y_col_ai_options, key="ai_scatter_y")
                            if x_col_ai and y_col_ai:
                                fig_for_ai = px.scatter(df, x=x_col_ai, y=y_col_ai, title=f"Scatter Plot: {y_col_ai} vs {x_col_ai}")
                                st.plotly_chart(fig_for_ai, use_container_width=True)
                                chart_description_for_ai = f"This is a scatter plot showing the relationship between '{x_col_ai}' (x-axis) and '{y_col_ai}' (y-axis). Each point represents a data record. Sample data points for x: {df[x_col_ai].dropna().head().tolist()}, for y: {df[y_col_ai].dropna().head().tolist()}"
                        else:
                            st.warning("Need at least two distinct numeric columns for a scatter plot.")

                    if fig_for_ai and chart_description_for_ai:
                        if st.button("✍️ Generate AI Summary for this Chart", key="ai_summarize_chart"):
                            with st.spinner("AI is analyzing the chart..."):
                                prompt_chart_summary = f"Analyze the following chart and its underlying data. Provide a concise summary of the key insights, trends, or patterns visible. Chart Description: {chart_description_for_ai}"
                                try:
                                    model_chart_summary = genai.GenerativeModel("gemini-2.0-flash")
                                    response_chart_summary = model_chart_summary.generate_content(prompt_chart_summary)
                                    st.markdown("#### AI Chart Summary:")
                                    st.markdown(response_chart_summary.text)
                                except Exception as e:
                                    st.error(f"Gemini API Error for Chart Summary: {str(e)}")
                else:
                    st.info("This tool requires numeric or categorical columns to generate a chart for AI summary.")
            else:
                st.info("Enter your Gemini API key in the sidebar to enable AI-powered chart summaries.")

        # NEW TOOL 5: Anomaly Explanation (Feature Contribution)
        with st.expander("💡 Anomaly Explanation (Feature Comparison)"):
            st.subheader("Understand Why a Data Point is Flagged as an Anomaly")
            st.info("This tool helps explain anomalies detected by the 'Anomaly Detection Dashboard'. Select an anomaly and see how its feature values compare to typical values.")
            if 'anomalies_detected_df' in st.session_state and not st.session_state.anomalies_detected_df.empty:
                outlier_df_explain = st.session_state.anomalies_detected_df
                st.write("Detected Anomalies (from Anomaly Detection Dashboard):")
                st.dataframe(outlier_df_explain.head())

                if not outlier_df_explain.empty:
                    selected_anomaly_idx = st.selectbox("Select Anomaly Index to Explain", outlier_df_explain.index.tolist(), key="explain_anomaly_idx")
                    if selected_anomaly_idx is not None:
                        anomaly_data_point = df.loc[selected_anomaly_idx]
                        st.write(f"#### Explaining Anomaly at Index: {selected_anomaly_idx}")
                        st.dataframe(anomaly_data_point.to_frame().T)

                        st.markdown("##### Feature Comparison:")
                        explanation_found_ae = False
                        for col in df.columns: # Iterate through all columns of the original df
                            if col in anomaly_data_point and pd.notna(anomaly_data_point[col]):
                                outlier_val = anomaly_data_point[col]
                                non_anomaly_data = df.drop(outlier_df_explain.index, errors='ignore') # Data excluding all detected anomalies

                                if pd.api.types.is_numeric_dtype(df[col]) and not non_anomaly_data[col].dropna().empty:
                                    mean_val = non_anomaly_data[col].mean()
                                    std_val = non_anomaly_data[col].std()
                                    q05_val = non_anomaly_data[col].quantile(0.05)
                                    q95_val = non_anomaly_data[col].quantile(0.95)

                                    if std_val > 0: # Avoid division by zero for z-score like comparison
                                        z_score_approx = (outlier_val - mean_val) / std_val
                                        if abs(z_score_approx) > 2.5:
                                            st.write(f"- **{col}**: Value `{outlier_val:.2f}` is significantly different (approx. {z_score_approx:.1f} std devs) from the typical mean (`{mean_val:.2f}`).")
                                            explanation_found_ae = True
                                        elif outlier_val > q95_val :
                                            st.write(f"- **{col}**: Value `{outlier_val:.2f}` is in the top 5% (above `{q95_val:.2f}`). Typical mean: `{mean_val:.2f}`.")
                                            explanation_found_ae = True
                                        elif outlier_val < q05_val:
                                            st.write(f"- **{col}**: Value `{outlier_val:.2f}` is in the bottom 5% (below `{q05_val:.2f}`). Typical mean: `{mean_val:.2f}`.")
                                            explanation_found_ae = True
                                elif df[col].dtype == 'object' and not non_anomaly_data[col].dropna().empty:
                                    mode_val = non_anomaly_data[col].mode()
                                    if not mode_val.empty and outlier_val != mode_val[0]:
                                        value_freq = non_anomaly_data[col].value_counts(normalize=True)
                                        if outlier_val in value_freq and value_freq[outlier_val] < 0.05: # If category is rare
                                            st.write(f"- **{col}**: Category `'{outlier_val}'` is uncommon (occurs <5% in non-anomalies). Most common is `'{mode_val[0]}'`. ")
                                            explanation_found_ae = True
                        if not explanation_found_ae:
                            st.info("This anomaly does not show extreme deviations on individual features compared to the rest of the data based on simple statistical checks. It might be an outlier due to a combination of factors.")
            else:
                st.info("Run the 'Anomaly Detection Dashboard' first to identify outliers that can be explained here.")

        # --- ADVANCED TOOL 1: Hierarchical Clustering & Dendrogram Visualization ---
        with st.expander("🔗 ADVANCED TOOL 1: Hierarchical Clustering & Dendrogram Visualization"):
            st.subheader("Explore Data Structure with Hierarchical Clustering")
            st.info("Perform hierarchical clustering on selected numeric features. Visualize relationships with a dendrogram and define clusters by cutting the tree.")
            if len(numeric_cols) >= 2:
                hc_features = st.multiselect(
                    "Select Numeric Features for Hierarchical Clustering",
                    numeric_cols,
                    default=numeric_cols[:min(3, len(numeric_cols))],
                    key="hc_features_select"
                )
                hc_linkage_method = st.selectbox(
                    "Linkage Method",
                    ['ward', 'complete', 'average', 'single'],
                    key="hc_linkage"
                )
                hc_metric = st.selectbox(
                    "Distance Metric",
                    ['euclidean', 'cityblock', 'cosine', 'correlation'], # Common metrics
                    key="hc_metric"
                )

                if len(hc_features) >= 2:
                    if st.button("Run Hierarchical Clustering", key="run_hc"):
                        hc_data = df[hc_features].copy().dropna()
                        if hc_data.empty or len(hc_data) < 2:
                            st.warning("Not enough data after dropping NaNs for selected features.")
                        else:
                            try:
                                # Standardize data
                                scaler_hc = StandardScaler()
                                scaled_hc_data = scaler_hc.fit_transform(hc_data)

                                # Perform hierarchical clustering
                                linked = linkage(scaled_hc_data, method=hc_linkage_method, metric=hc_metric)

                                st.write("#### Dendrogram")
                                fig_dendro, ax_dendro = plt.subplots(figsize=(12, 7))
                                dendrogram(linked,
                                           orientation='top',
                                           distance_sort='descending',
                                           show_leaf_counts=True,
                                           ax=ax_dendro,
                                           truncate_mode='lastp', # Show last p merged clusters
                                           p=30, # Number of merged clusters to show
                                           leaf_rotation=90.,
                                           leaf_font_size=8.)
                                ax_dendro.set_title(f"Hierarchical Clustering Dendrogram ({hc_linkage_method} linkage, {hc_metric} metric)")
                                ax_dendro.set_xlabel("Sample index or (cluster size)")
                                ax_dendro.set_ylabel("Distance")
                                plt.tight_layout()
                                st.pyplot(fig_dendro)

                                # Option to cut the tree for clusters
                                num_clusters_hc = st.slider("Number of Clusters to Form (by cutting dendrogram)", 2, 15, 3, key="hc_num_clusters_cut")
                                from scipy.cluster.hierarchy import fcluster
                                clusters_hc = fcluster(linked, num_clusters_hc, criterion='maxclust')
                                
                                hc_data_with_clusters = hc_data.copy()
                                hc_data_with_clusters['HC_Cluster'] = clusters_hc
                                
                                # Add to main df if desired
                                if st.checkbox("Add Hierarchical Clusters to main DataFrame?", key="hc_add_to_df"):
                                    df.loc[hc_data.index, 'HC_Cluster'] = clusters_hc
                                    st.success("'HC_Cluster' column added. Rerun other analyses if needed.")
                                    st.rerun()

                                st.write("#### Cluster Profiles (Mean Values)")
                                st.dataframe(hc_data_with_clusters.groupby('HC_Cluster')[hc_features].mean())
                                st.write("#### Cluster Sizes")
                                st.dataframe(hc_data_with_clusters['HC_Cluster'].value_counts().sort_index().rename("Size"))

                            except Exception as e:
                                st.error(f"Error during Hierarchical Clustering: {e}")
                else:
                    st.warning("Select at least two numeric features for Hierarchical Clustering.")
            else:
                st.info("Hierarchical Clustering requires at least two numeric columns.")

        # --- ADVANCED TOOL 2: Latent Dirichlet Allocation (LDA) for Topic Modeling ---
        with st.expander("📜 ADVANCED TOOL 2: Latent Dirichlet Allocation (LDA) for Topic Modeling"):
            st.subheader("Discover Hidden Topics in Text Data")
            st.info("Apply LDA to a text column to identify underlying topics. Each topic is represented by a set of characteristic words.")
            text_cols_lda = df.select_dtypes(include='object').columns.tolist()
            if text_cols_lda:
                lda_text_col = st.selectbox("Select Text Column for LDA", text_cols_lda, key="lda_text_col")
                lda_num_topics = st.slider("Number of Topics", 2, 20, 5, key="lda_num_topics")
                
                col_lda1, col_lda2, col_lda3 = st.columns(3)
                with col_lda1:
                    lda_max_features = st.number_input("Max Features (Vocab Size)", 100, 10000, 1000, step=100, key="lda_max_features")
                with col_lda2:
                    # min_df can be an int (absolute count) or float (proportion)
                    # For simplicity, let's use int here.
                    lda_min_df = st.number_input("Min Document Frequency (min_df)", 1, 100, 2, step=1, key="lda_min_df", help="Minimum number of documents a word must appear in.")
                with col_lda3:
                    lda_max_df = st.slider("Max Document Frequency (max_df)", 0.50, 1.0, 0.95, step=0.01, key="lda_max_df", help="Maximum proportion of documents a word can appear in.")
                
                lda_top_n_words = st.number_input("Top N Words per Topic to Display", 5, 20, 10, key="lda_top_n_words")

                if lda_text_col:
                    if st.button("Run LDA Topic Modeling", key="run_lda"):
                        lda_data = df[lda_text_col].astype(str).dropna()
                        if lda_data.empty or len(lda_data) < lda_num_topics:
                            st.warning("Not enough text data or fewer documents than the specified number of topics.")
                        else:
                            try:
                                # Create Document-Term Matrix
                                vectorizer = CountVectorizer(max_df=lda_max_df, min_df=lda_min_df, max_features=lda_max_features, stop_words='english')
                                dtm = vectorizer.fit_transform(lda_data)
                                feature_names_lda = vectorizer.get_feature_names_out()

                                if dtm.shape[1] == 0: # Check if any terms remained
                                    st.error("After pruning with the current min_df, max_df, and max_features settings, no terms remain in the vocabulary. Please try adjusting these parameters (e.g., lower min_df, higher max_df, or increase max_features).")
                                    st.stop()

                                # Fit LDA model
                                lda_model = LatentDirichletAllocation(n_components=lda_num_topics, random_state=42, learning_method='online')
                                lda_model.fit(dtm)

                                st.write("#### Top Words per Topic:")
                                topics_display = {}
                                for topic_idx, topic in enumerate(lda_model.components_):
                                    top_words_indices = topic.argsort()[:-lda_top_n_words - 1:-1]
                                    top_words = [feature_names_lda[i] for i in top_words_indices]
                                    topics_display[f"Topic {topic_idx+1}"] = ", ".join(top_words)
                                st.json(topics_display)

                                # Display Document-Topic Distribution (sample) by default
                                doc_topic_dist = lda_model.transform(dtm)
                                doc_topic_df = pd.DataFrame(doc_topic_dist, columns=[f"Topic {i+1}" for i in range(lda_num_topics)])
                                st.write("#### Document-Topic Distribution (First 100 rows):")
                                st.dataframe(doc_topic_df.head(100))

                            except Exception as e:
                                st.error(f"Error during LDA Topic Modeling: {e}")
            else:
                st.info("LDA Topic Modeling requires a text (object/string type) column.")

        # --- ADVANCED TOOL 3: Model Interpretability with PDP/ICE Plots ---
        with st.expander("🔍 ADVANCED TOOL 3: Model Interpretability (PDP/ICE Plots)"):
            st.subheader("Understand Model Predictions with Partial Dependence and ICE Plots")
            st.info("Visualize how feature values affect model predictions. A simple Random Forest model will be trained for demonstration.")
            if len(numeric_cols) >= 1 and (len(categorical_cols) >=1 or len(numeric_cols) >=2) : # Need target and at least one feature
                pdp_target_col = st.selectbox("Select Target Variable for Model", numeric_cols + categorical_cols, key="pdp_target")
                
                pdp_feature_options = [col for col in numeric_cols + categorical_cols if col != pdp_target_col]
                pdp_model_features = st.multiselect("Select Features for Model Training", pdp_feature_options, default=pdp_feature_options[:min(3, len(pdp_feature_options))], key="pdp_model_features")

                if pdp_target_col and pdp_model_features:
                    pdp_plot_features = st.multiselect("Select Features for PDP/ICE Plots (subset of trained features)", pdp_model_features, default=pdp_model_features[:min(2, len(pdp_model_features))], key="pdp_plot_features")

                    if st.button("Train Model & Generate PDP/ICE Plots", key="run_pdp_ice"):
                        try:
                            pdp_df_prep = df[[pdp_target_col] + pdp_model_features].copy().dropna()
                            
                            # Preprocessing
                            y_pdp = pdp_df_prep[pdp_target_col]
                            X_pdp = pdp_df_prep[pdp_model_features]
                            
                            is_classification_pdp = False
                            if y_pdp.dtype == 'object' or y_pdp.nunique() <= 10: # Heuristic for classification
                                is_classification_pdp = True
                                le_pdp = LabelEncoder()
                                y_pdp = le_pdp.fit_transform(y_pdp)
                                pdp_model = RandomForestClassifier(random_state=42, n_estimators=50)
                            else:
                                pdp_model = RandomForestRegressor(random_state=42, n_estimators=50)

                            X_pdp_processed = pd.get_dummies(X_pdp, drop_first=True)
                            pdp_model.fit(X_pdp_processed, y_pdp)
                            st.success(f"Model ({'Classifier' if is_classification_pdp else 'Regressor'}) trained successfully.")

                            if pdp_plot_features:
                                st.write("#### Partial Dependence Plots (PDP) & Individual Conditional Expectation (ICE) Plots")
                                for feature_to_plot in pdp_plot_features:
                                    if feature_to_plot in X_pdp_processed.columns: # Ensure feature is in processed columns
                                        st.markdown(f"##### Plots for Feature: {feature_to_plot}")
                                        fig_pdp_ice, ax_pdp_ice = plt.subplots(figsize=(10, 6))
                                        display = PartialDependenceDisplay.from_estimator(
                                            pdp_model,
                                            X_pdp_processed,
                                            features=[feature_to_plot],
                                            kind='both', # Show PDP and ICE
                                            ice_lines_kw={"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5},
                                            pd_line_kw={"color": "tab:orange", "linestyle": "--", "linewidth": 2},
                                            ax=ax_pdp_ice
                                        )
                                        ax_pdp_ice.set_title(f"PDP and ICE for {feature_to_plot}")
                                        st.pyplot(fig_pdp_ice)
                                    else:
                                        st.warning(f"Feature '{feature_to_plot}' not found in processed data (possibly removed during one-hot encoding or was part of a dropped category).")
                            else:
                                st.info("Select features to plot for PDP/ICE.")
                        except Exception as e:
                            st.error(f"Error during PDP/ICE generation: {e}")
                else:
                    st.info("Select a target variable and features for model training.")
            else:
                st.info("PDP/ICE plots require numeric and/or categorical columns for target and features.")

        # --- ADVANCED TOOL 4: Survival Regression with Cox Proportional Hazards Model ---
        with st.expander("📈 ADVANCED TOOL 4: Survival Regression (Cox PH Model)"):
            st.subheader("Model Time-to-Event Data with Covariates")
            st.info("Use the Cox Proportional Hazards model to understand how different features affect survival time. Requires duration, event observed, and covariate columns.")
            if numeric_cols and (categorical_cols or len(numeric_cols) >=2): # Need duration, event, and at least one covariate
                cox_duration_col = st.selectbox("Select Duration Column (Numeric)", numeric_cols, key="cox_duration")
                cox_event_col = st.selectbox("Select Event Observed Column (Binary: 0 or 1)", numeric_cols + categorical_cols, key="cox_event")
                
                cox_covariate_options = [col for col in numeric_cols + categorical_cols if col not in [cox_duration_col, cox_event_col]]
                cox_covariates = st.multiselect("Select Covariate Columns", cox_covariate_options, default=cox_covariate_options[:min(2, len(cox_covariate_options))], key="cox_covariates")

                if cox_duration_col and cox_event_col and cox_covariates:
                    if st.button("Run Cox Proportional Hazards Model", key="run_cox_ph"):
                        try:
                            cox_df_prep = df[[cox_duration_col, cox_event_col] + cox_covariates].copy().dropna()
                            
                            # Ensure event column is binary 0/1
                            if cox_df_prep[cox_event_col].nunique() == 2:
                                unique_event_vals_cox = sorted(cox_df_prep[cox_event_col].unique())
                                if len(unique_event_vals_cox) == 2: # Check again after potential dropna
                                    cox_df_prep[cox_event_col] = cox_df_prep[cox_event_col].map({unique_event_vals_cox[0]: 0, unique_event_vals_cox[1]: 1})
                            elif not cox_df_prep[cox_event_col].isin([0,1]).all():
                                st.error(f"Event column '{cox_event_col}' must be binary (0 or 1) or have two distinct values.")
                                st.stop()

                            if cox_df_prep.empty or len(cox_df_prep) < 10: # Need some data
                                st.warning("Not enough data for Cox PH model after filtering.")
                            else:
                                # One-hot encode categorical covariates
                                cox_df_processed = pd.get_dummies(cox_df_prep, columns=[col for col in cox_covariates if df[col].dtype=='object'], drop_first=True)
                                
                                cph = CoxPHFitter()
                                cph.fit(cox_df_processed, duration_col=cox_duration_col, event_col=cox_event_col)

                                st.write("#### Cox PH Model Summary:")
                                st.dataframe(cph.summary)

                                st.write("#### Hazard Ratios (exp(coef)):")
                                st.dataframe(np.exp(cph.params_).rename("Hazard Ratio (HR)"))
                                st.caption("HR > 1: Increased hazard (shorter survival) for a unit increase in covariate.")
                                st.caption("HR < 1: Decreased hazard (longer survival) for a unit increase in covariate.")

                                if len(cox_df_processed.columns) -2 <= 10: # Limit plotting if too many covariates
                                    st.write("#### Coefficient Plot:")
                                    fig_cph_coeffs, ax_cph_coeffs = plt.subplots(figsize=(8, max(4, len(cph.params_)*0.5)))
                                    cph.plot(ax=ax_cph_coeffs)
                                    plt.tight_layout()
                                    st.pyplot(fig_cph_coeffs)

                        except ImportError:
                            st.error("The 'lifelines' library is required for Cox PH Model. Please install it (`pip install lifelines`).")
                        except Exception as e:
                            st.error(f"Error during Cox PH Model: {e}")
                else:
                    st.info("Select Duration, Event Observed, and at least one Covariate column.")
            else:
                st.info("Cox PH Model requires numeric columns for duration, a binary event column, and covariate columns (numeric or categorical).")

        # --- ADVANCED TOOL 5: Customer Lifetime Value (CLV) Profiler ---
        with st.expander("💰 ADVANCED TOOL 5: Customer Lifetime Value (CLV) Profiler"):
            st.subheader("Segment and Profile Customers by Simplified CLV")
            st.info("Calculate a simplified CLV (e.g., total spend), segment customers, and profile their characteristics.")
            if categorical_cols and numeric_cols and date_cols: # Need ID, Amount, Date
                clv_customer_id_col = st.selectbox("Select Customer ID Column", categorical_cols + numeric_cols, key="clv_cust_id")
                clv_date_col = st.selectbox("Select Order Date Column", date_cols, key="clv_date")
                clv_amount_col = st.selectbox("Select Order Amount Column", numeric_cols, key="clv_amount")

                clv_profiling_features_num = st.multiselect(
                    "Select Numeric Features for Profiling CLV Segments",
                    [col for col in numeric_cols if col not in [clv_amount_col]],
                    default=[col for col in numeric_cols if col not in [clv_amount_col]][:min(2, len(numeric_cols)-1)] if len(numeric_cols)>1 else [],
                    key="clv_profile_num_feats"
                )
                clv_profiling_features_cat = st.multiselect(
                    "Select Categorical Features for Profiling CLV Segments",
                    [col for col in categorical_cols if col != clv_customer_id_col],
                    default=[col for col in categorical_cols if col != clv_customer_id_col][:min(2, len(categorical_cols)-1)] if len(categorical_cols)>1 else [],
                    key="clv_profile_cat_feats"
                )

                if clv_customer_id_col and clv_date_col and clv_amount_col:
                    if st.button("Calculate CLV & Profile Segments", key="run_clv_profiler"):
                        try:
                            clv_df_prep = df[[clv_customer_id_col, clv_date_col, clv_amount_col]].copy().dropna()
                            if clv_df_prep.empty:
                                st.warning("Not enough data for CLV calculation after filtering.")
                            else:
                                # Simplified CLV: Total spend per customer
                                customer_clv = clv_df_prep.groupby(clv_customer_id_col)[clv_amount_col].sum().reset_index()
                                customer_clv.rename(columns={clv_amount_col: 'SimplifiedCLV'}, inplace=True)

                                # Segment by CLV quantiles (e.g., Low, Medium, High)
                                customer_clv['CLV_Segment'] = pd.qcut(customer_clv['SimplifiedCLV'], q=3, labels=["Low Value", "Medium Value", "High Value"], duplicates='drop')

                                st.write("#### CLV Segmentation Summary:")
                                st.dataframe(customer_clv['CLV_Segment'].value_counts().reset_index().rename(columns={'index': 'Segment', 'CLV_Segment': 'Count'}))

                                # Merge CLV segments back to original df for profiling
                                df_with_clv_segment = pd.merge(df, customer_clv[[clv_customer_id_col, 'CLV_Segment']], on=clv_customer_id_col, how='left')

                                if clv_profiling_features_num or clv_profiling_features_cat:
                                    st.write("#### CLV Segment Profiles:")
                                    profile_agg_dict = {'SegmentSize': (clv_customer_id_col, 'nunique')} # Count unique customers per segment
                                    for p_col in clv_profiling_features_num:
                                        profile_agg_dict[f'{p_col}_mean'] = (p_col, 'mean')
                                    for p_col_cat in clv_profiling_features_cat:
                                        profile_agg_dict[f'{p_col_cat}_mode'] = (p_col_cat, lambda x: x.mode()[0] if not x.mode().empty else 'N/A')
                                    
                                    clv_segment_profiles = df_with_clv_segment.groupby('CLV_Segment').agg(**profile_agg_dict).reset_index()
                                    st.dataframe(clv_segment_profiles)

                                    # Visualizations
                                    for p_col_viz in clv_profiling_features_num:
                                        fig_clv_prof_num = px.box(df_with_clv_segment, x='CLV_Segment', y=p_col_viz, color='CLV_Segment',
                                                                  title=f"Distribution of '{p_col_viz}' by CLV Segment",
                                                                  category_orders={"CLV_Segment": ["Low Value", "Medium Value", "High Value"]})
                                        st.plotly_chart(fig_clv_prof_num, use_container_width=True)

                                    for p_col_cat_viz in clv_profiling_features_cat:
                                        clv_cat_summary = df_with_clv_segment.groupby('CLV_Segment')[p_col_cat_viz].value_counts(normalize=True).mul(100).rename('Percentage').reset_index()
                                        fig_clv_prof_cat = px.bar(clv_cat_summary, x='CLV_Segment', y='Percentage', color=p_col_cat_viz,
                                                                  title=f"Distribution of '{p_col_cat_viz}' by CLV Segment", barmode='group',
                                                                  category_orders={"CLV_Segment": ["Low Value", "Medium Value", "High Value"]})
                                        st.plotly_chart(fig_clv_prof_cat, use_container_width=True)
                                else:
                                    st.info("Select features to profile the CLV segments.")
                        except Exception as e:
                            st.error(f"Error during CLV Profiling: {e}")
                else:
                    st.info("Select Customer ID, Date, and Amount columns for CLV analysis.")
            else:
                st.info("CLV Profiler requires categorical/numeric ID, date, and numeric amount columns.")

        # --- ADVANCED TOOL 11: SQL Query Workbench ---
        with st.expander("🗃️ SQL Query Workbench (Data Interaction Category)", expanded=True):
            st.subheader("Execute SQL Queries on Your DataFrames")
            st.info("Select a dataset and write SQL queries to analyze it. The DataFrame will be treated as a table within the query context.")

            if not datasets: # Check if datasets dictionary is populated
                st.warning("Please upload at least one dataset to use the SQL Workbench.")
            else:
                sql_selected_dataset_name = st.selectbox(
                    "Select Dataset for SQL Query",
                    list(datasets.keys()),
                    key="sql_workbench_dataset_select"
                )
                if sql_selected_dataset_name:
                    df_to_query_sql = datasets[sql_selected_dataset_name]
                    # Sanitize dataset name to be a valid SQL table name
                    table_name_sql = re.sub(r'[^A-Za-z0-9_]+', '_', sql_selected_dataset_name)
                    if not table_name_sql[0].isalpha() and table_name_sql[0] != '_': # Ensure it starts with letter or underscore
                        table_name_sql = "_" + table_name_sql

                    query_sql = st.text_area("Enter your SQL Query:", height=150, key="sql_query_input",
                                             placeholder=f"Example: SELECT * FROM {table_name_sql} WHERE YourColumn > 10 LIMIT 100;")

                    # Use HTML details tag for collapsible examples to avoid nested expanders
                    st.markdown(f"""
                    <details>
                        <summary style="cursor:pointer; color: #A0AEC0; font-weight: 600;">💡 SQL Query Examples (click to expand)</summary>
                        <div style="padding-top: 10px;">
                        Replace `YourTable` with <code>{table_name_sql}</code> (or the name of your selected table) and adjust column names as needed.
                        <br><br>
                        <strong>🌟 1. Top 5 Most Viewed Movies</strong>
                        <pre><code class="language-sql">
SELECT title, views_millions
FROM {table_name_sql}
WHERE type = 'Movie' -- Assuming a 'type' column exists
ORDER BY views_millions DESC
LIMIT 5;
                        </code></pre>
                        <strong>📊 2. Average Budget by Decade and Type</strong>
                        <pre><code class="language-sql">
SELECT decade, type, AVG(budget_millions) AS avg_budget
FROM {table_name_sql} -- Assuming 'decade', 'type', 'budget_millions' columns
GROUP BY decade, type
ORDER BY avg_budget DESC;
                        </code></pre>
                        <strong>🏆 3. Directors with Most Award-Winning Shows</strong>
                        <pre><code class="language-sql">
SELECT director, SUM(awards_won) AS total_awards
FROM {table_name_sql} -- Assuming 'director', 'awards_won' columns
GROUP BY director
ORDER BY total_awards DESC
LIMIT 10;
                        </code></pre>
                        <strong>🎬 4. Top 5 Genres with Highest Average Views</strong>
                        <pre><code class="language-sql">
SELECT listed_in, AVG(views_millions) AS avg_views
FROM {table_name_sql} -- Assuming 'listed_in', 'views_millions' columns
GROUP BY listed_in
ORDER BY avg_views DESC
LIMIT 5;
                        </code></pre>
                        And many more! (See full list for other examples like most popular country, high budget movies, shows per language, sound mix types, nomination analysis, multi-genre shows).
                        Remember to adapt column names like <code>title</code>, <code>views_millions</code>, <code>type</code>, <code>budget_millions</code>, <code>director</code>, <code>awards_won</code>, <code>listed_in</code>, <code>country</code>, <code>language</code>, <code>sound_mix</code>, <code>nomination_for_best_picture</code> to match your dataset.
                        </div>
                    </details>
                    """, unsafe_allow_html=True)

                    if st.button("🚀 Run SQL Query", key="run_sql_query_button"):
                        if query_sql:
                            try:
                                con = duckdb.connect(database=':memory:', read_only=False)
                                # Register DataFrame as a table in DuckDB
                                con.register(table_name_sql, df_to_query_sql)
                                result_sql_df = con.execute(query_sql).fetchdf()
                                con.close()

                                st.write("#### SQL Query Results:")
                                if not result_sql_df.empty:
                                    st.dataframe(result_sql_df)
                                else:
                                    st.info("Query executed successfully, but returned no results.")
                            except Exception as e:
                                st.error(f"SQL Query Error: {str(e)}")
                        else:
                            st.warning("Please enter an SQL query.")

        # --- ADVANCED TOOL 13: Excel-like Query Workbench ---
        with st.expander("📊 Excel-like Query Workbench", expanded=True):
            st.subheader("Query DataFrames with Excel-like Expressions")
            st.info("Select a dataset and use pandas `query()` syntax to filter and analyze it. This is useful for Excel users familiar with formula-based filtering.")

            if not datasets:
                st.warning("Please upload at least one dataset to use the Excel-like Query Workbench.")
            else:
                excel_query_selected_dataset_name = st.selectbox(
                    "Select Dataset for Excel-like Query",
                    list(datasets.keys()),
                    key="excel_query_dataset_select"
                )
                if excel_query_selected_dataset_name:
                    df_to_query_excel = datasets[excel_query_selected_dataset_name]

                    excel_query_expression = st.text_area(
                        "Enter your Query Expression (pandas `df.query()` syntax):",
                        height=100,
                        key="excel_query_expression_input",
                        placeholder="Example: `YourColumnName > 100 and AnotherColumn == 'SomeValue'` or `(`ColumnA` + `ColumnB`) / 2 > `ColumnC``"
                    )
                    # st.caption("Use backticks (`) around column names with spaces or special characters. Refer to pandas `DataFrame.query()` documentation for syntax.")

                    # Collapsible Examples for Excel-like Query Workbench
                    st.markdown(f"""
                    <details>
                        <summary style="cursor:pointer; color: #A0AEC0; font-weight: 600;">💡 Excel-like Query Examples (click to expand)</summary>
                        <div style="padding-top: 10px;">
                        Remember to replace placeholder column names (e.g., <code>NumericColumn1</code>, <code>StringColumnA</code>, <code>`Column With Spaces`</code>) with actual column names from your selected dataset (<code>{excel_query_selected_dataset_name}</code>).
                        <br><br>
                        <strong>⚠️ Examples that might require data type preprocessing:</strong>
                        <pre><code class="language-plaintext">
# If 'ActualNumericColumn' is incorrectly a string/object type, this numeric comparison will error.
# Ensure it's converted to a numeric type for correct results.
ActualNumericColumn &lt; 0.4

# Similarly, if 'TP_Column' (which should be numeric) is a string, this will error:
TP_Column &gt; 500
                        </code></pre>
                        <p style="color: #FFA500;">If you encounter errors like "'&lt;' not supported between instances of 'str' and 'float'", use the "Smart Data Type Detection & Conversion" tool to convert the relevant column to a numeric type for proper numerical comparisons.</p>

                        <strong>✅ Examples that generally work (mind the data types for comparisons):</strong>
                        <pre><code class="language-plaintext">
# Correct string comparison for a text column
StringColumnA == 'SomeValue'

# Numeric comparison (assumes `Numeric Column With Spaces` is a numeric type)
`Numeric Column With Spaces` &gt; 2000

# String comparison with a string literal that looks like a number.
# This works if `StringTPColumn` is a string type.
# Note: This is a lexicographical (text) comparison, e.g., '60' > '500' would be TRUE.
`StringTPColumn` &gt; '500'

# Comparing two numeric columns (ensure both are numeric type)
NumericColumn1 &lt; 0.4
`NumericColumnA` &lt;= `NumericColumnB`

# Combining conditions
`CategoricalColumnX` == 'CategoryX_ValueY' and NumericColumn3 &gt; 1000

# Using string methods (ensure the column is string type)
`AnotherTextColumn`.str.startswith('Prefix')
                        </code></pre>
                        <strong>🐼 1. Movies with Views Over 500 Million</strong>
                        <pre><code class="language-plaintext">
views_millions > 500
                        </code></pre>
                        <strong>🐣 2. Top TV Shows with More Than 2 Seasons</strong>
                        <pre><code class="language-plaintext">
# Query for TV Shows with 'Seasons' in duration
type == 'TV Show' and duration.str.contains('Seasons')
# Note: Further pandas operations would be needed to extract the number of seasons and filter > 2, e.g.:
# df_filtered = df.query("type == 'TV Show' and duration.str.contains('Seasons')")
# df_filtered['num_seasons'] = df_filtered['duration'].str.extract('(\d+)').astype(int)
# result = df_filtered[df_filtered['num_seasons'] > 2]
                        </code></pre>
                        <strong>🍿 3. Movies Released After 2020 with Specific Aspect Ratios</strong>
                        <pre><code class="language-plaintext">
type == 'Movie' and release_year > 2020 and aspect_ratio in ['16:9', '2.39:1']
                        </code></pre>
                        <strong>🌍 4. Content by UK or Germany with >300M Views</strong>
                        <pre><code class="language-plaintext">
country in ['United Kingdom', 'Germany'] and views_millions > 300
                        </code></pre>
                        <strong>🏆 5. Award-Winning Directors with ≥40 Awards</strong>
                        <pre><code class="language-plaintext">
# Query for records with awards_won >= 40
awards_won >= 40
# Note: To select specific columns and drop duplicates, you'd apply pandas methods after the query:
# df_queried = df.query("awards_won >= 40")
# result = df_queried[['title', 'director', 'awards_won']].drop_duplicates()
                        </code></pre>
                        Refer to the <a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html" target="_blank">pandas DataFrame.query() documentation</a> for more syntax details. Use backticks (`) around column names with spaces or special characters (e.g., `My Column-Name`).
                        </div>
                    </details>
                    """, unsafe_allow_html=True)

                    if st.button("🔍 Run Excel-like Query", key="run_excel_query_button"):
                        if excel_query_expression:
                            try:
                                result_excel_df = df_to_query_excel.query(excel_query_expression)
                                st.write("#### Query Results:")
                                st.dataframe(result_excel_df)
                            except Exception as e:
                                st.error(f"Excel-like Query Error: {str(e)}")
                        else:
                            st.warning("Please enter an Excel-like query expression.")

        # --- ADVANCED TOOL 14: Enhanced Model Explainability with SHAP ---
        with st.expander("✨ ADVANCED TOOL 6: Enhanced Model Explainability with SHAP"):
            st.subheader("Explain Model Predictions with SHAP")
            st.info("Use SHAP (SHapley Additive exPlanations) to understand feature contributions to model predictions. A Random Forest model will be trained for demonstration.")
            if not df.empty:
                shap_target_col = st.selectbox("Select Target Variable for SHAP Model", df.columns, key="shap_target")
                shap_feature_options = [col for col in df.columns if col != shap_target_col]
                shap_features = st.multiselect("Select Features for SHAP Model Training", shap_feature_options, default=shap_feature_options[:min(5, len(shap_feature_options))], key="shap_model_features")

                if shap_target_col and shap_features:
                    if st.button("Train Model & Generate SHAP Plots", key="run_shap"):
                        try:
                            shap_df_prep = df[[shap_target_col] + shap_features].copy().dropna()
                            y_shap = shap_df_prep[shap_target_col]
                            X_shap = shap_df_prep[shap_features]
                            X_shap_processed = pd.get_dummies(X_shap, drop_first=True)

                            imputer_shap = SimpleImputer(strategy='median')
                            X_shap_imputed = imputer_shap.fit_transform(X_shap_processed)
                            X_shap_imputed_df = pd.DataFrame(X_shap_imputed, columns=X_shap_processed.columns, index=X_shap_processed.index)
                            
                            y_aligned_shap = y_shap.loc[X_shap_imputed_df.index].dropna()
                            X_shap_final = X_shap_imputed_df.loc[y_aligned_shap.index]

                            if X_shap_final.empty or y_aligned_shap.empty:
                                st.error("Not enough data after preprocessing for SHAP analysis.")
                                st.stop()

                            is_classification_shap = False
                            if pd.api.types.is_numeric_dtype(y_aligned_shap.dtype):
                                if y_aligned_shap.nunique() <= 10 and y_aligned_shap.nunique() > 1:
                                    is_classification_shap = True
                                elif y_aligned_shap.nunique() == 1:
                                    st.error(f"Target column '{shap_target_col}' has only one unique value. Cannot train model.")
                                    st.stop()
                            else:
                                is_classification_shap = True

                            if is_classification_shap:
                                le_shap = LabelEncoder()
                                y_shap_encoded = le_shap.fit_transform(y_aligned_shap)
                                if len(le_shap.classes_) <= 1:
                                    st.error(f"Target column '{shap_target_col}' effectively has only one class after encoding.")
                                    st.stop()
                                shap_model = RandomForestClassifier(random_state=42, n_estimators=50)
                            else:
                                y_shap_encoded = y_aligned_shap
                                shap_model = RandomForestRegressor(random_state=42, n_estimators=50)

                            shap_model.fit(X_shap_final, y_shap_encoded)
                            st.success(f"Model ({'Classifier' if is_classification_shap else 'Regressor'}) trained successfully for SHAP analysis.")

                            explainer = shap.TreeExplainer(shap_model)
                            shap_values = explainer.shap_values(X_shap_final)

                            st.write("#### SHAP Summary Plot (Bar)")
                            st.pyplot(shap.summary_plot(shap_values, X_shap_final, plot_type="bar", show=False))
                            plt.clf() # Clear the current figure to avoid overlap

                            st.write("#### SHAP Summary Plot (Dot/Violin)")
                            st.pyplot(shap.summary_plot(shap_values, X_shap_final, show=False))
                            plt.clf()

                            if is_classification_shap and isinstance(shap_values, list) and len(shap_values) > 1: # For multi-class classification
                                st.info("For multi-class classification, SHAP values are generated per class. Showing summary for class 1.")
                                st.pyplot(shap.summary_plot(shap_values[1], X_shap_final, plot_type="bar", class_names=le_shap.classes_, show=False))
                                plt.clf()

                        except Exception as e:
                            st.error(f"Error during SHAP analysis: {e}")
                else:
                    st.info("Select a target variable and features for the model.")
            else:
                st.info("Upload data to perform SHAP analysis.")

        # --- ADVANCED TOOL 1: Advanced Time Series Forecasting with Prophet ---
        with st.expander("📈 ADVANCED TOOL 7: Advanced Time Series Forecasting with Prophet"):
            st.subheader("Forecast Time Series Data using Prophet")
            st.info("Prophet is robust to missing data and shifts in trend, and typically handles seasonality well. Requires a date column and a numeric value column.")
            if date_cols and numeric_cols:
                prophet_date_col = st.selectbox("Select Date Column for Prophet", date_cols, key="prophet_date")
                prophet_value_col = st.selectbox("Select Value Column for Prophet", numeric_cols, key="prophet_value")
                prophet_periods = st.number_input("Periods to Forecast (Prophet)", 1, 365 * 2, 30, key="prophet_periods")
                prophet_freq = st.selectbox("Forecast Frequency", ['D', 'W', 'M'], index=0, key="prophet_freq", help="D: Day, W: Week, M: Month")

                if prophet_date_col and prophet_value_col:
                    if st.button("Run Prophet Forecast", key="run_prophet"):
                        try:
                            prophet_df_prep = df[[prophet_date_col, prophet_value_col]].copy().dropna()
                            prophet_df_prep.columns = ['ds', 'y'] # Prophet requires these column names
                            prophet_df_prep['ds'] = pd.to_datetime(prophet_df_prep['ds'])
                            prophet_df_prep = prophet_df_prep.sort_values('ds')

                            if len(prophet_df_prep) < 2:
                                st.warning("Not enough data points for Prophet after filtering (need at least 2).")
                            else:
                                model_prophet = Prophet()
                                model_prophet.fit(prophet_df_prep)
                                future_df = model_prophet.make_future_dataframe(periods=prophet_periods, freq=prophet_freq)
                                forecast_df = model_prophet.predict(future_df)

                                st.write("#### Prophet Forecast Plot")
                                fig_prophet_forecast = prophet_plot_plotly(model_prophet, forecast_df)
                                st.plotly_chart(fig_prophet_forecast, use_container_width=True)

                                st.write("#### Prophet Forecast Components")
                                fig_prophet_components = prophet_plot_components_plotly(model_prophet, forecast_df)
                                st.plotly_chart(fig_prophet_components, use_container_width=True)

                                st.write("#### Forecast Data (Last 10 periods + Forecast)")
                                st.dataframe(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(prophet_periods + 10))

                        except ImportError:
                            st.error("The 'prophet' library is required for this feature. Please install it (`pip install prophet`).")
                        except Exception as e:
                            st.error(f"Error during Prophet forecasting: {e}")
                else:
                    st.info("Select a date column and a numeric value column.")
            else:
                st.info("Prophet forecasting requires a date column and a numeric value column.")

        # --- ADVANCED TOOL 2: Interactive Chart Customization & Drill-Downs ---
        with st.expander("🎨 ADVANCED TOOL 8: Interactive Chart Customization"):
            st.subheader("Build Customized Interactive Charts")
            st.info("Select chart type, axes, and additional options like color, faceting for deeper visual exploration.")
            if not df.empty:
                chart_type_interactive = st.selectbox("Select Chart Type", ["Scatter", "Bar", "Line", "Histogram", "Box"], key="interactive_chart_type")

                x_col_interactive = st.selectbox("Select X-axis", [None] + df.columns.tolist(), key="interactive_x")
                y_col_interactive = st.selectbox("Select Y-axis", [None] + df.columns.tolist(), key="interactive_y")
                
                color_col_interactive = st.selectbox("Color by (Categorical Column - Optional)", [None] + categorical_cols, key="interactive_color")
                facet_row_interactive = st.selectbox("Facet by Row (Categorical Column - Optional)", [None] + categorical_cols, key="interactive_facet_row")
                facet_col_interactive = st.selectbox("Facet by Column (Categorical Column - Optional)", [None] + categorical_cols, key="interactive_facet_col")
                
                agg_func_interactive = None
                if chart_type_interactive in ["Bar", "Line"] and y_col_interactive and df[y_col_interactive].dtype == np.number:
                    agg_func_interactive = st.selectbox("Aggregation Function for Y-axis (if applicable)", ["None (raw values)", "mean", "sum", "count"], key="interactive_agg")
                    if agg_func_interactive == "None (raw values)": agg_func_interactive = None

                if x_col_interactive and (y_col_interactive or chart_type_interactive == "Histogram"):
                    try:
                        fig_interactive = None
                        plot_df_interactive = df.copy()
                        
                        # Apply aggregation if selected
                        if agg_func_interactive and x_col_interactive and y_col_interactive:
                            grouping_cols = [x_col_interactive]
                            if color_col_interactive: grouping_cols.append(color_col_interactive)
                            if facet_row_interactive: grouping_cols.append(facet_row_interactive)
                            if facet_col_interactive: grouping_cols.append(facet_col_interactive)
                            
                            plot_df_interactive = plot_df_interactive.groupby(list(set(grouping_cols)))[y_col_interactive].agg(agg_func_interactive).reset_index()

                        if chart_type_interactive == "Scatter":
                            fig_interactive = px.scatter(plot_df_interactive, x=x_col_interactive, y=y_col_interactive, color=color_col_interactive, facet_row=facet_row_interactive, facet_col=facet_col_interactive, title=f"Interactive Scatter: {y_col_interactive} vs {x_col_interactive}")
                        elif chart_type_interactive == "Bar":
                            fig_interactive = px.bar(plot_df_interactive, x=x_col_interactive, y=y_col_interactive, color=color_col_interactive, facet_row=facet_row_interactive, facet_col=facet_col_interactive, title=f"Interactive Bar: {y_col_interactive} by {x_col_interactive}")
                        elif chart_type_interactive == "Line":
                             fig_interactive = px.line(plot_df_interactive, x=x_col_interactive, y=y_col_interactive, color=color_col_interactive, facet_row=facet_row_interactive, facet_col=facet_col_interactive, title=f"Interactive Line: {y_col_interactive} over {x_col_interactive}", markers=True)
                        elif chart_type_interactive == "Histogram":
                            fig_interactive = px.histogram(plot_df_interactive, x=x_col_interactive, color=color_col_interactive, facet_row=facet_row_interactive, facet_col=facet_col_interactive, title=f"Interactive Histogram: {x_col_interactive}", marginal="rug")
                        elif chart_type_interactive == "Box":
                            fig_interactive = px.box(plot_df_interactive, x=x_col_interactive, y=y_col_interactive, color=color_col_interactive, facet_row=facet_row_interactive, facet_col=facet_col_interactive, title=f"Interactive Box Plot: {y_col_interactive} by {x_col_interactive}")
                        
                        if fig_interactive:
                            st.plotly_chart(fig_interactive, use_container_width=True)
                        else:
                            st.info("Chart could not be generated with current selections.")
                    except Exception as e:
                        st.error(f"Error generating interactive chart: {e}")
                else:
                    st.info("Select X-axis and Y-axis (or just X-axis for Histogram) to generate a chart.")
            else:
                st.info("Upload data to use the interactive chart builder.")

        # --- ADVANCED TOOL 4: Anomaly Investigation & Explanation ---
        with st.expander("🕵️ ADVANCED TOOL 9: Anomaly Investigation & Explanation"):
            st.subheader("Investigate and Explain Detected Anomalies")
            st.info("This tool helps explain anomalies detected by the 'Anomaly Detection Dashboard'. First, review feature comparisons, then optionally use AI for a narrative explanation.")

            if not gemini_api_key:
                st.warning("Please enter your Gemini API key in the sidebar to use the AI explanation feature of this tool.")
            elif 'anomalies_detected_df' not in st.session_state or st.session_state.anomalies_detected_df.empty:
                st.info("No anomalies detected yet. Please run the 'Anomaly Detection Dashboard' (Feature 11) first to identify anomalies.")
            else:
                anomalies_for_ai_df = st.session_state.anomalies_detected_df
                st.write("Detected Anomalies (available for AI investigation):")
                st.dataframe(anomalies_for_ai_df.head())

                selected_anomaly_index_ai = st.selectbox(
                    "Select Anomaly Index for AI Investigation",
                    anomalies_for_ai_df.index.tolist(),
                    key="ai_anomaly_select_idx"
                )

                if selected_anomaly_index_ai is not None:
                    anomaly_data_point_ai = df.loc[selected_anomaly_index_ai] # Renamed to avoid conflict
                    st.write(f"#### Details for Anomaly at Index: {selected_anomaly_index_ai}")
                    st.dataframe(anomaly_data_point_ai.to_frame().T)

                    st.markdown("##### Feature Comparison (Anomaly vs. Typical Data):")
                    explanation_found_ae_merged = False
                    for col_explain in df.columns:
                        if col_explain in anomaly_data_point_ai and pd.notna(anomaly_data_point_ai[col_explain]):
                            outlier_val_explain = anomaly_data_point_ai[col_explain]
                            non_anomaly_data_explain = df.drop(anomalies_for_ai_df.index, errors='ignore')

                            if pd.api.types.is_numeric_dtype(df[col_explain]) and not non_anomaly_data_explain[col_explain].dropna().empty:
                                mean_val_explain = non_anomaly_data_explain[col_explain].mean()
                                std_val_explain = non_anomaly_data_explain[col_explain].std()
                                q05_val_explain = non_anomaly_data_explain[col_explain].quantile(0.05)
                                q95_val_explain = non_anomaly_data_explain[col_explain].quantile(0.95)

                                if std_val_explain > 0:
                                    z_score_approx_explain = (outlier_val_explain - mean_val_explain) / std_val_explain
                                    if abs(z_score_approx_explain) > 2.5:
                                        st.write(f"- **{col_explain}**: Value `{outlier_val_explain:.2f}` is significantly different (approx. {z_score_approx_explain:.1f} std devs) from the typical mean (`{mean_val_explain:.2f}`).")
                                        explanation_found_ae_merged = True
                                    elif outlier_val_explain > q95_val_explain:
                                        st.write(f"- **{col_explain}**: Value `{outlier_val_explain:.2f}` is in the top 5% (above `{q95_val_explain:.2f}`). Typical mean: `{mean_val_explain:.2f}`.")
                                        explanation_found_ae_merged = True
                                    elif outlier_val_explain < q05_val_explain:
                                        st.write(f"- **{col_explain}**: Value `{outlier_val_explain:.2f}` is in the bottom 5% (below `{q05_val_explain:.2f}`). Typical mean: `{mean_val_explain:.2f}`.")
                                        explanation_found_ae_merged = True
                            elif df[col_explain].dtype == 'object' and not non_anomaly_data_explain[col_explain].dropna().empty:
                                mode_val_explain = non_anomaly_data_explain[col_explain].mode()
                                if not mode_val_explain.empty and outlier_val_explain != mode_val_explain[0]:
                                    value_freq_explain = non_anomaly_data_explain[col_explain].value_counts(normalize=True)
                                    if outlier_val_explain in value_freq_explain and value_freq_explain[outlier_val_explain] < 0.05:
                                        st.write(f"- **{col_explain}**: Category `'{outlier_val_explain}'` is uncommon (occurs <5% in non-anomalies). Most common is `'{mode_val_explain[0]}'`. ")
                                        explanation_found_ae_merged = True
                    if not explanation_found_ae_merged:
                        st.info("This anomaly does not show extreme deviations on individual features compared to the rest of the data based on simple statistical checks. It might be an outlier due to a combination of factors.")

                    st.markdown("---")
                    st.markdown("##### AI-Powered Narrative Explanation (Optional)")
                    if gemini_api_key: # Check again in case it was entered after page load
                        contextual_features_for_ai_merged = st.multiselect(
                            "Select Features to Provide as Context to AI for Narrative",
                            df.columns.tolist(),
                            default=[col for col in numeric_cols[:3] + categorical_cols[:2] if col in df.columns],
                            key="ai_anomaly_context_features_merged"
                        )
                        if st.button("🪄 Get AI Explanation for Anomaly", key="run_ai_anomaly_explanation_merged"):
                            with st.spinner("AI is investigating the anomaly..."):
                                anomaly_details_str_merged = "\n".join([f"- {col}: {anomaly_data_point_ai[col]}" for col in anomaly_data_point_ai.index if pd.notna(anomaly_data_point_ai[col])])
                                comparison_str_merged = ""
                                non_anomaly_data_ai_merged = df.drop(anomalies_for_ai_df.index, errors='ignore')
                                for col_ai_ctx in contextual_features_for_ai_merged:
                                    if col_ai_ctx in anomaly_data_point_ai and pd.notna(anomaly_data_point_ai[col_ai_ctx]):
                                        val_ai_ctx = anomaly_data_point_ai[col_ai_ctx]
                                        if pd.api.types.is_numeric_dtype(df[col_ai_ctx]) and not non_anomaly_data_ai_merged[col_ai_ctx].dropna().empty:
                                            mean_val_ai_ctx = non_anomaly_data_ai_merged[col_ai_ctx].mean()
                                            std_val_ai_ctx = non_anomaly_data_ai_merged[col_ai_ctx].std()
                                            comparison_str_merged += f"Feature '{col_ai_ctx}': Anomaly Value = {val_ai_ctx:.2f}, Typical Mean = {mean_val_ai_ctx:.2f}, Typical StdDev = {std_val_ai_ctx:.2f}\n"
                                        elif df[col_ai_ctx].dtype == 'object' and not non_anomaly_data_ai_merged[col_ai_ctx].dropna().empty:
                                            mode_val_ai_ctx = non_anomaly_data_ai_merged[col_ai_ctx].mode()[0] if not non_anomaly_data_ai_merged[col_ai_ctx].mode().empty else "N/A"
                                            comparison_str_merged += f"Feature '{col_ai_ctx}': Anomaly Value = '{val_ai_ctx}', Typical Mode = '{mode_val_ai_ctx}'\n"

                                prompt_ai_anomaly_merged = f"""
You are an expert data analyst specializing in anomaly investigation.
An anomaly has been detected in the dataset.

Anomalous Data Point (Index: {selected_anomaly_index_ai}):
{anomaly_details_str_merged}

Comparison with typical data for selected contextual features:
{comparison_str_merged if comparison_str_merged else "No specific feature comparisons provided beyond the anomaly's own values."}

Based on this information:
1. Provide a plausible narrative explanation for why this data point is considered anomalous. Consider the combination of feature deviations if applicable.
2. Suggest 2-3 potential root causes for this anomaly. These could be business-related (e.g., special promotion, data entry error, fraudulent activity, system glitch) or inherent data characteristics.
3. Recommend 2-3 concrete next steps for further investigation to confirm the nature and cause of this anomaly.
Be concise, insightful, and actionable. Structure your response clearly with headings for each of the three points.
"""
                                try:
                                    model_ai_anomaly_merged = genai.GenerativeModel("gemini-2.0-flash")
                                    response_ai_anomaly_merged = model_ai_anomaly_merged.generate_content(prompt_ai_anomaly_merged)
                                    st.markdown("#### AI Anomaly Investigation Report:")
                                    st.markdown(response_ai_anomaly_merged.text)
                                except Exception as e:
                                    st.error(f"Gemini API Error for Anomaly Investigation: {str(e)}")
                    else:
                        st.info("If you have a Gemini API key, enter it in the sidebar to enable AI-powered narrative explanations for anomalies.")

        # --- ADVANCED TOOL 5: Propensity Scoring Model ---
        with st.expander("🎯 ADVANCED TOOL 10: Propensity Scoring Model"):
            st.subheader("Predict Likelihood of a Binary Outcome")
            st.info("Train a Logistic Regression model to predict the probability (propensity score) of a binary outcome (e.g., purchase, churn, conversion).")
            if categorical_cols or numeric_cols: # Need features and a target
                prop_target_col = st.selectbox("Select Binary Target Column", df.columns, key="prop_target")
                prop_feature_options = [col for col in df.columns if col != prop_target_col]
                prop_features = st.multiselect("Select Feature Columns for Propensity Model", prop_feature_options, default=prop_feature_options[:min(3, len(prop_feature_options))], key="prop_features")

                if prop_target_col and prop_features:
                    if st.button("Train Propensity Score Model", key="run_prop_score"):
                        try:
                            prop_df_prep = df[[prop_target_col] + prop_features].copy().dropna()
                            
                            # Ensure target is binary 0/1
                            y_prop = prop_df_prep[prop_target_col]
                            if y_prop.nunique() == 2:
                                unique_prop_vals = sorted(y_prop.unique())
                                y_prop = y_prop.map({unique_prop_vals[0]: 0, unique_prop_vals[1]: 1})
                            elif not y_prop.isin([0,1]).all():
                                st.error(f"Target column '{prop_target_col}' must be binary (0/1) or have two distinct values.")
                                st.stop()

                            X_prop = prop_df_prep[prop_features]
                            X_prop_processed = pd.get_dummies(X_prop, drop_first=True) # One-hot encode categorical
                            
                            # Impute NaNs that might arise from get_dummies or were already there
                            imputer_prop = SimpleImputer(strategy='median')
                            X_prop_imputed = imputer_prop.fit_transform(X_prop_processed)
                            X_prop_imputed_df = pd.DataFrame(X_prop_imputed, columns=X_prop_processed.columns, index=X_prop_processed.index)

                            if len(X_prop_imputed_df) < 10:
                                st.warning("Not enough data for propensity scoring model after filtering and processing.")
                            else:
                                X_train_prop, X_test_prop, y_train_prop, y_test_prop = train_test_split(X_prop_imputed_df, y_prop.loc[X_prop_imputed_df.index], test_size=0.3, random_state=42, stratify=y_prop.loc[X_prop_imputed_df.index])
                                
                                prop_model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
                                prop_model.fit(X_train_prop, y_train_prop)
                                
                                propensity_scores_test = prop_model.predict_proba(X_test_prop)[:, 1] # Probability of class 1

                                st.write("#### Propensity Score Model Performance (Test Set)")
                                y_pred_prop_class = prop_model.predict(X_test_prop)
                                st.text(classification_report(y_test_prop, y_pred_prop_class, zero_division=0))

                                st.write("#### Distribution of Propensity Scores (Test Set)")
                                fig_prop_hist = px.histogram(x=propensity_scores_test, nbins=30, title="Propensity Score Distribution", labels={'x':'Propensity Score'})
                                st.plotly_chart(fig_prop_hist, use_container_width=True)

                                if st.checkbox("Add Propensity Scores to DataFrame?", key="add_prop_scores_df"):
                                    propensity_scores_all = prop_model.predict_proba(X_prop_imputed_df)[:, 1]
                                    df.loc[X_prop_imputed_df.index, f'{prop_target_col}_PropensityScore'] = propensity_scores_all
                                    st.success(f"Propensity scores for '{prop_target_col}' added. Rerun other analyses if needed.")
                                    st.rerun()
                        except Exception as e:
                            st.error(f"Error during Propensity Scoring: {e}")
                else:
                    st.info("Select a binary target column and feature columns.")
            else:
                st.info("Propensity Scoring requires columns for target and features.")

        # --- ADVANCED TOOL 6: Simplified Treatment Effect Estimation ---
        with st.expander("💊 ADVANCED TOOL 11: Simplified Treatment Effect Estimation"):
            st.subheader("Estimate Average Treatment Effect (ATE)")
            st.info("This tool provides a simplified estimation of the Average Treatment Effect (ATE) using regression adjustment. Select a binary treatment indicator, a numeric outcome, and optional covariates.")
            if categorical_cols or numeric_cols: # Need treatment, outcome
                ate_treatment_col = st.selectbox("Select Treatment Indicator Column (Binary: 0=Control, 1=Treated)", df.columns, key="ate_treatment")
                ate_outcome_col = st.selectbox("Select Outcome Column (Numeric)", numeric_cols, key="ate_outcome")
                ate_covariate_options = [col for col in df.columns if col not in [ate_treatment_col, ate_outcome_col]]
                ate_covariates = st.multiselect("Select Covariate Columns (Optional)", ate_covariate_options, key="ate_covariates")

                if ate_treatment_col and ate_outcome_col:
                    if st.button("Estimate Average Treatment Effect", key="run_ate"):
                        try:
                            ate_df_prep = df[[ate_treatment_col, ate_outcome_col] + ate_covariates].copy().dropna()
                            
                            # Ensure treatment is binary 0/1
                            y_ate_treatment = ate_df_prep[ate_treatment_col]
                            if y_ate_treatment.nunique() == 2:
                                unique_treat_vals = sorted(y_ate_treatment.unique())
                                y_ate_treatment = y_ate_treatment.map({unique_treat_vals[0]: 0, unique_treat_vals[1]: 1})
                            elif not y_ate_treatment.isin([0,1]).all():
                                st.error(f"Treatment column '{ate_treatment_col}' must be binary (0/1) or have two distinct values.")
                                st.stop()
                            ate_df_prep[ate_treatment_col] = y_ate_treatment # Update with 0/1

                            if len(ate_df_prep) < 20: # Need some data
                                st.warning("Not enough data for ATE estimation after filtering.")
                            else:
                                # Regression Adjustment
                                formula_ate = f"`{ate_outcome_col}` ~ `{ate_treatment_col}`"
                                if ate_covariates:
                                    formula_ate += " + " + " + ".join([f"`{cov}`" for cov in ate_covariates])
                                
                                # One-hot encode covariates if any are categorical
                                X_ate_features = [ate_treatment_col] + ate_covariates
                                X_ate_df = pd.get_dummies(ate_df_prep[X_ate_features], drop_first=True)
                                y_ate_df = ate_df_prep[ate_outcome_col]

                                # Impute NaNs in features (e.g., from dummification of sparse categories)
                                imputer_ate = SimpleImputer(strategy='median')
                                X_ate_imputed = imputer_ate.fit_transform(X_ate_df)
                                X_ate_imputed_df = pd.DataFrame(X_ate_imputed, columns=X_ate_df.columns, index=X_ate_df.index)

                                model_ate = LinearRegression()
                                model_ate.fit(X_ate_imputed_df, y_ate_df.loc[X_ate_imputed_df.index])
                                
                                # ATE is the coefficient of the treatment variable
                                treatment_coeff_name = ate_treatment_col # If not dummified
                                if ate_treatment_col not in X_ate_imputed_df.columns: # Check if it was dummified (e.g. if it was object type)
                                    # Find the dummified column name (assumes it's the first one if multiple categories existed)
                                    possible_dummy_cols = [col for col in X_ate_imputed_df.columns if col.startswith(ate_treatment_col + "_")]
                                    if possible_dummy_cols:
                                        treatment_coeff_name = possible_dummy_cols[0]
                                    else: # Should not happen if treatment is binary 0/1 and numeric
                                        st.error(f"Could not find treatment coefficient for '{ate_treatment_col}'.")
                                        st.stop()

                                ate_estimate = model_ate.coef_[X_ate_imputed_df.columns.get_loc(treatment_coeff_name)]

                                st.write("#### Simplified Average Treatment Effect (ATE) Estimation")
                                st.metric(f"Estimated ATE of '{ate_treatment_col}' on '{ate_outcome_col}'", f"{ate_estimate:.3f}")
                                st.caption(f"This means, on average, being in the 'treated' group (where {ate_treatment_col}=1) is associated with a {ate_estimate:.3f} unit change in '{ate_outcome_col}', holding other covariates constant.")

                                st.write("##### Regression Model Coefficients:")
                                coeffs_ate = pd.Series(model_ate.coef_, index=X_ate_imputed_df.columns).sort_values(ascending=False)
                                st.dataframe(coeffs_ate.rename("Coefficient"))
                        except Exception as e:
                            st.error(f"Error during ATE Estimation: {e}")
                else:
                    st.info("Select Treatment, Outcome, and optionally Covariate columns.")
            else:
                st.info("ATE Estimation requires columns for treatment, outcome, and optionally covariates.")

        # --- ADVANCED TOOL 7: Key Drivers Analysis ---
        with st.expander("🔑 ADVANCED TOOL 12: Key Drivers Analysis"):
            st.subheader("Identify Key Features Influencing a Target Variable")
            st.info("Train a model (Random Forest or Linear/Logistic Regression) and identify the most influential features (drivers) for a selected target variable.")
            if not df.empty:
                kda_target_col = st.selectbox("Select Target Variable for Key Drivers", df.columns, key="kda_target")
                kda_feature_options = [col for col in df.columns if col != kda_target_col]
                kda_features = st.multiselect("Select Feature Columns for Key Drivers Model", kda_feature_options, default=kda_feature_options[:min(5, len(kda_feature_options))], key="kda_features")
                
                kda_model_type = st.selectbox("Model Type for Driver Analysis", ["Random Forest", "Linear/Logistic Regression"], key="kda_model_type")

                if kda_target_col and kda_features:
                    if st.button("Analyze Key Drivers", key="run_kda"):
                        try:
                            kda_df_prep = df[[kda_target_col] + kda_features].copy().dropna()
                            y_kda = kda_df_prep[kda_target_col]
                            X_kda = kda_df_prep[kda_features]
                            X_kda_processed = pd.get_dummies(X_kda, drop_first=True)
                            
                            # Impute NaNs
                            imputer_kda = SimpleImputer(strategy='median')
                            X_kda_imputed = imputer_kda.fit_transform(X_kda_processed)
                            X_kda_imputed_df = pd.DataFrame(X_kda_imputed, columns=X_kda_processed.columns, index=X_kda_processed.index)

                            # Align y_kda with X_kda_imputed_df's index and handle potential NaNs in y_kda
                            y_aligned = y_kda.loc[X_kda_imputed_df.index]
                            if y_aligned.isnull().any():
                                st.warning(f"Target column '{kda_target_col}' has {y_aligned.isnull().sum()} missing values at indices corresponding to valid features. These rows will be dropped.")
                                valid_indices_after_y_dropna = y_aligned.dropna().index
                                y_aligned = y_aligned.loc[valid_indices_after_y_dropna]
                                X_kda_imputed_df = X_kda_imputed_df.loc[valid_indices_after_y_dropna]

                            if X_kda_imputed_df.empty or y_aligned.empty:
                                st.error("No data remains after aligning features and target. Cannot proceed with Key Drivers Analysis.")
                                st.stop()

                            # Determine task type and encode target
                            is_classification_kda = False
                            if pd.api.types.is_numeric_dtype(y_aligned.dtype):
                                if y_aligned.nunique() <= 10 and y_aligned.nunique() > 1: # Treat as classification if few unique numeric values
                                    is_classification_kda = True
                                    task_readable = "Classification (from Numeric Target)"
                                elif y_aligned.nunique() == 1:
                                    st.error(f"Target column '{kda_target_col}' has only one unique value. Cannot train model.")
                                    st.stop()
                                else: # Regression
                                    is_classification_kda = False
                                    task_readable = "Regression"
                            else: # Object, categorical, boolean
                                is_classification_kda = True
                                task_readable = "Classification (from Categorical/Boolean Target)"

                            st.info(f"Interpreting target '{kda_target_col}' for {task_readable}.")

                            if is_classification_kda:
                                le_kda = LabelEncoder()
                                y_kda_encoded = le_kda.fit_transform(y_aligned)
                                if len(le_kda.classes_) <= 1:
                                    st.error(f"Target column '{kda_target_col}' effectively has only one class after encoding. Cannot train model.")
                                    st.stop()
                            else:
                                y_kda_encoded = y_aligned # Already numeric for regression

                            if len(X_kda_imputed_df) < 10:
                                st.warning("Not enough data for Key Drivers Analysis after filtering.")
                            else:
                                if kda_model_type == "Random Forest":
                                    model_kda = RandomForestClassifier(random_state=42, n_estimators=100) if is_classification_kda else RandomForestRegressor(random_state=42, n_estimators=100)
                                    model_kda.fit(X_kda_imputed_df, y_kda_encoded)
                                    importances_kda = model_kda.feature_importances_
                                    drivers_df = pd.DataFrame({'Feature': X_kda_imputed_df.columns, 'Importance': importances_kda}).sort_values('Importance', ascending=False)
                                else: # Linear/Logistic Regression
                                    model_kda = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced') if is_classification_kda else LinearRegression()
                                    model_kda.fit(X_kda_imputed_df, y_kda_encoded)
                                    
                                    if is_classification_kda:
                                        if model_kda.coef_.ndim == 1: # Binary classification by some solvers or if only 1 class vs rest
                                            coefficients_kda = model_kda.coef_
                                        elif model_kda.coef_.shape[0] == 1: # Binary classification (standard case)
                                            coefficients_kda = model_kda.coef_[0]
                                        else: # Multiclass classification
                                            # Use mean of absolute coefficients across classes for 'Coefficient_Abs'
                                            # Use mean of coefficients for 'Coefficient' column (can be less interpretable for direction)
                                            abs_coeffs_for_drivers = np.mean(np.abs(model_kda.coef_), axis=0)
                                            mean_coeffs_for_drivers = np.mean(model_kda.coef_, axis=0)
                                            drivers_df = pd.DataFrame({'Feature': X_kda_imputed_df.columns, 
                                                                       'Coefficient_Abs': abs_coeffs_for_drivers, 
                                                                       'Coefficient': mean_coeffs_for_drivers}
                                                                     ).sort_values('Coefficient_Abs', ascending=False)
                                    else: # Regression
                                        coefficients_kda = model_kda.coef_
                                        drivers_df = pd.DataFrame({'Feature': X_kda_imputed_df.columns, 
                                                                   'Coefficient_Abs': np.abs(coefficients_kda), 
                                                                   'Coefficient': coefficients_kda}
                                                                 ).sort_values('Coefficient_Abs', ascending=False)
                                
                                # This block is for Random Forest or if drivers_df wasn't created for multiclass above
                                if kda_model_type == "Random Forest": # Or if drivers_df not yet defined for some reason
                                    metric_name_for_label = "Feature Importance"
                                    x_axis_col_name = 'Importance'
                                else: # Linear/Logistic
                                    metric_name_for_label = "Absolute Coefficient"
                                    x_axis_col_name = 'Coefficient_Abs'

                                st.write(f"#### Key Drivers for '{kda_target_col}' (using {kda_model_type})")
                                st.dataframe(drivers_df.head(15))

                                fig_kda = px.bar(drivers_df.head(15), x=x_axis_col_name, y='Feature', orientation='h', title=f"Top Key Drivers by {metric_name_for_label}", labels={x_axis_col_name: metric_name_for_label})
                                st.plotly_chart(fig_kda, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error during Key Drivers Analysis: {e}")
                else:
                    st.info("Select a target variable and feature columns.")
            else:
                st.info("Upload data to perform Key Drivers Analysis.")

        # --- ADVANCED TOOL 3: Robust Model Evaluation and Comparison Dashboard ---
        with st.expander("📊 ADVANCED TOOL 13: Model Evaluation & Comparison Dashboard"):
            st.subheader("Compare Performance of Trained Models")
            st.info("This tool allows comparison of models trained for the same task (classification or regression). For demonstration, it trains Logistic Regression and Random Forest for a selected binary target.")
            
            if not df.empty:
                eval_target_col = st.selectbox("Select Binary Target for Model Comparison", [col for col in df.columns if df[col].nunique()==2], key="eval_target") # Filter for binary cols
                eval_feature_options = [col for col in df.columns if col != eval_target_col]
                eval_features = st.multiselect("Select Features for Model Comparison", eval_feature_options, default=eval_feature_options[:min(5, len(eval_feature_options))], key="eval_features")

                st.sidebar.subheader("Hyperparameters for Comparison Models")
                rf_n_estimators_comp = st.sidebar.slider("Random Forest: N Estimators", 10, 200, 50, 10, key="eval_rf_n_est")
                rf_max_depth_comp = st.sidebar.slider("Random Forest: Max Depth", 2, 30, 10, 1, key="eval_rf_max_depth")

                if eval_target_col and eval_features:
                    if st.button("Train & Compare Models", key="run_model_comparison"):
                        with st.spinner("Training and comparing models..."):
                            try:
                                eval_df_prep = df[[eval_target_col] + eval_features].copy().dropna()
                                y_eval = eval_df_prep[eval_target_col]
                                X_eval = eval_df_prep[eval_features]

                                # Ensure target is 0/1
                                unique_eval_vals = sorted(y_eval.unique())
                                if len(unique_eval_vals) == 2:
                                    y_eval = y_eval.map({unique_eval_vals[0]: 0, unique_eval_vals[1]: 1})
                                else:
                                    st.error("Selected target for comparison is not binary.")
                                    st.stop()

                                X_eval_processed = pd.get_dummies(X_eval, drop_first=True)
                                imputer_eval = SimpleImputer(strategy='median')
                                X_eval_imputed = imputer_eval.fit_transform(X_eval_processed)
                                X_eval_imputed_df = pd.DataFrame(X_eval_imputed, columns=X_eval_processed.columns, index=X_eval_processed.index)

                                y_aligned_eval = y_eval.loc[X_eval_imputed_df.index].dropna()
                                X_final_eval = X_eval_imputed_df.loc[y_aligned_eval.index]

                                if X_final_eval.empty or y_aligned_eval.empty or y_aligned_eval.nunique() < 2:
                                    st.error("Not enough data or classes after preprocessing for model comparison.")
                                    st.stop()

                                X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(X_final_eval, y_aligned_eval, test_size=0.3, random_state=42, stratify=y_aligned_eval)

                                models_to_compare = {
                                    "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'),
                                    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=rf_n_estimators_comp, max_depth=rf_max_depth_comp, class_weight='balanced'),
                                    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
                                    "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=50) # Can add HPs for this too
                                }
                                
                                results_comparison = []
                                roc_curves_data = []

                                for name, model_instance in models_to_compare.items():
                                    model_instance.fit(X_train_eval, y_train_eval)
                                    y_pred_eval = model_instance.predict(X_test_eval)
                                    y_proba_eval = model_instance.predict_proba(X_test_eval)[:, 1]
                                    
                                    report = classification_report(y_test_eval, y_pred_eval, output_dict=True, zero_division=0)
                                    results_comparison.append({
                                        "Model": name,
                                        "Accuracy": report['accuracy'],
                                        "Precision (1)": report['1']['precision'] if '1' in report and isinstance(report['1'], dict) else report.get('weighted avg',{}).get('precision'),
                                        "Recall (1)": report['1']['recall'] if '1' in report and isinstance(report['1'], dict) else report.get('weighted avg',{}).get('recall'),
                                        "F1-score (1)": report['1']['f1-score'] if '1' in report and isinstance(report['1'], dict) else report.get('weighted avg',{}).get('f1-score'),
                                        "ROC AUC": roc_auc_score(y_test_eval, y_proba_eval)
                                    })
                                    fpr, tpr, _ = roc_curve(y_test_eval, y_proba_eval)
                                    roc_curves_data.append({'fpr': fpr, 'tpr': tpr, 'label': f'{name} (AUC = {roc_auc_score(y_test_eval, y_proba_eval):.2f})'})

                                st.write("#### Model Performance Metrics")
                                st.dataframe(pd.DataFrame(results_comparison))

                                st.write("#### ROC Curves")
                                fig_roc = go.Figure()
                                for curve_data in roc_curves_data:
                                    fig_roc.add_trace(go.Scatter(x=curve_data['fpr'], y=curve_data['tpr'], mode='lines', name=curve_data['label']))
                                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline (Random)', line=dict(dash='dash', color='grey')))
                                fig_roc.update_layout(title="ROC Curves Comparison", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                                st.plotly_chart(fig_roc, use_container_width=True)

                            except Exception as e:
                                st.error(f"Error during model comparison: {e}")
                else:
                    st.info("Select a binary target and feature columns for model comparison.")
            else:
                st.info("Upload data to use the model evaluation dashboard.")

        # --- ADVANCED TOOL 8: AI-Powered Segment Narrative Generator ---
        with st.expander("📝 ADVANCED TOOL 14: AI-Powered Segment Narrative Generator"):
            st.subheader("Generate Textual Summaries for Data Segments with AI")
            st.info("Select a segment column (e.g., 'Cluster' from K-Means, 'CLV_Segment') and features to describe. AI will generate a narrative for each segment.")
            if gemini_api_key:
                segment_col_options = [col for col in df.columns if df[col].nunique() < 20 and df[col].nunique() > 1] # Potential segment columns
                if segment_col_options:
                    ai_segment_col = st.selectbox("Select Segment Column", segment_col_options, key="ai_seg_col")
                    ai_seg_profile_features_num = st.multiselect("Select Numeric Features for Segment Description", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))], key="ai_seg_num_feats")
                    ai_seg_profile_features_cat = st.multiselect("Select Categorical Features for Segment Description", categorical_cols, default=categorical_cols[:min(2, len(categorical_cols))], key="ai_seg_cat_feats")

                    if ai_segment_col and (ai_seg_profile_features_num or ai_seg_profile_features_cat):
                        if st.button("Generate Segment Narratives with AI", key="run_ai_seg_narrative"):
                            with st.spinner("AI is crafting segment descriptions..."):
                                segments = df[ai_segment_col].unique()
                                for segment_val in segments:
                                    if pd.isna(segment_val): continue # Skip NaN segments
                                    
                                    segment_data = df[df[ai_segment_col] == segment_val]
                                    if segment_data.empty: continue

                                    narrative_prompt_seg = f"You are a data analyst. Describe the following data segment named '{segment_val}' from the column '{ai_segment_col}'. This segment has {len(segment_data)} records.\n"
                                    narrative_prompt_seg += "Key characteristics:\n"
                                    
                                    for feat_num in ai_seg_profile_features_num:
                                        if feat_num in segment_data.columns:
                                            narrative_prompt_seg += f"- Average '{feat_num}': {segment_data[feat_num].mean():.2f} (Median: {segment_data[feat_num].median():.2f})\n"
                                    for feat_cat in ai_seg_profile_features_cat:
                                        if feat_cat in segment_data.columns and not segment_data[feat_cat].mode().empty:
                                            narrative_prompt_seg += f"- Most common '{feat_cat}': {segment_data[feat_cat].mode()[0]}\n"
                                    
                                    narrative_prompt_seg += "\nProvide a concise (2-3 sentences) narrative summary of this segment, highlighting its distinguishing features based on the provided characteristics."

                                    try:
                                        model_seg_narrative = genai.GenerativeModel("gemini-2.0-flash")
                                        response_seg_narrative = model_seg_narrative.generate_content(narrative_prompt_seg)
                                        st.markdown(f"##### Narrative for Segment: {segment_val}")
                                        st.markdown(response_seg_narrative.text)
                                        st.markdown("---")
                                    except Exception as e:
                                        st.error(f"Gemini API Error for Segment '{segment_val}': {str(e)}")
                    else:
                        st.info("Select a segment column and features to describe.")
                else:
                    st.info("No suitable segment columns found (columns with 2-19 unique values). Run a clustering tool or create segments first.")
            else:
                st.info("Enter your Gemini API key in the sidebar to enable AI-powered segment narratives.")

        # --- ADVANCED TOOL 15: Feature Selection Utility ---
        with st.expander("🎯 ADVANCED TOOL 15: Feature Selection Utility"):
            st.subheader("Select Important Features for Modeling")
            st.info("Apply feature selection techniques to identify the most relevant features for a given target variable.")
            if not df.empty:
                fs_target_col = st.selectbox("Select Target Variable for Feature Selection", df.columns, key="fs_target")
                fs_feature_options = [col for col in df.columns if col != fs_target_col]
                
                fs_task_type = "Regression" # Default
                if df[fs_target_col].dtype == 'object' or df[fs_target_col].nunique() <= 20:
                    fs_task_type = "Classification"
                st.write(f"Inferred Task Type: **{fs_task_type}**")

                fs_method = st.selectbox("Feature Selection Method", 
                                         ["SelectKBest (ANOVA F-value for Regression, Chi2 for Classification)", 
                                          "Recursive Feature Elimination (RFE with RandomForest)"], 
                                         key="fs_method")
                
                num_features_to_select = 1 # Default if only one feature is available
                if len(fs_feature_options) > 1:
                    num_features_to_select = st.slider("Number of Top Features to Select", 1, len(fs_feature_options), min(5, len(fs_feature_options)), key="fs_k_features")
                elif fs_feature_options: # Exactly one feature
                    st.info("Only one feature is available for selection.")
                # If no features, the button logic below will handle it.

                if fs_target_col and fs_feature_options:
                    if st.button("Run Feature Selection", key="run_fs"):
                        with st.spinner("Selecting features..."):
                            try:
                                fs_df_prep = df[[fs_target_col] + fs_feature_options].copy().dropna()
                                y_fs = fs_df_prep[fs_target_col]
                                X_fs = fs_df_prep[fs_feature_options]

                                # Preprocessing: Impute NaNs and One-Hot Encode
                                numeric_transformer_fs = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
                                categorical_transformer_fs = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]) # sparse_output=False for RFE

                                from sklearn.compose import ColumnTransformer
                                preprocessor_fs = ColumnTransformer(transformers=[
                                    ('num', numeric_transformer_fs, X_fs.select_dtypes(include=np.number).columns.tolist()),
                                    ('cat', categorical_transformer_fs, X_fs.select_dtypes(include='object').columns.tolist())
                                ])
                                X_fs_processed = preprocessor_fs.fit_transform(X_fs)
                                # Get feature names after one-hot encoding
                                try:
                                    feature_names_processed_fs = preprocessor_fs.get_feature_names_out()
                                except AttributeError: # Older scikit-learn
                                    # Manual reconstruction of feature names
                                    feature_names_processed_fs = []
                                    for name, trans, cols in preprocessor_fs.transformers_:
                                        if trans == 'drop' or trans == 'passthrough': continue
                                        if hasattr(trans, 'get_feature_names_out'):
                                            feature_names_processed_fs.extend([f"{name}__{f}" for f in trans.get_feature_names_out(cols)])
                                        else: # Simple imputer/scaler
                                            feature_names_processed_fs.extend(cols)

                                X_fs_processed_df = pd.DataFrame(X_fs_processed, columns=feature_names_processed_fs, index=X_fs.index)
                                
                                # Align target
                                y_fs_aligned = y_fs.loc[X_fs_processed_df.index]
                                if fs_task_type == "Classification":
                                    le_fs = LabelEncoder()
                                    y_fs_aligned = le_fs.fit_transform(y_fs_aligned)

                                selected_features_list = []
                                if "SelectKBest" in fs_method:
                                    from sklearn.feature_selection import SelectKBest, f_regression, chi2
                                    score_func = f_regression if fs_task_type == "Regression" else chi2
                                    # For chi2, data must be non-negative. MinMax scale if necessary.
                                    if score_func == chi2:
                                        X_fs_processed_df_non_negative = MinMaxScaler().fit_transform(X_fs_processed_df)
                                        selector = SelectKBest(score_func=score_func, k=min(num_features_to_select, X_fs_processed_df.shape[1]))
                                        selector.fit(X_fs_processed_df_non_negative, y_fs_aligned)
                                    else:
                                        selector = SelectKBest(score_func=score_func, k=min(num_features_to_select, X_fs_processed_df.shape[1]))
                                        selector.fit(X_fs_processed_df, y_fs_aligned)
                                    selected_features_list = X_fs_processed_df.columns[selector.get_support()].tolist()
                                    scores_df = pd.DataFrame({'Feature': X_fs_processed_df.columns, 'Score': selector.scores_}).sort_values('Score', ascending=False)
                                    st.write("#### Feature Scores (SelectKBest):")
                                    st.dataframe(scores_df)

                                elif "RFE" in fs_method:
                                    from sklearn.feature_selection import RFE
                                    estimator = RandomForestRegressor(random_state=42, n_estimators=50) if fs_task_type == "Regression" else RandomForestClassifier(random_state=42, n_estimators=50)
                                    selector = RFE(estimator, n_features_to_select=num_features_to_select, step=1)
                                    selector.fit(X_fs_processed_df, y_fs_aligned)
                                    selected_features_list = X_fs_processed_df.columns[selector.support_].tolist()
                                    ranking_df = pd.DataFrame({'Feature': X_fs_processed_df.columns, 'Ranking': selector.ranking_}).sort_values('Ranking')
                                    st.write("#### Feature Ranking (RFE):")
                                    st.dataframe(ranking_df)

                                st.write(f"#### Top {num_features_to_select} Selected Features:")
                                st.write(selected_features_list)

                            except Exception as e:
                                st.error(f"Error during Feature Selection: {e}")
                else:
                    st.info("Select a target variable to proceed with feature selection.")
            else:
                st.info("Upload data to use the Feature Selection Utility.")

        # --- ADVANCED TOOL 15: Predictive Customer Churn Model ---
        with st.expander("💔 ADVANCED TOOL 16: Predictive Customer Churn Model"):
            st.subheader("Identify Customers at Risk of Churning")
            st.info("Train a classification model to predict customer churn. Requires a 'Churn' indicator column (binary) and relevant customer features.")
            if not df.empty:
                churn_target_col = st.selectbox("Select Churn Indicator Column (Binary: 1=Churned, 0=Active)", df.columns, key="churn_target_col")
                churn_feature_options = [col for col in df.columns if col != churn_target_col]
                churn_features = st.multiselect("Select Features for Churn Model", churn_feature_options, default=churn_feature_options[:min(5, len(churn_feature_options))], key="churn_model_features")

                if churn_target_col and churn_features:
                    if st.button("Train Churn Prediction Model", key="run_churn_model"):
                        try:
                            churn_df_prep = df[[churn_target_col] + churn_features].copy().dropna()
                            y_churn = churn_df_prep[churn_target_col]

                            if y_churn.nunique() == 2:
                                unique_churn_vals = sorted(y_churn.unique())
                                y_churn = y_churn.map({unique_churn_vals[0]: 0, unique_churn_vals[1]: 1})
                            elif not y_churn.isin([0,1]).all():
                                st.error(f"Churn target column '{churn_target_col}' must be binary (0/1) or have two distinct values.")
                                st.stop()

                            X_churn = churn_df_prep[churn_features]
                            X_churn_processed = pd.get_dummies(X_churn, drop_first=True)
                            
                            imputer_churn = SimpleImputer(strategy='median')
                            X_churn_imputed = imputer_churn.fit_transform(X_churn_processed)
                            X_churn_imputed_df = pd.DataFrame(X_churn_imputed, columns=X_churn_processed.columns, index=X_churn_processed.index)

                            y_aligned_churn = y_churn.loc[X_churn_imputed_df.index].dropna()
                            X_final_churn = X_churn_imputed_df.loc[y_aligned_churn.index]

                            if X_final_churn.empty or y_aligned_churn.empty or len(y_aligned_churn.unique()) < 2:
                                st.error("Not enough data or classes after preprocessing for churn model training.")
                                st.stop()

                            X_train_churn, X_test_churn, y_train_churn, y_test_churn = train_test_split(X_final_churn, y_aligned_churn, test_size=0.3, random_state=42, stratify=y_aligned_churn)

                            churn_model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
                            churn_model.fit(X_train_churn, y_train_churn)
                            y_pred_churn = churn_model.predict(X_test_churn)
                            y_proba_churn = churn_model.predict_proba(X_test_churn)[:, 1]

                            st.write("#### Churn Model Performance (Test Set)")
                            st.text(classification_report(y_test_churn, y_pred_churn, zero_division=0))
                            st.metric("ROC AUC Score", f"{roc_auc_score(y_test_churn, y_proba_churn):.3f}")

                            st.write("#### Feature Importances for Churn Prediction")
                            importances_churn = pd.Series(churn_model.feature_importances_, index=X_final_churn.columns).sort_values(ascending=False)
                            st.bar_chart(importances_churn.head(10))

                        except Exception as e:
                            st.error(f"Error during Churn Model training: {e}")
                else:
                    st.info("Select a binary churn target and feature columns.")
            else:
                st.info("Upload data to train a churn prediction model.")

        # --- ADVANCED TOOL 16: Dynamic Pricing Simulation (Conceptual) ---
        with st.expander("⚖️ ADVANCED TOOL 17: Dynamic Pricing Simulation"):
            st.subheader("Simulate Revenue Impact of Price Changes")
            st.info("Simulate how changes in price might affect demand and revenue for a selected product/category. Requires current price, quantity, and an estimated price elasticity.")
            if categorical_cols and numeric_cols:
                price_sim_item_col = st.selectbox("Select Product/Category Column for Pricing", categorical_cols, key="price_sim_item")
                price_sim_current_price_col = st.selectbox("Select Current Price Column", numeric_cols, key="price_sim_price")
                price_sim_current_qty_col = st.selectbox("Select Current Quantity Sold Column", numeric_cols, key="price_sim_qty")

                if price_sim_item_col and price_sim_current_price_col and price_sim_current_qty_col:
                    selected_item_price_sim = st.selectbox("Select a Specific Product/Category to Simulate", df[price_sim_item_col].unique(), key="price_sim_select_item")
                    
                    item_data_sim = df[df[price_sim_item_col] == selected_item_price_sim]
                    if not item_data_sim.empty:
                        avg_current_price = item_data_sim[price_sim_current_price_col].mean()
                        avg_current_qty = item_data_sim[price_sim_current_qty_col].mean()

                        st.write(f"Current Average Price for '{selected_item_price_sim}': {avg_current_price:.2f}")
                        st.write(f"Current Average Quantity for '{selected_item_price_sim}': {avg_current_qty:.2f}")

                        price_elasticity_sim = st.slider("Estimated Price Elasticity of Demand", -5.0, -0.1, -1.5, 0.1, key="price_sim_elasticity", help="Typically negative. E.g., -1.5 means a 10% price increase leads to a 15% quantity decrease.")
                        price_change_pct_sim = st.slider("Simulated Price Change (%)", -50, 50, 10, key="price_sim_change_pct")

                        if st.button("Simulate Pricing Impact", key="run_price_sim"):
                            new_price_sim = avg_current_price * (1 + price_change_pct_sim / 100)
                            qty_change_pct_sim = price_elasticity_sim * price_change_pct_sim 
                            new_qty_sim = avg_current_qty * (1 + qty_change_pct_sim / 100)
                            new_qty_sim = max(0, new_qty_sim) # Quantity cannot be negative

                            current_revenue_sim = avg_current_price * avg_current_qty
                            new_revenue_sim = new_price_sim * new_qty_sim

                            st.write("#### Simulation Results:")
                            col_sim1, col_sim2, col_sim3 = st.columns(3)
                            with col_sim1:
                                st.metric("New Price", f"{new_price_sim:.2f}", delta=f"{(new_price_sim - avg_current_price):.2f} ({price_change_pct_sim}%)")
                            with col_sim2:
                                st.metric("New Quantity", f"{new_qty_sim:.2f}", delta=f"{(new_qty_sim - avg_current_qty):.2f} ({qty_change_pct_sim:.1f}%)")
                            with col_sim3:
                                st.metric("New Revenue", f"{new_revenue_sim:.2f}", delta=f"{(new_revenue_sim - current_revenue_sim):.2f}")
                            st.caption("Note: This is a simplified simulation based on estimated elasticity and average values.")
                    else:
                        st.warning(f"No data found for selected item '{selected_item_price_sim}'.")
                else:
                    st.info("Select product/category, current price, and current quantity columns.")
            else:
                st.info("Dynamic Pricing Simulation requires categorical and numeric columns.")

        # --- ADVANCED TOOL 17: Sales Funnel Conversion Analysis (Conceptual) ---
        with st.expander("💧 ADVANCED TOOL 18: Sales Funnel Conversion Analysis"):
            st.subheader("Analyze Conversion Rates Through a Sales Funnel")
            st.info("Define stages of your sales funnel and map them to numeric columns representing counts/values at each stage to visualize conversion rates.")
            if not df.empty:
                funnel_stages_str = st.text_input("Enter Funnel Stages (comma-separated, e.g., Website Visits, Product Views, Add to Cart, Purchase)", 
                                                  "Website Visits,Product Views,Add to Cart,Purchase", key="funnel_stages_input")
                funnel_stages = [stage.strip() for stage in funnel_stages_str.split(',') if stage.strip()]

                stage_columns = {}
                if funnel_stages:
                    st.write("Map Funnel Stages to Numeric Columns (Counts/Values at each stage):")
                    for stage in funnel_stages:
                        stage_columns[stage] = st.selectbox(f"Column for '{stage}'", [None] + numeric_cols, key=f"funnel_col_{stage.replace(' ','_')}")
                
                if st.button("Analyze Funnel Conversion", key="run_funnel_analysis") and all(stage_columns.values()):
                    try:
                        funnel_data = []
                        for stage in funnel_stages:
                            col_name = stage_columns[stage]
                            if col_name in df.columns and pd.api.types.is_numeric_dtype(df[col_name]):
                                funnel_data.append(df[col_name].sum()) # Summing up values for each stage
                            else:
                                st.error(f"Column '{col_name}' for stage '{stage}' is not valid or not numeric.")
                                st.stop()
                        
                        if len(funnel_data) == len(funnel_stages):
                            fig_funnel = go.Figure(go.Funnel(
                                y = funnel_stages,
                                x = funnel_data,
                                textposition = "inside",
                                textinfo = "value+percent previous",
                                marker = {"color": [custom_color if i==0 else px.colors.sequential.Blues[len(px.colors.sequential.Blues)-1-i%len(px.colors.sequential.Blues)] for i in range(len(funnel_stages))]}
                            ))
                            fig_funnel.update_layout(title="Sales Funnel Conversion Rates")
                            st.plotly_chart(fig_funnel, use_container_width=True)
                        else:
                            st.warning("Could not gather data for all funnel stages.")
                    except Exception as e:
                        st.error(f"Error during Funnel Analysis: {e}")
                elif funnel_stages and not all(stage_columns.values()):
                    st.info("Please map all defined funnel stages to numeric columns.")
            else:
                st.info("Upload data to perform funnel analysis.")

        # --- ADVANCED TOOL 18: Inventory Optimization Suggestions (Conceptual AI) ---
        with st.expander("📦 ADVANCED TOOL 19: AI-Powered Inventory Optimization Suggestions"):
            st.subheader("Get AI-Powered Suggestions for Inventory Management")
            st.info("Provide product ID, sales quantity, and optionally current inventory and lead time. AI will offer actionable suggestions.")
            if gemini_api_key:
                if categorical_cols and numeric_cols:
                    inv_prod_id_col = st.selectbox("Select Product ID Column", categorical_cols + numeric_cols, key="inv_prod_id")
                    inv_sales_qty_col = st.selectbox("Select Sales Quantity Column (e.g., last 30 days)", numeric_cols, key="inv_sales_qty")
                    inv_current_stock_col = st.selectbox("Select Current Inventory Column (Optional)", [None] + numeric_cols, key="inv_current_stock")
                    inv_lead_time_col = st.selectbox("Select Supplier Lead Time (Days) Column (Optional)", [None] + numeric_cols, key="inv_lead_time")

                    if inv_prod_id_col and inv_sales_qty_col:
                        selected_prod_inv_opt = st.selectbox("Select a Specific Product for Inventory Advice", df[inv_prod_id_col].unique()[:100], key="inv_opt_select_prod") # Limit for dropdown
                        
                        if st.button("🪄 Get Inventory Optimization Advice", key="run_inv_opt_ai"):
                            prod_data_inv = df[df[inv_prod_id_col] == selected_prod_inv_opt].iloc[0] if not df[df[inv_prod_id_col] == selected_prod_inv_opt].empty else None
                            if prod_data_inv is not None:
                                sales_qty = prod_data_inv[inv_sales_qty_col]
                                current_stock = prod_data_inv.get(inv_current_stock_col, "Not Provided") if inv_current_stock_col else "Not Provided"
                                lead_time = prod_data_inv.get(inv_lead_time_col, "Not Provided") if inv_lead_time_col else "Not Provided"

                                inv_prompt = f"""
                                You are an inventory management expert. For a product '{selected_prod_inv_opt}':
                                - Recent Sales Quantity (e.g., last 30 days): {sales_qty}
                                - Current Stock Level: {current_stock}
                                - Supplier Lead Time (days): {lead_time}

                                Based on this information, provide:
                                1. A brief assessment of the current inventory situation (e.g., risk of stockout, overstock).
                                2. Two actionable suggestions for optimizing inventory for this product (e.g., reorder point, safety stock considerations, demand forecasting improvements).
                                Keep the advice practical and focused.
                                """
                                with st.spinner("AI is analyzing inventory data..."):
                                    try:
                                        model_inv_opt = genai.GenerativeModel("gemini-2.0-flash")
                                        response_inv_opt = model_inv_opt.generate_content(inv_prompt)
                                        st.markdown("#### AI Inventory Optimization Advice:")
                                        st.markdown(response_inv_opt.text)
                                    except Exception as e:
                                        st.error(f"Gemini API Error for Inventory Advice: {str(e)}")
                            else:
                                st.warning(f"No data found for product '{selected_prod_inv_opt}'.")
                    else:
                        st.info("Select Product ID and Sales Quantity columns.")
                else:
                    st.info("Inventory Optimization requires product ID and sales quantity columns.")
            else:
                st.info("Enter your Gemini API key in the sidebar for AI-powered inventory suggestions.")

        # --- ADVANCED TOOL 19: AI-Powered Predictive Maintenance Advisor (Conceptual) ---
        with st.expander("🛠️ ADVANCED TOOL 20: AI-Powered Predictive Maintenance Advisor (Conceptual)"):
            st.subheader("Get AI Advice on Predictive Maintenance")
            st.info("Provide equipment/product ID and relevant operational data (e.g., usage hours, error counts, age). AI will offer conceptual maintenance advice.")
            if gemini_api_key:
                if categorical_cols or numeric_cols: # Need some columns for ID and features
                    pm_item_id_col = st.selectbox("Select Equipment/Product ID Column", categorical_cols + numeric_cols, key="pm_item_id")
                    
                    pm_feature_options = [col for col in numeric_cols + date_cols + categorical_cols if col != pm_item_id_col]
                    pm_features_for_ai = st.multiselect(
                        "Select Relevant Operational Features for AI Context",
                        pm_feature_options,
                        default=pm_feature_options[:min(4, len(pm_feature_options))],
                        key="pm_ai_features",
                        help="E.g., UsageHours, ErrorCount, AgeInMonths, LastServiceDate, Temperature"
                    )

                    if pm_item_id_col and pm_features_for_ai:
                        selected_item_pm_advice = st.selectbox("Select a Specific Equipment/Product for Maintenance Advice", df[pm_item_id_col].unique()[:100], key="pm_select_item_advice") # Limit for dropdown
                        
                        if st.button("🪄 Get Predictive Maintenance Advice", key="run_pm_advice_ai"):
                            item_data_pm = df[df[pm_item_id_col] == selected_item_pm_advice].iloc[0] if not df[df[pm_item_id_col] == selected_item_pm_advice].empty else None
                            if item_data_pm is not None:
                                item_details_str_pm = "\n".join([f"- {feat}: {item_data_pm.get(feat, 'N/A')}" for feat in pm_features_for_ai])
                                
                                pm_prompt = f"""
                                You are a predictive maintenance expert. For an equipment/product '{selected_item_pm_advice}', the following operational data is provided:
                                {item_details_str_pm}

                                Based on this information, provide:
                                1. A brief conceptual assessment of its current operational state or potential risks (e.g., high usage, nearing end-of-life based on age, frequent errors).
                                2. Two high-level, conceptual suggestions for predictive maintenance actions or monitoring strategies for this item.
                                Keep the advice general and conceptual, focusing on types of actions rather than specific technical details.
                                """
                                try:
                                    model_pm_advice = genai.GenerativeModel("gemini-2.0-flash")
                                    response_pm_advice = model_pm_advice.generate_content(pm_prompt)
                                    st.markdown("#### AI Predictive Maintenance Advice:")
                                    st.markdown(response_pm_advice.text)
                                except Exception as e:
                                    st.error(f"Gemini API Error for Predictive Maintenance Advice: {str(e)}")
                            else:
                                st.warning(f"No data found for equipment/product '{selected_item_pm_advice}'.")
                    else:
                        st.info("Select an ID column and relevant operational features.")
                else:
                    st.info("Predictive Maintenance Advisor requires ID and feature columns.")
            else:
                st.info("Enter your Gemini API key in the sidebar for AI-powered predictive maintenance advice.")

        # --- ADVANCED TOOL 21: Scenario Planning & Impact Analysis (AI-Enhanced) ---
        with st.expander("🚀 ADVANCED TOOL 21: Scenario Planning & Impact Analysis (AI-Enhanced)"):
            st.subheader("Explore Potential Impacts of Defined Scenarios with AI")
            st.info("Describe a scenario (e.g., 'What if demand for Product X doubles due to a marketing campaign?') and let AI provide a qualitative impact assessment based on the dataset's context.")
            if gemini_api_key:
                scenario_description = st.text_area("Describe your Scenario:", height=100, 
                                                    placeholder="E.g., What if there's a 20% increase in 'Sales' for 'Region'='North' due to a new local partnership?",
                                                    key="scenario_desc_input")
                
                scenario_context_features = st.multiselect(
                    "Select Key Dataset Columns Relevant to this Scenario (for AI context)",
                    df.columns.tolist(),
                    default=df.columns.tolist()[:min(5, len(df.columns))],
                    key="scenario_context_features"
                )

                if scenario_description and scenario_context_features:
                    if st.button("🔮 Analyze Scenario Impact with AI", key="run_scenario_ai"):
                        with st.spinner("AI is simulating the scenario..."):
                            # Provide a sample of the data for context
                            data_sample_scenario = df[scenario_context_features].head(5).to_string()
                            
                            prompt_scenario_ai = f"""
                            You are a strategic business analyst. Consider the following dataset characteristics:
                            The dataset has columns: {', '.join(df.columns.tolist())}.
                            Selected relevant columns for this scenario are: {', '.join(scenario_context_features)}.
                            A small sample of data from these relevant columns:
                            {data_sample_scenario}

                            Now, analyze the following scenario: "{scenario_description}"

                            Based on this scenario and the general nature of the provided dataset columns, provide a qualitative impact assessment. Discuss:
                            1. Potential primary impacts on key metrics (mention which metrics from the dataset might be affected).
                            2. Possible secondary or ripple effects on other aspects of the business/data.
                            3. Key assumptions you are making in your analysis.
                            4. 2-3 critical factors or uncertainties that could influence the actual outcome.
                            Keep the analysis concise and strategic.
                            """
                            try:
                                model_scenario_ai = genai.GenerativeModel("gemini-2.0-flash")
                                response_scenario_ai = model_scenario_ai.generate_content(prompt_scenario_ai)
                                st.markdown("#### AI Scenario Impact Assessment:")
                                st.markdown(response_scenario_ai.text)
                            except Exception as e:
                                st.error(f"Gemini API Error for Scenario Analysis: {str(e)}")
                else:
                    st.info("Describe your scenario and select relevant context features.")
            else:
                st.info("Enter your Gemini API key in the sidebar for AI-powered scenario planning.")

        # --- Miscellaneous Tool: Data Dictionary Generator ---
        with st.expander("📚 Miscellaneous Tool: Data Dictionary Generator"):
            st.subheader("Generate a Data Dictionary for Your Dataset")
            if not df.empty:
                data_dict = pd.DataFrame({
                    "Column": df.columns,
                    "Data Type": [str(df[col].dtype) for col in df.columns],
                    "Unique Values": [df[col].nunique() for col in df.columns],
                    "Missing %": [df[col].isnull().mean()*100 for col in df.columns]
                })
                st.dataframe(data_dict)
                st.download_button("Download Data Dictionary", data_dict.to_csv(index=False), file_name="data_dictionary.csv")
            else:
                st.info("Upload data to generate a data dictionary.")

        # --- Miscellaneous Tool: Random Row Sampler ---
        with st.expander("🎲 Miscellaneous Tool: Random Row Sampler"):
            st.subheader("Sample Random Rows from Your Data")
            if not df.empty:
                n_sample = st.number_input("Number of Rows to Sample", 1, min(10, len(df)), 5)
                if st.button("Sample Rows"):
                    st.dataframe(df.sample(n=n_sample))
            else:
                st.info("Upload data to sample rows.")

        # --- Miscellaneous Tool: Column Renamer ---
        with st.expander("✏️ Miscellaneous Tool: Column Renamer"):
            st.subheader("Rename Columns Easily")
            if not df.empty:
                col_to_rename = st.selectbox("Select Column to Rename", df.columns)
                new_col_name = st.text_input("New Column Name", "")
                if st.button("Rename Column"):
                    if new_col_name and new_col_name not in df.columns:
                        df.rename(columns={col_to_rename: new_col_name}, inplace=True)
                        st.success(f"Renamed '{col_to_rename}' to '{new_col_name}'.")
                        st.rerun()
                    else:
                        st.warning("Provide a unique new column name.")
            else:
                st.info("Upload data to rename columns.")

        # --- Miscellaneous Tool: Value Replacer ---
        with st.expander("🔄 Miscellaneous Tool: Value Replacer"):
            st.subheader("Replace Values in a Column")
            if not df.empty:
                col_replace = st.selectbox("Select Column", df.columns)
                old_value = st.text_input("Value to Replace")
                new_value = st.text_input("Replace With")
                if st.button("Replace Value"):
                    df[col_replace] = df[col_replace].replace(old_value, new_value)
                    st.success(f"Replaced '{old_value}' with '{new_value}' in '{col_replace}'.")
                    st.rerun()
            else:
                st.info("Upload data to replace values.")

        # --- Miscellaneous Tool: Duplicate Column Finder ---
        with st.expander("🧬 Miscellaneous Tool: Duplicate Column Finder"):
            st.subheader("Find Columns with Identical Data")
            if not df.empty:
                duplicates = []
                for i, col1 in enumerate(df.columns):
                    for col2 in df.columns[i+1:]:
                        if df[col1].equals(df[col2]):
                            duplicates.append((col1, col2))
                if duplicates:
                    st.write("Duplicate columns found:")
                    for col1, col2 in duplicates:
                        st.write(f"{col1} and {col2}")
                else:
                    st.info("No duplicate columns found.")
            else:
                st.info("Upload data to check for duplicate columns.")

        # --- Miscellaneous Tool: Column Value Counter ---
        with st.expander("🔢 Miscellaneous Tool: Column Value Counter"):
            st.subheader("Count Occurrences of a Value in a Column")
            if not df.empty:
                col_count = st.selectbox("Select Column", df.columns, key="col_value_counter_col")
                value_count = st.text_input("Value to Count", key="col_value_counter_val")
                if st.button("Count Value", key="col_value_counter_btn"):
                    count = (df[col_count] == value_count).sum()
                    st.write(f"Value '{value_count}' appears {count} times in '{col_count}'.")
            else:
                st.info("Upload data to count values.")

        # --- Miscellaneous Tool: Dataframe Shape Viewer ---
        with st.expander("📏 Miscellaneous Tool: Dataframe Shape Viewer"):
            st.subheader("Quickly View DataFrame Shape")
            if not df.empty:
                st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            else:
                st.info("Upload data to view shape.")

        # --- Miscellaneous Tool: Column Type Summary ---
        with st.expander("🧮 Miscellaneous Tool: Column Type Summary"):
            st.subheader("Summary of Column Data Types")
            if not df.empty:
                type_summary = df.dtypes.value_counts().reset_index()
                type_summary.columns = ["Data Type", "Count"]
                st.dataframe(type_summary)
            else:
                st.info("Upload data to view column type summary.")

        # --- Miscellaneous Tool: Null Value Heatmap ---
        with st.expander("🌡️ Miscellaneous Tool: Null Value Heatmap"):
            st.subheader("Visualize Missing Data as a Heatmap")
            if not df.empty:
                import seaborn as sns
                fig, ax = plt.subplots(figsize=(min(12, len(df.columns)*0.7), 6))
                sns.heatmap(df.isnull(), cbar=False, ax=ax)
                st.pyplot(fig)
            else:
                st.info("Upload data to visualize missing values.")

        # --- Miscellaneous Tool: Custom Theme Designer ---
        with st.expander("🖌️ Miscellaneous Tool: Custom Theme Designer"):
            st.subheader("Create Your Custom Theme")
            
            col1, col2 = st.columns(2)
            with col1:
                primary_color = st.color_picker("Primary Color", "#38B2AC")      # Teal
                secondary_color = st.color_picker("Secondary Color", "#805AD5")  # Purple
                text_color = st.color_picker("Text Color", "#E2E8F0")            # Light Gray
            with col2:
                bg_color_theme = st.color_picker("Background Color", "#1A202C", key="theme_bg_color") # Renamed to avoid conflict
                accent_color = st.color_picker("Accent Color", "#ED8936")        # Orange
                
            theme_name = st.text_input("Theme Name", "My Custom Theme")

            if st.button("Apply Custom Theme"):
                custom_css = f"""
                <style>
                .stApp {{
                    background-color: {bg_color_theme};
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
                    color: {bg_color_theme}; /* Text color for button, ensure contrast */
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
                    "background": bg_color_theme,
                    "accent": accent_color
                }
                st.download_button("Download Theme", 
                                 json.dumps(theme_config, indent=2),
                                 f"{theme_name.lower().replace(' ', '_')}_theme.json",
                                 key="theme_download_button")


    # Sample data generator for testing (moved outside expanders)
    if st.button("🎲 Generate Sample Data & Explore Features", key="generate_sample_data_button"):
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

# Auto-refresh functionality (moved to the end of the main script execution)
if uploaded_files and datasets and refresh_interval > 0: # Only refresh if data is loaded
    time.sleep(refresh_interval)
    st.rerun()

# Footer with session info (moved to the end)
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1: st.info(f"🕒 Session: {datetime.now().strftime('%H:%M:%S')}")
with footer_col2: st.info(f"🎨 Theme: {selected_theme}")
with footer_col3: st.info(f"📚 Datasets: {len(datasets) if uploaded_files and datasets else 0}")
