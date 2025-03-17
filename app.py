import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import altair_saver
import json

st.set_page_config(layout="wide")
st.title("Advanced Automatic Dashboard Creator")
st.write("Upload a CSV or Excel file to automatically generate an advanced dashboard with multiple visualization options.")

# Sidebar: Global options
st.sidebar.header("Dashboard Options")
color_scheme = st.sidebar.selectbox("Select Color Scheme", options=['viridis', 'plasma', 'inferno', 'magma', 'cividis'], index=0)
sns.set_palette(color_scheme)  # Use for seaborn plots

# File uploader for CSV and Excel files
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

# Function to create a download button for Matplotlib figures
def download_button_fig(fig, filename):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    st.download_button(label="Download Chart as PNG", data=buf, file_name=filename, mime="image/png")

# Function to create a download button for Altair charts
def download_button_altair(chart, filename):
    buf = BytesIO()
    try:
        altair_saver.save(chart, buf, fmt='png')
        buf.seek(0)
        st.download_button(label="Download Chart as PNG", data=buf, file_name=filename, mime="image/png")
    except Exception as e:
        if "No enabled saver found" in str(e):
            st.warning(
                "PNG export for Altair charts is not enabled. "
                "To enable this feature, please install additional dependencies, e.g., run "
                "`pip install altair_saver[selenium]` or set up Node.js with vega-lite."
            )
        else:
            st.error("Error saving chart: " + str(e))

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
        st.error("Error reading file:")
        st.error(e)
        df = None

    if df is not None:
        st.subheader("Data Preview")
        st.write(df.head())

        st.subheader("Data Summary")
        st.write(df.describe(include='all'))

        # Identify column types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Detect date columns (attempt conversion)
        date_cols = []
        for col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
                date_cols.append(col)
            except Exception:
                continue

        st.markdown("---")
        st.header("Recommended Visualizations")

        # 1. Histograms for Numeric Columns (Altair)
        if numeric_cols:
            st.subheader("Histograms for Numeric Columns")
            for col in numeric_cols:
                st.write(f"**Histogram for {col}:**")
                hist_chart = alt.Chart(df).mark_bar().encode(
                    alt.X(f"{col}:Q", bin=True),
                    y='count()',
                    tooltip=[col, 'count()']
                ).properties(width=600, height=300, title=f"Histogram for {col}") \
                  .configure_mark(color=color_scheme)
                st.altair_chart(hist_chart, use_container_width=True)
                download_button_altair(hist_chart, f"{col}_histogram.png")
        else:
            st.write("No numeric columns found for histograms.")

        # 2. Bar Charts for Categorical Columns (Altair)
        if categorical_cols:
            st.subheader("Bar Charts for Categorical Columns")
            for col in categorical_cols:
                st.write(f"**Bar Chart for {col}:**")
                bar_chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X(f"{col}:N"),
                    y='count()',
                    tooltip=[col, 'count()']
                ).properties(width=600, height=300, title=f"Bar Chart for {col}")
                st.altair_chart(bar_chart, use_container_width=True)
                download_button_altair(bar_chart, f"{col}_barchart.png")
        else:
            st.write("No categorical columns found for bar charts.")

        # 3. Correlation Heatmap (Altair)
        if len(numeric_cols) >= 2:
            st.subheader("Correlation Heatmap")
            corr = df[numeric_cols].corr()
            corr_long = corr.reset_index().melt('index')
            heatmap = alt.Chart(corr_long).mark_rect().encode(
                x=alt.X('variable:O', title='Variable'),
                y=alt.Y('index:O', title='Variable'),
                color=alt.Color('value:Q', scale=alt.Scale(scheme=color_scheme))
            ).properties(width=600, height=400, title="Correlation Heatmap")
            st.altair_chart(heatmap, use_container_width=True)
            download_button_altair(heatmap, "correlation_heatmap.png")
        else:
            st.write("Not enough numeric columns for a correlation heatmap.")

        # 4. Scatter Plot Matrix (Pairplot using Seaborn)
        if len(numeric_cols) > 1:
            st.subheader("Scatter Plot Matrix (Pairplot)")
            try:
                pairplot = sns.pairplot(df[numeric_cols])
                st.pyplot(pairplot.fig)
                download_button_fig(pairplot.fig, "scatter_plot_matrix.png")
            except Exception as e:
                st.error("Error generating scatter plot matrix: " + str(e))
        else:
            st.write("Not enough numeric columns for a scatter plot matrix.")

        # 5. Box Plots for Numeric Columns (Seaborn)
        if numeric_cols:
            st.subheader("Box Plots for Numeric Columns")
            for col in numeric_cols:
                st.write(f"**Box Plot for {col}:**")
                fig, ax = plt.subplots()
                sns.boxplot(y=df[col], ax=ax)
                ax.set_title(f"Box Plot for {col}")
                st.pyplot(fig)
                download_button_fig(fig, f"{col}_boxplot.png")
        else:
            st.write("No numeric columns available for box plots.")

        # 6. Time Series Analysis (Altair)
        if date_cols and numeric_cols:
            st.subheader("Time Series Analysis")
            selected_date = st.selectbox("Select Date Column", options=date_cols)
            selected_numeric = st.selectbox("Select Numeric Column for Time Series", options=numeric_cols)
            df_sorted = df.sort_values(by=selected_date)
            time_chart = alt.Chart(df_sorted).mark_line().encode(
                x=alt.X(f"{selected_date}:T", title=selected_date),
                y=alt.Y(f"{selected_numeric}:Q", title=selected_numeric),
                tooltip=[selected_date, selected_numeric]
            ).properties(width=600, height=300, title=f"Time Series of {selected_numeric} over {selected_date}")
            st.altair_chart(time_chart, use_container_width=True)
            download_button_altair(time_chart, f"time_series_{selected_numeric}.png")
        else:
            st.write("Time series analysis not applicable (need at least one date and one numeric column).")

        # 7. Outlier Detection Summary using IQR
        st.subheader("Outlier Detection Summary")
        outlier_summary = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            outlier_summary[col] = int(len(outliers))
        st.write("Count of detected outliers for each numeric column:")
        st.write(outlier_summary)

        # 8. Data Filtering for Categorical Columns
        if categorical_cols:
            st.sidebar.markdown("### Data Filtering")
            filtered_df = df.copy()
            for col in categorical_cols:
                unique_vals = df[col].dropna().unique().tolist()
                selected_vals = st.sidebar.multiselect(f"Filter {col}", options=unique_vals, default=unique_vals)
                filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
            st.subheader("Filtered Data Preview")
            st.write(filtered_df.head())

        # 9. Pivot Table Generator
        st.sidebar.markdown("### Pivot Table Generator")
        pivot_index = st.sidebar.selectbox("Select Pivot Index", options=df.columns, key="pivot_index")
        pivot_columns = st.sidebar.selectbox("Select Pivot Columns", options=df.columns, key="pivot_columns")
        if numeric_cols:
            pivot_values = st.sidebar.selectbox("Select Value Column", options=numeric_cols, key="pivot_values")
        else:
            pivot_values = None
        if st.sidebar.button("Generate Pivot Table"):
            if pivot_values:
                pivot_table = pd.pivot_table(df, index=pivot_index, columns=pivot_columns, values=pivot_values, aggfunc='mean')
                st.subheader("Pivot Table")
                st.write(pivot_table)
            else:
                st.error("No numeric columns available for pivot values.")

        # 10. Save Dashboard Configuration
        config = {
            "color_scheme": color_scheme,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "date_columns": date_cols,
        }
        st.sidebar.markdown("### Dashboard Configuration")
        st.sidebar.download_button("Download Dashboard Config", data=json.dumps(config, indent=4), file_name="dashboard_config.json", mime="application/json")

        # 11. Auto Report Generation (Simple HTML Report)
        st.subheader("Auto Report Generation")
        report_html = f"""
        <html>
        <head><title>Data Analysis Report</title></head>
        <body>
        <h1>Data Analysis Report</h1>
        <h2>Data Summary</h2>
        {df.describe().to_html()}
        <h2>Outlier Summary</h2>
        <p>{json.dumps(outlier_summary)}</p>
        </body>
        </html>
        """
        st.download_button("Download HTML Report", data=report_html, file_name="data_analysis_report.html", mime="text/html")
