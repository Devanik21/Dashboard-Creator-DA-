import streamlit as st
import pandas as pd
import altair as alt

st.title("Automatic Dashboard Creator")
st.write("Upload a CSV or Excel file to automatically generate dashboards.")

# File uploader accepts CSV and Excel files
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read file into a DataFrame
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error("Unsupported file format")
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

        st.subheader("Automatic Visualizations")

        # Visualize numeric columns: histograms
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            st.write("### Histograms for Numeric Columns")
            for col in numeric_cols:
                st.write(f"**Histogram for {col}:**")
                chart = alt.Chart(df).mark_bar().encode(
                    alt.X(f"{col}:Q", bin=True),
                    y='count()'
                ).properties(width=600, height=300)
                st.altair_chart(chart, use_container_width=True)
        else:
            st.write("No numeric columns found for histograms.")

        # Visualize categorical columns: bar charts
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            st.write("### Bar Charts for Categorical Columns")
            for col in categorical_cols:
                st.write(f"**Bar Chart for {col}:**")
                chart = alt.Chart(df).mark_bar().encode(
                    x=f"{col}:N",
                    y='count()'
                ).properties(width=600, height=300)
                st.altair_chart(chart, use_container_width=True)
        else:
            st.write("No categorical columns found for bar charts.")

        # Create a correlation heatmap for numeric columns if possible
        if len(numeric_cols) >= 2:
            st.write("### Correlation Heatmap")
            corr = df[numeric_cols].corr()
            # Melt the correlation matrix into long format for Altair
            corr_long = corr.reset_index().melt('index')
            heatmap = alt.Chart(corr_long).mark_rect().encode(
                x=alt.X('variable:O', title='Variable'),
                y=alt.Y('index:O', title='Variable'),
                color=alt.Color('value:Q', scale=alt.Scale(scheme='viridis'))
            ).properties(width=600, height=400)
            st.altair_chart(heatmap, use_container_width=True)
        else:
            st.write("Not enough numeric columns for a correlation heatmap.")
