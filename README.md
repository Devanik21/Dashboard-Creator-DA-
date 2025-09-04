# 📊 Advanced Data Explorer & Visualizer 🔮

**Your All-in-One Solution for Interactive Data Analysis, Visualization, and AI-Powered Insights.**

This Streamlit application empowers users to upload datasets and perform a wide array of analyses, from basic data profiling and cleaning to advanced machine learning, time series forecasting, geospatial visualization, and AI-driven insights using Google's Gemini API.
 
---

🧪 **Demo**  [https://63uxugggjkkrghy7vuxkeh.streamlit.app]()



---

✨ **Overview & Key Features**

This application is designed to be a comprehensive toolkit for data analysts, scientists, and enthusiasts. It offers a rich set of features, including:

*   **Data Ingestion & Profiling:**
    *   Upload multiple CSV, XLSX, or JSON files.
    *   Quick data overview (dimensions, missing values, data types).
    *   Advanced data profiling with quality scores.
    *   Smart data type detection and conversion suggestions.
*   **Interactive Data Manipulation:**
    *   Advanced filtering system for numeric and categorical columns.
    *   User-defined calculated fields using formulas.
    *   Data deduplication utility.
    *   Interactive data binning.
    *   Column renamer and value replacer.
*   **Visualization Suite:**
    *   Quick visualization builder (Scatter, Line, Bar, Histogram, Box).
    *   Interactive chart customization with Plotly Express (color, faceting).
    *   Geospatial data visualization (point maps, heatmaps).
    *   Network analysis for categorical co-occurrence.
    *   Dendrograms for hierarchical clustering.
*   **Statistical Analysis:**
    *   Correlation analysis (Pearson, Spearman, Kendall) with heatmaps.
    *   A/B testing and ANOVA suite.
    *   Distribution fitting and goodness-of-fit tests.
    *   Cross-tabulation / Contingency tables with Chi-squared tests.
*   **Machine Learning Lab:**
    *   **Supervised Learning:**
        *   Automated predictive pipeline (Linear Regression, Polynomial Regression, Random Forest Regressor).
        *   Decision Tree and Random Forest explorers (for classification and regression) with hyperparameter tuning.
        *   Predictive Customer Churn Model.
        *   Propensity Scoring Model.
    *   **Unsupervised Learning:**
        *   K-Means clustering analysis.
        *   Anomaly detection dashboard (IQR, Z-Score, Isolation Forest).
        *   Hierarchical clustering.
        *   Latent Dirichlet Allocation (LDA) for topic modeling.
        *   Principal Component Analysis (PCA) explorer.
    *   **Model Interpretability:**
        *   SHAP (SHapley Additive exPlanations) for Random Forest models.
        *   Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE) plots.
*   **Time Series Analysis:**
    *   Trend analysis and simple forecasting (Exponential Smoothing).
    *   Advanced forecasting with Prophet.
    *   Automated time series anomaly detection (STL).
    *   Time-Lagged Cross-Correlation analysis.
*   **Specialized Analytics:**
    *   Market Basket Analysis (Association Rules).
    *   Advanced Cohort Analysis (Retention & Behavior).
    *   Customer Lifetime Value (CLV) Profiler.
    *   Survival Analysis (Kaplan-Meier & Cox PH Model).
    *   Simplified Treatment Effect Estimation.
    *   Key Drivers Analysis.
    *   Comparative Product Performance (Top vs. Bottom N%).
    *   Dynamic Pricing Simulation.
    *   Sales Funnel Conversion Analysis.
*   **AI-Powered Insights (Google Gemini):**
    *   Ask questions about your data in natural language.
    *   Automated narrative report generation.
    *   AI chart-to-text summarizer.
    *   Anomaly investigation and explanation.
    *   Inventory optimization suggestions.
    *   Predictive maintenance advisor (conceptual).
    *   Scenario planning and impact analysis.
    *   AI-powered segment narrative generator.
*   **Data Interaction & Utilities:**
    *   SQL Query Workbench (using DuckDB).
    *   Excel-like Query Workbench (pandas `query()` syntax).
    *   Data Dictionary Generator.
    *   Random Row Sampler.
    *   Duplicate Column Finder & Column Value Counter.
*   **Customization & Export:**
    *   Theme selection (light, dark, cyberpunk) and custom theme designer.
    *   Export filtered data and generated reports.
    *   Auto-refresh option for dashboards.

---

🛠️ **Tech Stack**

*   **Core:** Streamlit, Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn, Plotly (Express & Graph Objects), Altair, Folium, WordCloud, NetworkX
*   **Machine Learning:** Scikit-learn (KMeans, IsolationForest, PCA, Classifiers, Regressors, Preprocessing, Metrics, etc.)
*   **Statistical Analysis:** SciPy (stats), Statsmodels
*   **Specialized Libraries:**
    *   `mlxtend` (Market Basket Analysis)
    *   `nltk` (Sentiment Analysis - VADER)
    *   `lifelines` (Survival Analysis)
    *   `duckdb` (SQL Query Workbench)
    *   `shap` (Model Explainability)
    *   `prophet` (Time Series Forecasting)
*   **AI Integration:** Google Generative AI (`google-generativeai` for Gemini API)

---

🧠 **Setup & Run**

1.  **Clone this repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install requirements:**
    A `requirements.txt` file would typically list all necessary packages. You can generate one using `pip freeze > requirements.txt` after installing everything. Key packages include:
    ```bash
    pip install streamlit pandas numpy altair matplotlib seaborn plotly scikit-learn folium google-generativeai scipy statsmodels mlxtend nltk networkx wordcloud lifelines duckdb shap prophet
    ```
    *(You might need to install `nltk` data separately, e.g., `nltk.download('vader_lexicon')`)*

4.  **Set up API Key (for AI features):**
    *   Create a file named `.streamlit/secrets.toml` in your project root.
    *   Add your Gemini API key:
        ```toml
        GEMINI_API_KEY = "YOUR_API_KEY_HERE"
        ```
    *   Alternatively, you can enter the API key directly in the application's sidebar during runtime.

5.  **Run the Streamlit app:**
    ```bash
    streamlit run app2.py
    ```

---

👨‍💻 **Author**

*   **Devanik Saha**
    *   GitHub: [Devanik21](https://github.com/Devanik21)
    *   LinkedIn: [Devanik Debnath](https://www.linkedin.com/in/devanik/)
    *   *National Institute of Technology | ECE | Passionate about AI, ML, and Cryptography*

---

📜 **License**

This project is licensed under the MIT License - see the LICENSE file for details. Feel free to use, remix, and build upon it! 💖

---

✨ **Built with love, magic, and Gemini** ✨

Need help with deployment, datasets, or a landing page? Ping me~ ☁️🌈
