<div align="center">

# -- ! Data Profiling ! --
### *End-to-End Data Preprocessing, Feature Engineering & Exploratory Data Analysis*

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Statistical-4C72B0?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org/)

<br/>

> *"Data is the new oil — but only after it's been refined."*

</div>

---

## 📋 Table of Contents

- [📌 Overview](#-overview)
- [🎯 Problem Statement](#-problem-statement)
- [✨ Key Features](#-key-features)
- [🏗️ Project Structure](#️-project-structure)
- [🔄 Project Workflow](#-project-workflow)
- [📦 Part A — Fundamentals](#-part-a--fundamentals)
- [📥 Part B — Data Acquisition](#-part-b--data-acquisition)
- [🧹 Part C — Data Understanding & Cleaning](#-part-c--data-understanding--cleaning)
- [📊 Part D — Exploratory Data Analysis](#-part-d--exploratory-data-analysis)
- [🔬 Part E — Data Profiling](#-part-e--data-profiling)
- [🛠️ Tech Stack](#️-tech-stack)
- [📈 Results & Insights](#-results--insights)
- [🏆 Advantages](#-advantages)
- [📄 License](#-license)
- [👤 Author](#-author)
- [🙏 Acknowledgements](#-acknowledgements)

---

## 📌 Overview

The **Data Profiling Objective** is a comprehensive, real-world data science project focused on **Data Preprocessing** and **Feature Engineering**. It simulates the workflow of a **Junior Data Analyst** at a consumer insights company, covering every phase from raw data ingestion to generating a complete profiling report that makes datasets machine-learning ready.

This project is designed to:
- Build strong data intuition through hands-on exploration
- Handle messy, multi-source data as encountered in real industry settings
- Apply structured data cleaning, transformation, and analysis techniques
- Produce actionable insights from customer behavior data

---

## 🎯 Problem Statement

> **Objective:** Predict whether a customer will churn based on their purchase behavior.

You have been hired as a **Junior Data Analyst** at a consumer insights company. The company has provided a dataset containing customer purchase behavior collected from **multiple data sources** — CSV files, JSON files, a SQL database, and a live REST API.

| 📂 Data Source | 📄 Format | 🔍 Content |
|---------------|-----------|-----------|
| Internal CRM Export | `.csv` | Customer demographics |
| Behavior Logs | `.json` | Clickstream & session data |
| Transactional DB | `SQL` | Purchase history |
| Third-Party Service | `REST API` | External enrichment data |

The goal is to frame a machine learning problem (predict customer churn) and perform comprehensive **data preprocessing and profiling** to make the dataset ML-ready.

---

## ✨ Key Features

| Feature | Description |
|--------|-------------|
| 🔌 **Multi-Source Ingestion** | Load data from CSV, JSON, SQL, and REST APIs |
| 🧹 **Smart Data Cleaning** | Handle nulls, duplicates, type mismatches, and outliers |
| 📊 **Rich EDA** | Univariate, bivariate, and multivariate visual analysis |
| 📐 **Tensor Fundamentals** | NumPy-based tensor theory with practical examples |
| 🧮 **Automated Profiling** | Full Pandas Profiling report generation |
| 🔁 **Reproducible Pipeline** | Step-by-step documented, end-to-end notebook |
| 🎯 **ML-Ready Output** | Clean, encoded, and normalized features for modeling |
| 📈 **Correlation Insights** | Heatmaps and pair plots for feature relationship analysis |

---

## 🏗️ Project Structure

```
📦 data-profiling/
│
├── 📁 data/
│   ├── 📄 data.csv
│   ├── 📄 Employee.csv      
│   ├── 📄 Employees.json     
│   └── 📄 profiling_report.html   
│
├── 📁 notebooks/
│   ├── 📓 Data_profiling.ipynb    
│
├── 📁 outputs/
│   ├── 📊 univariate_plots/
│   ├── 📊 bivariate_plots/
│   └── 📊 multivariate_plots/
│
├── 📄 README.md
└── 📄 LICENSE
```

---

## 🔄 Project Workflow

```
Raw Data Sources
      │
      ▼
┌─────────────────────────────┐
│   Part A: Fundamentals      │  ← Theory, Tensors, Problem Framing
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Part B: Data Acquisition  │  ← CSV, JSON, SQL, API
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Part C: Data Cleaning     │  ← Nulls, Duplicates, Types
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Part D: EDA               │  ← Uni/Bi/Multivariate Analysis
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Part E: Profiling Report  │  ← Automated HTML Report
└─────────────────────────────┘
             │
             ▼
     ML-Ready Dataset ✅
```

---

## 📦 Part A — Fundamentals

### 📝 1. What is Data Analysis?

Data Analysis is the systematic process of inspecting, cleaning, transforming, and modeling data with the goal of discovering useful information, drawing conclusions, and supporting decision-making. It bridges raw data and meaningful insights using statistical and computational methods.

---

### 🗺️ 2. Data Science Project Planning — Steps

| Step | Phase | Description |
|------|-------|-------------|
| 1️⃣ | **Problem Definition** | Identify the business question and success criteria |
| 2️⃣ | **Data Collection** | Gather data from all relevant sources |
| 3️⃣ | **Data Exploration** | Understand structure, types, and distributions |
| 4️⃣ | **Data Cleaning** | Fix quality issues — nulls, duplicates, errors |
| 5️⃣ | **Feature Engineering** | Create or transform features for modeling |
| 6️⃣ | **Modeling** | Select, train, and evaluate ML algorithms |
| 7️⃣ | **Evaluation** | Validate using appropriate metrics |
| 8️⃣ | **Deployment** | Push model to production or reporting environment |
| 9️⃣ | **Monitoring** | Track model performance over time |

---

### 🤖 3. ML Problem Statement — Customer Churn Prediction

> **Problem Type:** Binary Classification
>
> **Target Variable:** `churn` (1 = Churned, 0 = Retained)
>
> **Input Features:** Purchase frequency, recency, monetary value, age, income, session count, support tickets, contract type
>
> **Goal:** Given historical customer behavior data, predict with high accuracy whether a customer will stop using the service within the next 30 days.
>
> **Success Metric:** F1-Score ≥ 0.80, AUC-ROC ≥ 0.85

---

### 🧮 4. Tensors — In-Depth Explanation

**What is a Tensor?**

A **Tensor** is a multi-dimensional mathematical object used to represent data of varying complexity. Tensors generalize scalars, vectors, and matrices to higher dimensions and are the fundamental data structure in machine learning.

| Tensor Type | Dimensions | Example |
|-------------|-----------|---------|
| 🔵 Scalar | 0D | A single temperature reading: `42.5` |
| 📏 Vector | 1D | A list of prices: `[10, 20, 30]` |
| 📊 Matrix | 2D | A dataset table: rows × columns |
| 🧊 3D Tensor | 3D | Batch of images: samples × height × width |
| 🔷 nD Tensor | nD | Video data, NLP embeddings |

**NumPy Tensor Creation Methods:**

- `np.array()` → Create from Python list
- `np.zeros()` → Initialize with zeros
- `np.ones()` → Initialize with ones
- `np.random.rand()` → Random uniform values
- `np.random.randn()` → Random normal values
- `arr.reshape()` → Change tensor dimensions
- `arr.T` → Transpose
- `np.dot()` → Dot product

---

## 📥 Part B — Data Acquisition

### 🔌 5. Multi-Source Data Import

#### 📄 CSV Loading with Pandas
- **Method:** `pd.read_csv()`
- **Key Parameters:** `filepath`, `sep`, `encoding`, `parse_dates`, `dtype`
- **Use case:** Structured flat files exported from CRM or spreadsheets

#### 🗂️ JSON Parsing
- **Method:** `pd.read_json()` or Python's built-in `json.load()`
- **Key Parameters:** `orient`, `lines`, `dtype`
- **Use case:** Nested behavior logs, API response snapshots

#### 🗄️ SQL Database Connection
- **Method:** `pd.read_sql()` with `sqlalchemy.create_engine()`
- **Key Parameters:** `con` (connection string), `sql` (query string)
- **Supported DBs:** PostgreSQL, MySQL, SQLite, Microsoft SQL Server
- **Use case:** Transactional records stored in relational databases

#### 🌐 REST API Data Fetch
- **Method:** `requests.get()` followed by `.json()` parsing
- **Example APIs:** [RandomUser API](https://randomuser.me/), [JSONPlaceholder](https://jsonplaceholder.typicode.com/)
- **Key Steps:** Send GET request → Parse JSON response → Normalize with `pd.json_normalize()`
- **Use case:** Real-time or periodically updated external data enrichment

---

## 🧹 Part C — Data Understanding & Cleaning

### 🔍 6. Initial Data Exploration

| Method | Purpose |
|--------|---------|
| `.head(n)` | View first n rows of the DataFrame |
| `.tail(n)` | View last n rows of the DataFrame |
| `.info()` | Column names, non-null counts, and dtypes |
| `.describe()` | Statistical summary of numeric columns |
| `.shape` | Number of rows and columns as a tuple |
| `.dtypes` | Data type of each column |
| `.isnull().sum()` | Count of missing values per column |
| `.duplicated().sum()` | Count of fully duplicate rows |
| `.nunique()` | Number of unique values per column |
| `.value_counts()` | Frequency distribution of categorical values |

---

### 🧽 7. Data Cleaning Steps

#### 🩹 Handling Missing Values
- **Strategy 1 — Imputation (Numerical):** Fill with `.mean()`, `.median()`, or `SimpleImputer`
- **Strategy 2 — Imputation (Categorical):** Fill with `.mode()` or a constant like `"Unknown"`
- **Strategy 3 — Removal:** Drop rows with `df.dropna()` when missingness is high (> 50%)
- **Strategy 4 — Forward Fill:** Use `df.fillna(method='ffill')` for time-series data

#### 🔧 Fixing Inconsistent Data Types
- **Method:** `df['col'].astype(dtype)`
- **Common conversions:** `object → datetime` via `pd.to_datetime()`, `string → numeric` via `pd.to_numeric(errors='coerce')`

#### 🗑️ Dropping Irrelevant Columns
- **Method:** `df.drop(columns=['col1', 'col2'])`
- **Criteria:** High percentage nulls, constant values, non-predictive identifiers (e.g., row IDs)

#### 🔄 Handling Duplicates
- **Detection:** `df.duplicated()`
- **Removal:** `df.drop_duplicates(keep='first')`

---

## 📊 Part D — Exploratory Data Analysis

### 📉 8. Univariate Analysis

> *Analysis of a single variable in isolation to understand its distribution and spread.*

| Variable | Plot Type | Key Insight |
|----------|-----------|------------|
| `Age` | Histogram + KDE | Distribution shape (skewed/normal) |
| `Income` | Box Plot | Median, quartiles, and outliers |
| `Purchases` | Count Plot / Histogram | Frequency of purchase activity |
| Categorical columns | Bar Chart | Class balance |

**Key Statistical Measures:**
- **Central Tendency:** Mean, Median, Mode
- **Spread:** Standard Deviation, Variance, IQR
- **Shape:** Skewness, Kurtosis

---

### 📈 9. Bivariate Analysis

> *Analysis of two variables to understand their relationship.*

#### Gender vs. Purchases
- **Plot:** Grouped Bar Chart, Box Plot by Gender
- **Method:** `sns.boxplot(x='Gender', y='Purchases')`
- **Insight:** Determine if purchase behavior differs significantly across gender categories

#### Income vs. Churn
- **Plot:** Violin Plot, KDE Plot by Churn status
- **Method:** `sns.violinplot(x='Churn', y='Income')`
- **Insight:** Identify whether income level is a strong predictor of churn likelihood

---

### 🌐 10. Multivariate Analysis

> *Analysis of three or more variables simultaneously to uncover interactions.*

#### Correlation Heatmap
- **Method:** `sns.heatmap(df.corr(), annot=True, cmap='coolwarm')`
- **Purpose:** Identify strongly correlated feature pairs; detect multicollinearity
- **Insight:** Guides feature selection before modeling

#### Pair Plots
- **Method:** `sns.pairplot(df, hue='Churn')`
- **Purpose:** Visualize pairwise feature distributions and scatter relationships
- **Insight:** Spot natural clusters and linear separability between churn classes

---

## 🔬 Part E — Data Profiling

### 📋 11. Pandas Profiling Report

> Automated, comprehensive report generated using `ydata-profiling` (formerly `pandas-profiling`).

**Installation:**
```
pip install ydata-profiling
```

**Report Generation:**
- **Class:** `ProfileReport(df, title="Customer Churn Analysis")`
- **Method:** `.to_file("profiling_report.html")`

**Report Sections:**

| Section | Content |
|---------|---------|
| 📊 **Overview** | Dataset shape, memory usage, variable types |
| 📉 **Variables** | Per-column stats: min, max, mean, std, histogram |
| ⚠️ **Warnings** | Flags for high cardinality, skewness, high correlation, constant columns |
| 🔗 **Correlations** | Pearson, Spearman, Cramér's V matrices |
| ❌ **Missing Values** | Heatmap and bar chart of missing data patterns |
| 🔀 **Interactions** | Scatter plots for selected variable pairs |
| 📄 **Sample** | First and last rows of the dataset |

**Profiling Report Highlights:**
- Automatically detects and warns on **zero variance** columns
- Identifies **high cardinality** categorical features
- Reports **skewness** and suggests transformations
- Flags **potential outliers** based on IQR method
- Provides a downloadable, shareable `.html` report

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| 🐍 **Python** | 3.8+ | Core programming language |
| 🐼 **Pandas** | 2.0+ | Data manipulation and analysis |
| 🔢 **NumPy** | 1.24+ | Numerical computing and tensors |
| 📊 **Matplotlib** | 3.7+ | Base plotting library |
| 🌊 **Seaborn** | 0.12+ | Statistical data visualization |
| 🔍 **ydata-profiling** | 4.5+ | Automated data profiling reports |
| 🗄️ **SQLAlchemy** | 2.0+ | SQL database ORM and connection |
| 🌐 **Requests** | 2.31+ | HTTP requests for REST API |
| 📓 **Jupyter Notebook** | 7.0+ | Interactive development environment |
| 🔧 **Scikit-learn** | 1.3+ | Data preprocessing utilities |

---


---

## 📈 Results & Insights

After completing all five parts, the following outputs are produced:

- ✅ **Cleaned Dataset** — ML-ready with no missing values or inconsistencies
- 📊 **15+ Visualizations** — Distribution plots, relationship plots, heatmaps, pair plots
- 📋 **Profiling Report** — Full HTML report with data quality warnings and statistics
- 🔍 **Feature Insights** — Identified top features correlated with customer churn
- 📐 **Tensor Examples** — Documented NumPy-based tensor operations with output

---

## 🏆 Advantages

| Advantage | Detail |
|-----------|--------|
| 🔄 **Reusability** | Modular functions work on any tabular dataset |
| 📚 **Educational** | Step-by-step explanations for every concept |
| 🌍 **Real-World Relevance** | Mimics actual analyst workflows at industry companies |
| ⚡ **Efficiency** | Automated profiling saves hours of manual inspection |
| 🧪 **Reproducibility** | Seeded random states and documented steps |
| 🔌 **Multi-Format Support** | Handles CSV, JSON, SQL, and API — all in one project |
| 📖 **Beginner Friendly** | Well-commented code and rich documentation |
| 🛡️ **Data Quality Focus** | Warnings and checks at every stage of the pipeline |

---

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full details.

```
MIT License — Free to use, modify, and distribute with attribution.
```

---

## 👤 Author

<div align="center">

### Ayush Isamaliya

[![GitHub](https://img.shields.io/badge/GitHub-yourhandle-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/isamaliya16)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ayush-isamaliya-686533312/)

> *"Turning raw data into decisions, one insight at a time."*

**🎓 Role:** Junior Data Analyst | Data Science Enthusiast \
**📍 Location:** India\
**🛠️ Skills:** Python · Pandas · SQL · Machine Learning · Data Visualization

</div>

---

## 🙏 Acknowledgements

Special thanks to the following resources and communities that made this project possible:

- 📚 [Pandas Documentation](https://pandas.pydata.org/docs/) — Official Pandas reference
- 📊 [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html) — Visualization inspiration
- 🔬 [ydata-profiling](https://github.com/ydataai/ydata-profiling) — Automated profiling library
- 🤖 [Scikit-learn Docs](https://scikit-learn.org/stable/) — Preprocessing utilities
- 🌐 [JSONPlaceholder](https://jsonplaceholder.typicode.com/) — Free REST API for testing
- 🧮 [NumPy Documentation](https://numpy.org/doc/) — Tensor and array operations
- 💬 [Stack Overflow Community](https://stackoverflow.com/) — Problem-solving support
- 📖 [Kaggle Learn](https://www.kaggle.com/learn) — Data science learning platform

---

<div align="center">

---

*Made with ❤️ and ☕ — Last updated: 20 March,2026*

</div>
