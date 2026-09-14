# Medical Insurance Cost – Statistical Analysis & Interactive Dashboard

## DS602 – Statistical Methods | Lab-4

This project applies statistical methods to the Medical Insurance Cost dataset and presents the analysis through an interactive Streamlit dashboard.

### Live Dashboard

[Open Interactive Streamlit Dashboard](https://ameya-trivedi202618047ds602statistical-methods-guaqyedncrvwkwy.streamlit.app/)

## 1. Project Overview

The project studies medical insurance charges using:

- Exploratory Data Analysis (EDA)
- Descriptive statistics
- Distribution and bivariate analysis
- Correlation analysis
- Hypothesis testing
- Multiple Linear Regression using OLS
- Regression diagnostics
- Multicollinearity analysis using VIF
- Live insurance-charge prediction

The dashboard provides interactive filters, statistical tests, visualizations, model results, diagnostics, and prediction.

## 2. Dataset

Dataset file: `data/insurance.csv`

The dataset contains 1,338 observations and the following variables:

| Variable | Type | Description |
|---|---|---|
| `age` | Numerical | Age of primary beneficiary |
| `sex` | Categorical | Sex of beneficiary |
| `bmi` | Numerical | Body Mass Index |
| `children` | Count | Number of children/dependents |
| `smoker` | Categorical | Smoking status |
| `region` | Categorical | Geographic region |
| `charges` | Numerical | Medical insurance charges |

Target variable for regression: `charges`.

## 3. Statistical Methods

### Descriptive Statistics
The project calculates:

- Mean
- Median
- Standard deviation
- IQR
- Skewness
- Kurtosis

### Hypothesis Test 1 – Two-Group Comparison

The dashboard allows a categorical grouping variable and numerical metric to be selected.

Assumption checks:

1. Shapiro-Wilk normality test
2. Levene's test for variance equality

Depending on the assumptions, the application uses:

- Independent two-sample t-test, or
- Mann-Whitney U test

### Hypothesis Test 2 – Chi-Square Test

The Chi-Square test of independence is used to examine the association between two categorical variables.

Decision rule at α = 0.05:

- p-value < 0.05 → Reject H0
- p-value ≥ 0.05 → Fail to Reject H0

### Multiple Linear Regression

Medical insurance charges are modeled using:

- Age
- BMI
- Children
- Sex
- Smoker
- Region

Categorical predictors are dummy encoded and the model is fitted using `statsmodels.api.OLS`.

The dashboard reports:

- Regression coefficients
- Standard errors
- t-statistics
- p-values
- 95% confidence intervals
- R²
- Adjusted R²
- F-statistic

### Regression Diagnostics

The project includes:

- Residuals-vs-fitted plot
- Q-Q plot
- Jarque-Bera normality test
- Breusch-Pagan heteroscedasticity test
- VIF for continuous predictors

## 4. Streamlit Dashboard

The dashboard contains three main areas:

### Tab 1 – Data Exploration
- Sidebar filters
- Dataset preview
- Descriptive statistics
- Distribution plots
- Scatter plots
- Correlation heatmap

### Tab 2 – Hypothesis Testing Lab
- Dynamic group selection
- Numerical metric selection
- Shapiro-Wilk test
- Levene's test
- t-test / Mann-Whitney U test
- Chi-Square test
- p-values and conclusions

### Tab 3 – Live Prediction & Diagnostics
- User input for predictor values
- Predicted medical insurance charge
- 95% prediction interval
- OLS results
- Residual diagnostics
- Jarque-Bera test
- Breusch-Pagan test
- VIF

## 5. Technologies

- Python
- Pandas
- NumPy
- SciPy
- Statsmodels
- Streamlit
- Plotly
- Seaborn
- Matplotlib
- Git / GitHub

## 6. Project Structure

```text
Medical-Insurance-Dashboard/
├── data/
│   └── insurance.csv
├── app.py
├── analysis.py
├── requirements.txt
├── README.md
├── Documentation_DS602.pdf
└── .gitignore
```

## 7. Installation and Running

Create a virtual environment:

```text
python -m venv .venv
```

Activate it on Windows:

```text
.venv\Scripts\activate
```

Install dependencies:

```text
pip install -r requirements.txt
```

Run the dashboard:

```text
streamlit run app.py
```

The local application normally opens at:

```text
http://localhost:8501
```

## 8. Files

- `app.py` – Streamlit dashboard
- `analysis.py` – statistical analysis, hypothesis testing, OLS, diagnostics, VIF, and prediction
- `data/insurance.csv` – dataset
- `requirements.txt` – Python dependencies
- `README.md` – project overview and usage
- `Documentation_DS602.pdf` – detailed academic documentation
- `.gitignore` – excludes environment and cache files

## 9. Reproducibility

The numerical statistical results are generated directly from `data/insurance.csv` through the Python analysis functions and Streamlit dashboard. This avoids manually entering results into the documentation.

## 10. Conclusion

This project demonstrates an end-to-end statistical workflow for medical insurance costs, combining EDA, descriptive statistics, hypothesis testing, multiple linear regression, diagnostics, VIF analysis, and interactive prediction in Streamlit.

## Academic Information

- Course: DS602 – Statistical Methods
- Lab: Lab-4 – Applied Statistical Modeling & Interactive Web Dashboard
- Program: M.Sc. Data Science
- Semester: 1
- Student: Ameya Trivedi
- Student ID: 202618047
