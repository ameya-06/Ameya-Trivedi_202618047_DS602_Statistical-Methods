# Medical Insurance Cost – Statistical Analysis & Interactive Dashboard

## DS602 – Statistical Methods | Lab-4

This project performs statistical analysis of the Medical Insurance Cost dataset and presents the results through an interactive Streamlit dashboard.

The project covers Exploratory Data Analysis (EDA), descriptive statistics, hypothesis testing, multiple linear regression using Ordinary Least Squares (OLS), regression diagnostics, multicollinearity analysis, and live prediction.

## 1. Project Overview

The main objective of this project is to analyze medical insurance charges and study how demographic and lifestyle factors are associated with insurance costs.

### Live Dashboard

[Open Interactive Streamlit Dashboard](https://ameya-trivedi202618047ds602statistical-methods-guaqyedncrvwkwy.streamlit.app/)

The project includes:

- Exploratory Data Analysis (EDA)
- Descriptive statistics
- Measures of central tendency and dispersion
- Distribution analysis
- Bivariate analysis
- Correlation analysis
- Hypothesis testing
- Parametric and non-parametric testing
- Chi-Square test of independence
- Multiple Linear Regression
- Ordinary Least Squares (OLS)
- Regression coefficient interpretation
- Model significance
- R-squared and Adjusted R-squared
- Residual analysis
- Q-Q plot
- Jarque-Bera normality test
- Breusch-Pagan heteroscedasticity test
- Variance Inflation Factor (VIF)
- Interactive Streamlit dashboard
- Live insurance charge prediction

## 2. Dataset

The project uses the Medical Insurance Cost dataset.

Dataset file: `data/insurance.csv`

The dataset contains the following variables:

- `age` – Age of the primary beneficiary
- `sex` – Gender of the beneficiary
- `bmi` – Body Mass Index
- `children` – Number of children/dependents
- `smoker` – Smoking status
- `region` – Residential region
- `charges` – Medical insurance charges

The dependent or target variable used in the regression model is `charges`.

## 3. Technologies and Libraries Used

The project is implemented using Python and the following libraries:

### Pandas

Used for loading the dataset, data cleaning, data manipulation, filtering, descriptive statistics, and DataFrame operations.

### NumPy

Used for numerical calculations, array operations, and mathematical computations.

### SciPy

Used for statistical hypothesis testing, including Shapiro-Wilk normality test, Levene's test, independent two-sample t-test, Mann-Whitney U test, Chi-Square test, and One-Way ANOVA.

### Statsmodels

Used for Ordinary Least Squares regression, multiple linear regression, regression coefficients, standard errors, t-statistics, p-values, confidence intervals, R-squared, Adjusted R-squared, F-statistic, Jarque-Bera test, Breusch-Pagan test, and Variance Inflation Factor (VIF).

### Streamlit

Used to build the interactive web-based statistical dashboard.

### Plotly

Used for interactive charts and visualizations.

### Seaborn

Used for statistical visualizations such as histograms, distribution plots, scatter plots, and correlation heatmaps.

### Matplotlib

Used for statistical and regression diagnostic plots such as residual plots and Q-Q plots.

## 4. Project Structure

The project structure is:

Medical-Insurance-Dashboard/
├── data/
│   └── insurance.csv
├── app.py
├── analysis.py
├── requirements.txt
├── README.md
└── .gitignore

## 5. Exploratory Data Analysis

Exploratory Data Analysis is performed to understand the structure, distribution, and relationships present in the dataset.

The analysis includes:

- Dataset dimensions
- Dataset preview
- Data types
- Missing-value checking
- Numerical summaries
- Categorical variable analysis
- Distribution analysis
- Bivariate analysis
- Correlation analysis

The Streamlit dashboard allows users to interactively filter the dataset and explore the resulting changes in the visualizations.

## 6. Descriptive Statistics

Descriptive statistics are calculated for the numerical variables.

The following measures are included:

### Mean

The arithmetic average of the observations.

### Median

The middle value of the observations after sorting.

### Standard Deviation

Measures the spread of observations around the mean.

### Quartiles

The first quartile (Q1), median (Q2), and third quartile (Q3) are calculated.

### Interquartile Range

IQR = Q3 - Q1

It represents the spread of the middle 50% of observations.

### Skewness

Skewness is used to measure the asymmetry of a distribution.

### Kurtosis

Kurtosis is used to describe the shape and tail behaviour of a distribution.

## 7. Distribution Analysis

Distribution plots are used to understand the behaviour of the numerical variables.

The main variables examined include:

- Age
- BMI
- Children
- Charges

Histograms and KDE plots are used to visualize the distributions.

These plots help identify central tendency, spread, skewness, possible outliers, and overall distribution shape.

## 8. Bivariate Analysis

Bivariate analysis is used to study relationships between two variables.

The dashboard provides interactive visualizations such as:

- Age vs Charges
- BMI vs Charges
- Children vs Charges

Categorical variables can also be compared with insurance charges using appropriate visualizations.

## 9. Correlation Analysis

A correlation matrix is used to study the linear relationships between numerical variables.

Correlation coefficients range from -1 to +1.

A positive value indicates a positive linear relationship, while a negative value indicates a negative linear relationship.

A correlation heatmap is provided in the dashboard for easier interpretation.

## 10. Hypothesis Testing

Hypothesis testing is used to make statistical decisions about relationships and differences in the dataset.

All hypothesis tests use the significance level:

α = 0.05

The general decision rule is:

If p-value < 0.05:
Reject H0

If p-value >= 0.05:
Fail to Reject H0

## 11. Hypothesis Test 1 – Two Group Comparison

The first hypothesis-testing procedure compares a continuous numerical variable between two groups.

For example, smokers and non-smokers can be compared using medical insurance charges.

The dashboard allows the user to select the grouping variable, numerical metric, and two groups.

### 11.1 Shapiro-Wilk Normality Test

The Shapiro-Wilk test is used to examine whether the selected groups approximately follow a normal distribution.

Null Hypothesis (H0):

The data follows a normal distribution.

Alternative Hypothesis (H1):

The data does not follow a normal distribution.

The test is evaluated at α = 0.05.

### 11.2 Levene's Test

Levene's test is used to examine whether the two groups have equal variances.

Null Hypothesis (H0):

The variances of the two groups are equal.

Alternative Hypothesis (H1):

The variances of the two groups are different.

### 11.3 Independent Two-Sample t-Test

If the assumptions for a parametric comparison are satisfied, an independent two-sample t-test is performed.

Null Hypothesis (H0):

The means of the two groups are equal.

Alternative Hypothesis (H1):

The means of the two groups are significantly different.

The final decision is based on the p-value.

### 11.4 Mann-Whitney U Test

If the normality assumption is not satisfied, the non-parametric Mann-Whitney U test is used.

Null Hypothesis (H0):

There is no statistically significant difference between the two groups.

Alternative Hypothesis (H1):

There is a statistically significant difference between the two groups.

This provides a non-parametric alternative to the independent two-sample t-test.

## 12. Hypothesis Test 2 – Chi-Square Test of Independence

The second hypothesis-testing procedure examines the relationship between two categorical variables.

For example, smoker and region can be examined using the Chi-Square test of independence.

Null Hypothesis (H0):

The two categorical variables are independent.

Alternative Hypothesis (H1):

The two categorical variables are associated.

The test uses α = 0.05.

Decision rule:

p-value < 0.05 → Reject H0

p-value >= 0.05 → Fail to Reject H0

The dashboard displays the contingency table, Chi-Square statistic, degrees of freedom, p-value, and final conclusion.

## 13. Multiple Linear Regression

Multiple Linear Regression is used to model medical insurance charges using multiple explanatory variables.

The general regression model is:

Y = β0 + β1X1 + β2X2 + ... + βkXk + ε

where:

- Y = Medical insurance charges
- β0 = Intercept
- β1 ... βk = Regression coefficients
- X1 ... Xk = Predictor variables
- ε = Error term

The regression model is fitted using Ordinary Least Squares (OLS).

## 14. OLS Regression Model

The target variable is `charges`.

The explanatory variables include:

- Age
- BMI
- Children
- Sex
- Smoker
- Region

Categorical variables are encoded appropriately before fitting the regression model.

The OLS model provides:

- Coefficients
- Standard errors
- t-statistics
- p-values
- Confidence intervals
- R-squared
- Adjusted R-squared
- F-statistic
- Model significance

## 15. Interpretation of Regression Coefficients

Each regression coefficient represents the expected change in medical insurance charges associated with a one-unit change in the corresponding predictor while holding the other predictors constant.

For categorical variables, coefficients represent the estimated difference relative to the reference category.

A predictor is considered statistically significant at the 5% significance level when:

p-value < 0.05

The dashboard displays the estimated coefficient and its 95% confidence interval.

## 16. Model Fit

### R-Squared

R-squared represents the proportion of variation in medical insurance charges explained by the predictors included in the model.

### Adjusted R-Squared

Adjusted R-squared accounts for the number of predictors in the model and provides a more conservative measure of model fit.

### F-Test

The overall F-test evaluates whether the regression model provides statistically significant explanatory power.

## 17. Regression Diagnostics

Regression diagnostics are performed to assess whether important assumptions of the linear regression model are reasonably satisfied.

The dashboard provides graphical and statistical diagnostics.

## 18. Residuals vs Fitted Values

A residuals-vs-fitted plot is used to assess:

- Linearity
- Constant variance
- Systematic patterns in residuals

Ideally, residuals should be randomly scattered around zero without a strong systematic pattern.

## 19. Q-Q Plot

The Q-Q plot is used to visually assess the normality of the regression residuals.

If the residuals approximately follow a normal distribution, the points should lie reasonably close to the reference line.

## 20. Jarque-Bera Test

The Jarque-Bera test is used to statistically assess the normality of regression residuals.

The test is based on:

- Skewness
- Kurtosis

Null Hypothesis (H0):

The residuals are normally distributed.

Alternative Hypothesis (H1):

The residuals are not normally distributed.

The decision is made using α = 0.05.

## 21. Breusch-Pagan Test

The Breusch-Pagan test is used to test for heteroscedasticity.

Null Hypothesis (H0):

The residuals have constant variance.

Alternative Hypothesis (H1):

The residual variance is not constant.

Decision rule:

p-value < 0.05 → Evidence of heteroscedasticity

p-value >= 0.05 → No significant evidence of heteroscedasticity

## 22. Multicollinearity – VIF

Variance Inflation Factor (VIF) is used to assess multicollinearity among explanatory variables.

A high VIF indicates that a predictor may have a strong linear relationship with other predictors.

VIF values are displayed in the dashboard for the relevant predictors.

This helps identify possible multicollinearity problems before interpreting regression coefficients.

## 23. Streamlit Dashboard

The project contains an interactive Streamlit dashboard divided into three main tabs.

### Tab 1 – Data Exploration

The Data Exploration tab contains:

- Interactive sidebar filters
- Dataset preview
- Dataset dimensions
- Descriptive statistics
- Numerical summaries
- Distribution plots
- Histograms
- KDE plots
- Scatter plots
- Correlation heatmap

The user can change filters and explore the dataset interactively.

### Tab 2 – Hypothesis Testing Lab

The Hypothesis Testing Lab provides an interactive statistical testing interface.

The user can select:

- Grouping variable
- Numerical metric
- Two groups for comparison

The application performs:

1. Shapiro-Wilk normality test
2. Levene's test
3. Independent two-sample t-test when appropriate
4. Mann-Whitney U test when appropriate

The tab also provides a Chi-Square test for categorical variables.

The final decision is automatically displayed using α = 0.05.

### Tab 3 – Live Prediction & Diagnostics

The Live Prediction & Diagnostics tab allows users to enter predictor values interactively.

The application generates:

- Predicted medical insurance charges
- 95% prediction interval
- Regression results
- Residual diagnostics
- Q-Q plot
- Jarque-Bera test
- Breusch-Pagan test
- VIF results

This tab combines prediction with statistical model diagnostics.

## 24. Live Prediction

The OLS regression model is used to generate predicted medical insurance charges for user-provided input values.

The user can enter or select:

- Age
- Sex
- BMI
- Number of children
- Smoking status
- Region

The dashboard then calculates the predicted insurance charge.

A 95% prediction interval is also provided to communicate uncertainty around an individual prediction.

## 25. Statistical Decision Framework

All hypothesis tests are evaluated using α = 0.05.

The decision framework is:

p-value < 0.05 → Reject the Null Hypothesis

p-value >= 0.05 → Fail to Reject the Null Hypothesis

The dashboard reports the statistical result and an interpretable conclusion.

## 26. Analysis Workflow

The complete analysis follows this workflow:

Load Dataset
↓
Data Cleaning & Validation
↓
Exploratory Data Analysis
↓
Descriptive Statistics
↓
Distribution & Relationship Analysis
↓
Hypothesis Testing
↓
OLS Multiple Linear Regression
↓
Regression Coefficient Interpretation
↓
Model Fit Evaluation
↓
Regression Diagnostics
↓
VIF / Multicollinearity Analysis
↓
Interactive Prediction

## 27. Installation

### Step 1 – Clone the Repository

Clone the project repository and open the project folder in VS Code.

### Step 2 – Create Virtual Environment

From the project root directory, run:

python -m venv .venv

### Step 3 – Activate Virtual Environment

On Windows, run:

.venv\Scripts\activate

### Step 4 – Install Required Libraries

Run this command from the project root directory:

pip install -r requirements.txt

## 28. Running the Application

Make sure the terminal is located at the project root directory:

Medical-Insurance-Dashboard

Then run:

streamlit run app.py

The Streamlit application will open in the browser.

The default local address is:

http://localhost:8501

## 29. Running the Analysis

The statistical analysis functions are contained in `analysis.py`.

The Streamlit dashboard uses these analysis functions to perform statistical calculations and display the results interactively.

## 30. File Descriptions

### app.py

Main Streamlit application containing:

- Dashboard interface
- Three tabs
- Sidebar filters
- Data exploration
- Hypothesis testing interface
- Regression prediction
- Diagnostic visualizations

### analysis.py

Contains statistical analysis functionality including:

- Descriptive statistics
- Hypothesis tests
- OLS regression
- Regression diagnostics
- VIF calculations
- Prediction-related calculations

### data/insurance.csv

Medical Insurance Cost dataset used for the analysis.

### requirements.txt

Contains all Python packages required to run the project.

### README.md

Contains project documentation, methodology, statistical procedures, installation instructions, and usage information.

## 31. Results and Findings

The application calculates the statistical results directly from the provided `insurance.csv` dataset.

The dashboard presents:

- Descriptive statistics
- Group-wise comparisons
- Hypothesis-test statistics
- p-values
- Statistical decisions
- Regression coefficients
- Confidence intervals
- R-squared
- Adjusted R-squared
- Diagnostic test results
- VIF values
- Prediction results

The numerical results are generated dynamically by the application rather than manually entered into the documentation.

## 32. Conclusion

This project demonstrates a complete statistical modeling workflow using Python and the Medical Insurance Cost dataset.

The analysis combines exploratory data analysis, descriptive statistics, hypothesis testing, multiple linear regression, OLS estimation, regression diagnostics, multicollinearity analysis, and interactive visualization.

The Streamlit dashboard makes the statistical analysis interactive by allowing users to explore the data, perform hypothesis tests, examine regression results, evaluate model assumptions, and generate live insurance charge predictions.

The project therefore provides an end-to-end implementation of statistical methods applied to a real-world medical insurance cost dataset.

## 33. Academic Information

Course: DS602 – Statistical Methods

Lab: Lab-4 – Applied Statistical Modeling & Interactive Web Dashboard

Dataset: Medical Insurance Costs

Target Variable: `charges`

Significance Level: α = 0.05

## Author

Student: Ameya Trivedi

Student ID: 202618047