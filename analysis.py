"""
analysis.py
Statistical functions for the Medical Insurance Lab-4 dashboard.

Dataset:
    data/insurance.csv

Expected standard Medical Insurance columns:
    age, sex, bmi, children, smoker, region, charges
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


EXPECTED_COLUMNS = [
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region",
    "charges",
]

DATA_COLUMNS = {
    "numeric": ["age", "bmi", "children", "charges"],
    "categorical": ["sex", "smoker", "region"],
}


def load_data(path: str | Path = "data/insurance.csv") -> pd.DataFrame:
    """
    Load insurance.csv and validate the columns required by the assignment.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    # Make column names robust to accidental spaces/capitalization.
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
    )

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            "insurance.csv is missing these required columns: "
            f"{missing}\n\nFound columns: {df.columns.tolist()}"
        )

    # Keep the required columns in the assignment's standard order.
    df = df[EXPECTED_COLUMNS].copy()

    # Convert numeric variables safely.
    for col in ["age", "bmi", "children", "charges"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean categorical values.
    for col in ["sex", "smoker", "region"]:
        df[col] = df[col].astype("string").str.strip()

    # Remove rows that cannot be used for the dashboard.
    df = df.dropna(subset=EXPECTED_COLUMNS).reset_index(drop=True)

    if df.empty:
        raise ValueError("The dataset contains no complete usable rows.")

    return df


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Required descriptive metrics:
    mean, median, standard deviation, IQR, skewness and kurtosis.
    """
    numeric = df.select_dtypes(include=np.number)

    rows = []

    for column in numeric.columns:
        series = numeric[column].dropna()

        rows.append(
            {
                "Feature": column,
                "Mean": series.mean(),
                "Median": series.median(),
                "Std Dev": series.std(),
                "IQR": series.quantile(0.75) - series.quantile(0.25),
                "Skewness": series.skew(),
                "Kurtosis": series.kurtosis(),
            }
        )

    return pd.DataFrame(rows)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return Pearson correlation matrix for numerical variables."""
    return df[DATA_COLUMNS["numeric"]].corr()


def _get_two_groups(
    df: pd.DataFrame,
    group_variable: str,
    group_1: str,
    group_2: str,
    metric_variable: str,
):
    """
    Return the two numeric samples selected by the user.
    """
    work = df[[group_variable, metric_variable]].dropna().copy()
    work[group_variable] = work[group_variable].astype(str)

    sample_1 = work.loc[
        work[group_variable] == str(group_1),
        metric_variable,
    ].astype(float)

    sample_2 = work.loc[
        work[group_variable] == str(group_2),
        metric_variable,
    ].astype(float)

    if len(sample_1) < 2 or len(sample_2) < 2:
        raise ValueError(
            "Each selected group must contain at least two observations."
        )

    return sample_1, sample_2


def two_group_hypothesis_test(
    df: pd.DataFrame,
    group_variable: str,
    metric_variable: str,
    group_1: str,
    group_2: str,
    alpha: float = 0.05,
) -> dict:
    """
    Execute Lab-4 Hypothesis Test 1.

    1. Shapiro-Wilk for each group.
    2. Levene's test for equal variance.
    3. If both groups are normal:
         independent two-sample t-test.
       Otherwise:
         Mann-Whitney U test.
    """
    sample_1, sample_2 = _get_two_groups(
        df,
        group_variable,
        group_1,
        group_2,
        metric_variable,
    )

    # Shapiro-Wilk.
    # SciPy's Shapiro implementation can become unreliable for extremely
    # large samples; a reproducible sample of 5000 is used in that case.
    shapiro_1_sample = (
        sample_1
        if len(sample_1) <= 5000
        else sample_1.sample(5000, random_state=42)
    )

    shapiro_2_sample = (
        sample_2
        if len(sample_2) <= 5000
        else sample_2.sample(5000, random_state=42)
    )

    shapiro_1_stat, shapiro_1_p = stats.shapiro(shapiro_1_sample)
    shapiro_2_stat, shapiro_2_p = stats.shapiro(shapiro_2_sample)

    # Levene's test.
    levene_stat, levene_p = stats.levene(
        sample_1,
        sample_2,
        center="median",
    )

    both_normal = (
        shapiro_1_p >= alpha
        and shapiro_2_p >= alpha
    )

    if both_normal:
        # Levene determines whether equal variance can reasonably be assumed.
        equal_var = levene_p >= alpha

        test_stat, p_value = stats.ttest_ind(
            sample_1,
            sample_2,
            equal_var=equal_var,
        )

        test_name = (
            "Independent Two-Sample t-test "
            + ("(equal variance)" if equal_var else "(Welch)")
        )
    else:
        test_stat, p_value = stats.mannwhitneyu(
            sample_1,
            sample_2,
            alternative="two-sided",
        )

        test_name = "Mann-Whitney U test"

    return {
        "mean_1": sample_1.mean(),
        "mean_2": sample_2.mean(),
        "n_1": len(sample_1),
        "n_2": len(sample_2),
        "shapiro_1_stat": shapiro_1_stat,
        "shapiro_1_p": shapiro_1_p,
        "shapiro_2_stat": shapiro_2_stat,
        "shapiro_2_p": shapiro_2_p,
        "levene_stat": levene_stat,
        "levene_p": levene_p,
        "test_name": test_name,
        "test_stat": test_stat,
        "p_value": p_value,
        "alpha": alpha,
    }


def chi_square_test(
    df: pd.DataFrame,
    categorical_1: str,
    categorical_2: str,
    alpha: float = 0.05,
) -> dict:
    """Pearson Chi-Square test of independence."""
    work = df[[categorical_1, categorical_2]].dropna().copy()

    observed = pd.crosstab(
        work[categorical_1],
        work[categorical_2],
    )

    statistic, p_value, dof, expected = stats.chi2_contingency(
        observed
    )

    expected_df = pd.DataFrame(
        expected,
        index=observed.index,
        columns=observed.columns,
    )

    return {
        "observed": observed,
        "expected": expected_df,
        "statistic": statistic,
        "p_value": p_value,
        "dof": dof,
        "alpha": alpha,
    }


def one_way_anova(
    df: pd.DataFrame,
    group_variable: str,
    metric_variable: str,
    alpha: float = 0.05,
) -> dict:
    """One-Way ANOVA for a numerical variable across 3+ groups."""
    work = df[[group_variable, metric_variable]].dropna()

    groups = [
        group[metric_variable].astype(float).values
        for _, group in work.groupby(group_variable, sort=True)
    ]

    if len(groups) < 3:
        raise ValueError(
            "One-Way ANOVA requires at least three groups."
        )

    f_stat, p_value = stats.f_oneway(*groups)

    return {
        "f_stat": f_stat,
        "p_value": p_value,
        "group_count": len(groups),
        "alpha": alpha,
    }


def _prepare_regression_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create OLS design matrix.

    Continuous predictors:
        age, bmi, children

    Categorical predictors:
        sex, smoker, region

    One dummy category from each categorical predictor is dropped to avoid
    perfect multicollinearity with the intercept.
    """
    predictors = [
        "age",
        "bmi",
        "children",
        "sex",
        "smoker",
        "region",
    ]

    work = df[predictors + ["charges"]].dropna().copy()

    X = pd.get_dummies(
        work[predictors],
        columns=["sex", "smoker", "region"],
        drop_first=True,
        dtype=float,
    )

    X = X.astype(float)
    X = sm.add_constant(X, has_constant="add")

    y = work["charges"].astype(float)

    return X, y


def fit_ols_model(
    df: pd.DataFrame,
):
    """
    Fit multiple linear regression using statsmodels.api.OLS,
    as explicitly required by the assignment.
    """
    X, y = _prepare_regression_data(df)

    model = sm.OLS(
        y,
        X,
    ).fit()

    return model, X, y


def coefficient_table(model) -> pd.DataFrame:
    """
    Return estimated coefficients, p-values and 95% confidence intervals.
    """
    confidence_intervals = model.conf_int(alpha=0.05)

    table = pd.DataFrame(
        {
            "Coefficient": model.params,
            "P-value": model.pvalues,
            "CI Lower 95%": confidence_intervals[0],
            "CI Upper 95%": confidence_intervals[1],
        }
    )

    return table


def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate VIF for the continuous predictors required by the assignment:
        age, bmi, children

    The intercept is excluded.
    """
    continuous_predictors = [
        column
        for column in ["age", "bmi", "children"]
        if column in X.columns
    ]

    rows = []

    for column in continuous_predictors:
        other_columns = [
            c
            for c in continuous_predictors
            if c != column
        ]

        if not other_columns:
            vif_value = 1.0
        else:
            target = X[column].astype(float)
            predictors = sm.add_constant(
                X[other_columns].astype(float),
                has_constant="add",
            )

            aux_model = sm.OLS(
                target,
                predictors,
            ).fit()

            r_squared = aux_model.rsquared

            if np.isclose(r_squared, 1.0):
                vif_value = np.inf
            else:
                vif_value = 1.0 / (1.0 - r_squared)

        rows.append(
            {
                "Feature": column,
                "VIF": vif_value,
            }
        )

    return pd.DataFrame(rows)


def diagnostic_tests(model, X: pd.DataFrame) -> dict:
    """
    Gauss-Markov diagnostic tests.

    Jarque-Bera:
        residual normality check.

    Breusch-Pagan:
        heteroscedasticity check.
    """
    residuals = model.resid

    jb_result = stats.jarque_bera(residuals)
    jb_stat = float(jb_result.statistic)
    jb_p = float(jb_result.pvalue)

    bp_lm_stat, bp_lm_p, bp_f_stat, bp_f_p = het_breuschpagan(
        residuals,
        X,
    )

    return {
        "jb_stat": jb_stat,
        "jb_p": float(jb_p),
        "bp_lm_stat": float(bp_lm_stat),
        "bp_lm_p": float(bp_lm_p),
        "bp_f_stat": float(bp_f_stat),
        "bp_f_p": float(bp_f_p),
    }


def residual_figure(model) -> go.Figure:
    """Interactive residuals-vs-fitted plot."""
    fitted = model.fittedvalues
    residuals = model.resid

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fitted,
            y=residuals,
            mode="markers",
            name="Residuals",
            hovertemplate=(
                "Fitted: %{x:.2f}<br>"
                "Residual: %{y:.2f}<extra></extra>"
            ),
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        annotation_text="Residual = 0",
    )

    fig.update_layout(
        title="Residuals vs Fitted Values",
        xaxis_title="Fitted Values",
        yaxis_title="Residuals",
        height=450,
    )

    return fig


def qq_figure(model):
    """Matplotlib Q-Q plot of OLS residuals."""
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    stats.probplot(
        model.resid,
        dist="norm",
        plot=ax,
    )

    ax.set_title("Q-Q Plot of Regression Residuals")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Ordered Residuals")
    fig.tight_layout()

    return fig


def _new_prediction_row(
    model,
    age: float,
    bmi: float,
    children: int,
    sex: str,
    smoker: str,
    region: str,
) -> pd.DataFrame:
    """
    Create one prediction row with exactly the same columns as the OLS model.
    """
    new_data = pd.DataFrame(
        [
            {
                "age": age,
                "bmi": bmi,
                "children": children,
                "sex": sex,
                "smoker": smoker,
                "region": region,
            }
        ]
    )

    new_data = pd.get_dummies(
        new_data,
        columns=["sex", "smoker", "region"],
        drop_first=True,
        dtype=float,
    )

    # Match the model's columns exactly.
    new_data = new_data.reindex(
        columns=[
            column
            for column in model.model.exog_names
            if column != "const"
        ],
        fill_value=0,
    )

    new_data = new_data.astype(float)
    new_data = sm.add_constant(
        new_data,
        has_constant="add",
    )

    new_data = new_data.reindex(
        columns=model.model.exog_names,
        fill_value=0,
    )

    return new_data


def predict_with_interval(
    model,
    age: float,
    bmi: float,
    children: int,
    sex: str,
    smoker: str,
    region: str,
) -> pd.Series:
    """
    Generate a real-time prediction and a 95% prediction interval.
    """
    new_x = _new_prediction_row(
        model,
        age,
        bmi,
        children,
        sex,
        smoker,
        region,
    )

    prediction = model.get_prediction(new_x).summary_frame(
        alpha=0.05
    )

    return prediction.iloc[0]
