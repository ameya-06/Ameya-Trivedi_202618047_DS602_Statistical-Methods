"""
Medical Insurance Costs - Lab 4 Dashboard
Course: Statistical Modeling with Python
M.Sc. Data Science - Semester 1

Run:
    streamlit run app.py

Project structure:
    Medical-Insurance-Dashboard/
    ├── app.py
    ├── analysis.py
    ├── requirements.txt
    └── data/
        └── insurance.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import (
    DATA_COLUMNS,
    load_data,
    descriptive_statistics,
    correlation_matrix,
    two_group_hypothesis_test,
    chi_square_test,
    one_way_anova,
    fit_ols_model,
    coefficient_table,
    calculate_vif,
    diagnostic_tests,
    predict_with_interval,
    residual_figure,
    qq_figure,
)


# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="Medical Insurance | Statistical Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .sub-title {
        color: #6b7280;
        margin-top: 0.2rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">📊 Medical Insurance Statistical Dashboard</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Lab-4: Applied Statistical Modeling & Interactive Web Dashboard</div>',
    unsafe_allow_html=True,
)

# ============================================================
# LOAD DATA
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "insurance.csv"

try:
    df = load_data(DATA_PATH)
except Exception as exc:
    st.error("Unable to load data/insurance.csv")
    st.code(str(exc))
    st.info(
        "Place your insurance.csv file inside the project's data folder:\n\n"
        "Medical-Insurance-Dashboard/data/insurance.csv"
    )
    st.stop()


# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.header("🔎 Data Filters")

age_min = int(df["age"].min())
age_max = int(df["age"].max())

age_range = st.sidebar.slider(
    "Age range",
    min_value=age_min,
    max_value=age_max,
    value=(age_min, age_max),
)

sex_values = sorted(df["sex"].dropna().unique().tolist())
smoker_values = sorted(df["smoker"].dropna().unique().tolist())
region_values = sorted(df["region"].dropna().unique().tolist())

selected_sex = st.sidebar.multiselect(
    "Sex",
    sex_values,
    default=sex_values,
)

selected_smoker = st.sidebar.multiselect(
    "Smoker",
    smoker_values,
    default=smoker_values,
)

selected_region = st.sidebar.multiselect(
    "Region",
    region_values,
    default=region_values,
)

filtered = df[
    df["age"].between(age_range[0], age_range[1])
    & df["sex"].isin(selected_sex)
    & df["smoker"].isin(selected_smoker)
    & df["region"].isin(selected_region)
].copy()

st.sidebar.markdown("---")
st.sidebar.write(f"**Filtered records:** {len(filtered):,}")
st.sidebar.write(f"**Total records:** {len(df):,}")

if filtered.empty:
    st.warning("No records match the selected filters. Please change the filters.")
    st.stop()


# ============================================================
# THREE REQUIRED TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(
    [
        "1️⃣ Data Exploration",
        "2️⃣ Hypothesis Testing Lab",
        "3️⃣ Live Prediction & Diagnostics",
    ]
)


# ============================================================
# TAB 1 - DATA EXPLORATION
# ============================================================
with tab1:
    st.header("Data Exploration")

    # Dataset summary
    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Records", f"{len(filtered):,}")
    k2.metric("Mean Charges", f"${filtered['charges'].mean():,.2f}")
    k3.metric("Median Charges", f"${filtered['charges'].median():,.2f}")
    k4.metric("Mean BMI", f"{filtered['bmi'].mean():.2f}")

    st.subheader("Dataset Preview")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    # Descriptive metrics required by assignment
    st.subheader("Descriptive Statistics")

    desc = descriptive_statistics(filtered)
    st.dataframe(
        desc.style.format(
            {
                "Mean": "{:.3f}",
                "Median": "{:.3f}",
                "Std Dev": "{:.3f}",
                "IQR": "{:.3f}",
                "Skewness": "{:.3f}",
                "Kurtosis": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Distribution plots
    st.subheader("Distribution Plot")

    distribution_variable = st.selectbox(
        "Select numerical feature",
        DATA_COLUMNS["numeric"],
        index=DATA_COLUMNS["numeric"].index("charges"),
    )

    hist_fig = px.histogram(
        filtered,
        x=distribution_variable,
        marginal="box",
        nbins=30,
        title=f"Distribution of {distribution_variable}",
    )
    hist_fig.update_layout(height=450)
    st.plotly_chart(hist_fig, use_container_width=True)

    # Bivariate scatter
    st.subheader("Bivariate Scatter Plot")

    scatter_x = st.selectbox(
        "X-axis",
        [c for c in DATA_COLUMNS["numeric"] if c != "charges"],
        index=0,
    )
    scatter_y = st.selectbox(
        "Y-axis",
        DATA_COLUMNS["numeric"],
        index=DATA_COLUMNS["numeric"].index("charges"),
    )

    scatter_fig = px.scatter(
        filtered,
        x=scatter_x,
        y=scatter_y,
        color="smoker",
        hover_data=["age", "bmi", "children", "charges"],
        title=f"{scatter_y} vs {scatter_x}",
    )
    scatter_fig.update_layout(height=500)
    st.plotly_chart(scatter_fig, use_container_width=True)

    # Correlation matrix
    st.subheader("Correlation Matrix")

    corr = correlation_matrix(filtered)
    corr_fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Correlation Matrix of Numerical Features",
    )
    corr_fig.update_layout(height=500)
    st.plotly_chart(corr_fig, use_container_width=True)


# ============================================================
# TAB 2 - HYPOTHESIS TESTING
# ============================================================
with tab2:
    st.header("Hypothesis Testing Lab")

    st.markdown(
        """
        **Significance level:** α = 0.05

        **Decision rule:** p-value < 0.05 → Reject H0; otherwise → Fail to Reject H0.
        """
    )

    # --------------------------------------------------------
    # TEST 1: TWO GROUP COMPARISON
    # --------------------------------------------------------
    st.subheader("Hypothesis Test 1 — Compare Two Groups")

    st.markdown(
        """
        **H0:** There is no significant difference in the selected continuous
        metric between the two selected groups.

        **H1:** There is a significant difference in the selected continuous
        metric between the two selected groups.

        The dashboard first checks **Shapiro-Wilk normality** and **Levene's
        equal-variance assumption**. If both groups are normal, an independent
        two-sample t-test is used; otherwise, the **Mann-Whitney U test** is used.
        """
    )

    ht1_col1, ht1_col2 = st.columns(2)

    with ht1_col1:
        group_variable = st.selectbox(
            "Categorical grouping variable",
            DATA_COLUMNS["categorical"],
            index=DATA_COLUMNS["categorical"].index("smoker"),
            key="ht1_group_variable",
        )

    with ht1_col2:
        metric_variable = st.selectbox(
            "Continuous metric",
            DATA_COLUMNS["numeric"],
            index=DATA_COLUMNS["numeric"].index("charges"),
            key="ht1_metric_variable",
        )

    available_groups = sorted(
        filtered[group_variable].dropna().astype(str).unique().tolist()
    )

    if len(available_groups) < 2:
        st.warning("The selected grouping variable does not contain at least two groups.")
    else:
        default_two = available_groups[:2]

        selected_groups = st.multiselect(
            "Select exactly TWO groups",
            available_groups,
            default=default_two,
            max_selections=2,
            key="ht1_groups",
        )

        if len(selected_groups) != 2:
            st.warning("Please select exactly two groups.")
        else:
            result = two_group_hypothesis_test(
                filtered,
                group_variable,
                metric_variable,
                selected_groups[0],
                selected_groups[1],
                alpha=0.05,
            )

            c1, c2, c3 = st.columns(3)
            c1.metric(
                f"Mean — {selected_groups[0]}",
                f"{result['mean_1']:,.3f}",
            )
            c2.metric(
                f"Mean — {selected_groups[1]}",
                f"{result['mean_2']:,.3f}",
            )
            c3.metric(
                "Final p-value",
                f"{result['p_value']:.6g}",
            )

            st.markdown("#### Assumption Checks")

            assumption_table = pd.DataFrame(
                [
                    {
                        "Test": "Shapiro-Wilk",
                        "Group": str(selected_groups[0]),
                        "Statistic": result["shapiro_1_stat"],
                        "p-value": result["shapiro_1_p"],
                        "Decision": (
                            "Normal"
                            if result["shapiro_1_p"] >= 0.05
                            else "Not normal"
                        ),
                    },
                    {
                        "Test": "Shapiro-Wilk",
                        "Group": str(selected_groups[1]),
                        "Statistic": result["shapiro_2_stat"],
                        "p-value": result["shapiro_2_p"],
                        "Decision": (
                            "Normal"
                            if result["shapiro_2_p"] >= 0.05
                            else "Not normal"
                        ),
                    },
                    {
                        "Test": "Levene",
                        "Group": "Both groups",
                        "Statistic": result["levene_stat"],
                        "p-value": result["levene_p"],
                        "Decision": (
                            "Equal variance"
                            if result["levene_p"] >= 0.05
                            else "Unequal variance"
                        ),
                    },
                ]
            )

            st.dataframe(
                assumption_table.style.format(
                    {"Statistic": "{:.6f}", "p-value": "{:.6g}"}
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Final Test")

            st.write(f"**Test used:** {result['test_name']}")
            st.write(f"**Test statistic:** {result['test_stat']:.6f}")
            st.write(f"**p-value:** {result['p_value']:.6g}")

            if result["p_value"] < 0.05:
                st.error(
                    "Conclusion: Reject H0 at α = 0.05. "
                    "There is statistically significant evidence of a difference "
                    "between the two groups."
                )
            else:
                st.success(
                    "Conclusion: Fail to Reject H0 at α = 0.05. "
                    "There is not enough statistical evidence of a difference "
                    "between the two groups."
                )

            box_fig = px.box(
                filtered[filtered[group_variable].astype(str).isin(selected_groups)],
                x=group_variable,
                y=metric_variable,
                color=group_variable,
                points="outliers",
                title=f"{metric_variable} across selected {group_variable} groups",
            )
            st.plotly_chart(box_fig, use_container_width=True)

    st.markdown("---")

    # --------------------------------------------------------
    # TEST 2: CHI-SQUARE OR ONE-WAY ANOVA
    # --------------------------------------------------------
    st.subheader("Hypothesis Test 2")

    test2_choice = st.radio(
        "Choose one hypothesis test",
        ["Chi-Square Test", "One-Way ANOVA"],
        horizontal=True,
    )

    if test2_choice == "Chi-Square Test":
        st.markdown(
            """
            **H0:** The two categorical variables are independent.

            **H1:** The two categorical variables are associated.
            """
        )

        c1, c2 = st.columns(2)

        with c1:
            cat1 = st.selectbox(
                "First categorical variable",
                DATA_COLUMNS["categorical"],
                key="chi_cat1",
            )

        with c2:
            other_cats = [c for c in DATA_COLUMNS["categorical"] if c != cat1]
            cat2 = st.selectbox(
                "Second categorical variable",
                other_cats,
                key="chi_cat2",
            )

        chi_result = chi_square_test(filtered, cat1, cat2, alpha=0.05)

        c1, c2, c3 = st.columns(3)
        c1.metric("Chi-Square", f"{chi_result['statistic']:.6f}")
        c2.metric("Degrees of Freedom", str(chi_result["dof"]))
        c3.metric("p-value", f"{chi_result['p_value']:.6g}")

        st.markdown("#### Contingency Table")
        st.dataframe(
            chi_result["observed"],
            use_container_width=True,
        )

        if chi_result["p_value"] < 0.05:
            st.error(
                "Conclusion: Reject H0 at α = 0.05. "
                "The categorical variables are statistically associated."
            )
        else:
            st.success(
                "Conclusion: Fail to Reject H0 at α = 0.05. "
                "There is not enough evidence of an association."
            )

    else:
        st.markdown(
            """
            **H0:** The mean numerical value is the same across all groups.

            **H1:** At least one group mean is different.
            """
        )

        c1, c2 = st.columns(2)

        with c1:
            anova_group = st.selectbox(
                "Grouping variable",
                DATA_COLUMNS["categorical"],
                index=DATA_COLUMNS["categorical"].index("region"),
                key="anova_group",
            )

        with c2:
            anova_metric = st.selectbox(
                "Numerical metric",
                DATA_COLUMNS["numeric"],
                index=DATA_COLUMNS["numeric"].index("charges"),
                key="anova_metric",
            )

        group_count = filtered[anova_group].nunique()

        if group_count < 3:
            st.warning(
                "One-Way ANOVA requires at least three groups. "
                "Try region for the standard insurance dataset."
            )
        else:
            anova_result = one_way_anova(
                filtered,
                anova_group,
                anova_metric,
                alpha=0.05,
            )

            c1, c2 = st.columns(2)
            c1.metric("F-statistic", f"{anova_result['f_stat']:.6f}")
            c2.metric("p-value", f"{anova_result['p_value']:.6g}")

            if anova_result["p_value"] < 0.05:
                st.error(
                    "Conclusion: Reject H0 at α = 0.05. "
                    "At least one group mean is statistically different."
                )
            else:
                st.success(
                    "Conclusion: Fail to Reject H0 at α = 0.05. "
                    "There is not enough evidence that the group means differ."
                )

            anova_fig = px.box(
                filtered,
                x=anova_group,
                y=anova_metric,
                color=anova_group,
                title=f"{anova_metric} across {anova_group} groups",
            )
            st.plotly_chart(anova_fig, use_container_width=True)


# ============================================================
# TAB 3 - LIVE PREDICTION & DIAGNOSTICS
# ============================================================
with tab3:
    st.header("Live Prediction & Diagnostics")

    try:
        model, X, y = fit_ols_model(df)

        # ----------------------------------------------------
        # MODEL FORMULATION
        # ----------------------------------------------------
        st.subheader("Multiple Linear Regression — OLS")

        st.markdown(
            """
            The dependent variable is **medical charges**. The model uses
            age, BMI, number of children, sex, smoking status, and region.
            Categorical variables are represented using dummy variables.
            """
        )

        st.code(
            "Y = β0 + β1(age) + β2(bmi) + β3(children) + "
            "β4(sex) + β5(smoker) + β6(region) + ε",
            language="text",
        )

        # Model fit
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R²", f"{model.rsquared:.4f}")
        m2.metric("Adjusted R²", f"{model.rsquared_adj:.4f}")
        m3.metric("Observations", f"{int(model.nobs):,}")
        m4.metric("Overall model p-value", f"{model.f_pvalue:.6g}")

        # Coefficients
        st.subheader("Parameter Estimates")

        coef = coefficient_table(model)
        st.dataframe(
            coef.style.format(
                {
                    "Coefficient": "{:.6f}",
                    "P-value": "{:.6g}",
                    "CI Lower 95%": "{:.6f}",
                    "CI Upper 95%": "{:.6f}",
                }
            ),
            use_container_width=True,
        )

        with st.expander("View full statsmodels OLS summary"):
            st.text(model.summary().as_text())

        # ----------------------------------------------------
        # LIVE PREDICTION
        # ----------------------------------------------------
        st.subheader("🔮 Live Medical Charges Prediction")

        left, right = st.columns(2)

        with left:
            pred_age = st.slider(
                "Age",
                min_value=int(df["age"].min()),
                max_value=int(df["age"].max()),
                value=int(df["age"].median()),
                key="pred_age",
            )

            pred_bmi = st.number_input(
                "BMI",
                min_value=0.0,
                max_value=100.0,
                value=float(df["bmi"].median()),
                step=0.1,
                key="pred_bmi",
            )

            pred_children = st.number_input(
                "Children",
                min_value=0,
                max_value=int(df["children"].max()),
                value=int(df["children"].median()),
                step=1,
                key="pred_children",
            )

        with right:
            pred_sex = st.selectbox(
                "Sex",
                sex_values,
                key="pred_sex",
            )

            pred_smoker = st.selectbox(
                "Smoker",
                smoker_values,
                key="pred_smoker",
            )

            pred_region = st.selectbox(
                "Region",
                region_values,
                key="pred_region",
            )

        prediction = predict_with_interval(
            model,
            age=pred_age,
            bmi=pred_bmi,
            children=pred_children,
            sex=pred_sex,
            smoker=pred_smoker,
            region=pred_region,
        )

        p1, p2, p3 = st.columns(3)
        p1.metric(
            "Predicted Charges",
            f"${prediction['mean']:,.2f}",
        )
        p2.metric(
            "95% Prediction Interval — Lower",
            f"${prediction['obs_ci_lower']:,.2f}",
        )
        p3.metric(
            "95% Prediction Interval — Upper",
            f"${prediction['obs_ci_upper']:,.2f}",
        )

        st.caption(
            "The displayed interval is a 95% prediction interval for an "
            "individual future observation."
        )

        # ----------------------------------------------------
        # GAUSS-MARKOV DIAGNOSTICS
        # ----------------------------------------------------
        st.subheader("Gauss-Markov Diagnostic Checks")

        diagnostics = diagnostic_tests(model, X)

        d1, d2 = st.columns(2)

        with d1:
            st.plotly_chart(
                residual_figure(model),
                use_container_width=True,
            )

        with d2:
            st.pyplot(
                qq_figure(model),
                use_container_width=True,
            )

        st.markdown("#### Diagnostic Test Results")

        diag_df = pd.DataFrame(
            [
                {
                    "Diagnostic": "Jarque-Bera",
                    "Statistic": diagnostics["jb_stat"],
                    "p-value": diagnostics["jb_p"],
                    "Interpretation": (
                        "Residual normality not rejected"
                        if diagnostics["jb_p"] >= 0.05
                        else "Evidence against residual normality"
                    ),
                },
                {
                    "Diagnostic": "Breusch-Pagan LM",
                    "Statistic": diagnostics["bp_lm_stat"],
                    "p-value": diagnostics["bp_lm_p"],
                    "Interpretation": (
                        "No significant heteroscedasticity evidence"
                        if diagnostics["bp_lm_p"] >= 0.05
                        else "Evidence of heteroscedasticity"
                    ),
                },
            ]
        )

        st.dataframe(
            diag_df.style.format(
                {
                    "Statistic": "{:.6f}",
                    "p-value": "{:.6g}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # VIF
        # ----------------------------------------------------
        st.subheader("Multicollinearity — Variance Inflation Factor")

        vif = calculate_vif(X)
        st.dataframe(
            vif.style.format({"VIF": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "VIF is reported for the continuous predictors as required by "
            "the assignment. Values around 5+ are commonly treated as a "
            "warning sign; interpret them in context."
        )

    except Exception as exc:
        st.error("The regression/diagnostics section could not be completed.")
        st.exception(exc)
