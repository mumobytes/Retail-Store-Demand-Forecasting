# Walmart Retail Stores Demand Forecasting with a Time-Series Model.

**A data science project exploring demand forecasting across 30,490 retail item-store combinations using time-series feature engineering, LightGBM, temporal backtesting and demand-pattern-based model selection.**

## Project Overview

Retail demand is difficult to forecast accurately because individual products behave differently. Some sell consistently, others experience strong weekly patterns and many have long periods of zero sales followed by occasional purchases.

A single forecasting model may struggle to capture these patterns equally well. This project investigates whether combining machine learning with a simple, interpretable forecasting baseline can improve demand predictions across a large and heterogeneous retail dataset.

Using the **Walmart Forecasting Accuracy dataset**, I developed a hybrid forecasting pipeline that combines LightGBM with a four-week weekday-average baseline. The system uses historical demand patterns to route relatively regular-demand series to the machine learning model while assigning intermittent-demand series to the statistical baseline. Calendar events and state-specific SNAP indicators were then incorporated to investigate whether additional contextual information could improve forecasts.

The final pipeline produces 28-day-ahead forecasts for **30,490 item-store series** structure.

The central finding is that model complexity alone does not guarantee better forecasts. A demand-pattern-aware hybrid achieved lower error than the standalone LightGBM model and simple baseline in the rolling backtest. Adding event and SNAP features delivered a further, smaller improvement.

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Data Quality and Exploratory Analysis](#data-quality-and-exploratory-analysis)
- [Forecasting Methodology](#forecasting-methodology)
- [Model Evaluation](#model-evaluation)
- [Final Forecasting Pipeline](#final-forecasting-pipeline)
- [Deliverables](#deliverables)
- [Technology Stack](#technology-stack)
- [How to Reproduce the Project](#how-to-reproduce-the-project)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Key Takeaways](#key-takeaways)
- [Author](#author)

## Problem Statement

Retailers must estimate future product demand to support inventory planning, replenishment, stock allocation and operational decision-making.

Forecasting becomes especially challenging when the task involves thousands of products across multiple stores. Demand distributions can be sparse, seasonal patterns vary by product and store, and events may influence purchasing behaviour. Forecasting each series independently can also become computationally expensive and difficult to maintain.

The objective of this project was to develop and evaluate a scalable forecasting approach that could:

1. Predict daily unit sales 28 days into the future.
2. Capture temporal patterns, including short-term lags, weekly seasonality and rolling demand behaviour.
3. Handle both relatively regular and intermittent sales series.
4. Incorporate relevant calendar and event information.
5. Compare machine learning against transparent statistical baselines using time-aware evaluation.
6. Generate forecasts for the final M5 evaluation horizon in the required submission format.

The project was focused on building a forecasting workflow in which data quality, validation strategy, feature design and model selection all contribute to the final result.

## Objectives

The project followed a structured data science workflow:

- **Data engineering:** inspect the source files, validate identifiers and dates, and prepare aligned time-series data.
- **Exploratory data analysis:** examine sparsity and weekly demand patterns.
- **Baseline modelling:** establish simple benchmarks before introducing machine learning.
- **Feature engineering:** construct lagged sales, rolling statistics and calendar features.
- **Machine learning:** train a LightGBM regression model using pooled examples from multiple item-store series.
- **Hybrid modelling:** use different forecasting approaches for different demand patterns.
- **Feature experimentation:** evaluate calendar-event and SNAP information.
- **Temporal validation:** assess performance on multiple rolling forecast windows.
- **Forecast generation:** train the selected model configuration and produce a complete 28-day projection.

## Dataset

This project uses the **Walmart Forecasting Accuracy dataset**, which contains historical Walmart retail sales from the United States.

**Source:** [M5 Forecasting Accuracy — Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)

The dataset contains daily sales, item and store metadata, calendar information, event labels, SNAP indicators and selling prices.

### Dataset Scale

| Attribute | Description |
|---|---|
| Item-store time series | 30,490 |
| Distinct products | 3,049 |
| Stores | 10 |
| Departments | 7 |
| Product categories | 3 |
| States | 3 |
| Daily sales history in the validation file | 1,913 days |
| Daily sales history in the evaluation file | 1,941 days |
| Forecast horizon | 28 days |

The data covers a period beginning on **29 January 2011**. The final evaluation forecast covers **23 May to 19 June 2016**.

The project uses the original United States retail dataset. It does not claim to model Kenyan retail demand; applying the pipeline to Kenyan grocery data would require a suitable local dataset and separate validation.

### Main Data Sources

- `sales_train_validation.csv` — historical unit sales used for model development and validation.
- `sales_train_evaluation.csv` — extended historical sales used to train the final forecasting model.
- `calendar.csv` — dates, weekdays, event information and SNAP indicators.
- `sell_prices.csv` — product-store selling prices.
- `sample_submission.csv` — required submission structure for the 28-day forecast.

The source dataset is not included in this repository. Users should obtain it from Kaggle and follow the dataset's applicable terms.

## Data Quality and Exploratory Analysis

Before modelling, the data was audited to reduce the risk of invalid records and incorrect temporal alignment.

### Data Quality Checks

The audit examined:

- Duplicate item-store identifiers.
- Duplicate calendar day identifiers and dates.
- Duplicate product-store-week price keys.
- Continuity of the daily calendar.
- Consistency of item-store series across the sales files.
- Missing and negative sales observations.
- Alignment between historical sales and forecast horizons.

The audited data contained no duplicate item-store keys, duplicate calendar identifiers or dates, duplicate price keys, missing sales values or negative sales values in the inspected sales tables. Day identifiers were consecutive and item-store series were consistent across the two sales files.

These checks established a reliable foundation for feature construction and temporal evaluation.

### Demand Sparsity

A major finding from exploratory analysis was the prevalence of zero sales.

- More than half of daily item-store observations were zero.
- The median item-store series had zero sales on approximately **73.5% of observed days**.
- Approximately **5,284 series (17.3%)** were active on no more than 10% of days.

This matters because a model optimised across all series must handle fundamentally different demand behaviours. A model that performs well for frequently purchased products may not be the best choice for products with infrequent sales.

### Weekly Seasonality

Aggregated demand showed a recurring weekly pattern, with lower sales around Wednesday and higher demand over weekends.

This motivated the use of weekday-aware baselines and made weekday information a useful candidate for the machine learning feature set.

## Forecasting Methodology

The project progressed from simple benchmarks to a hybrid model, evaluating each stage against temporal validation data.

### 1. Establishing Baseline Forecasts

Three simple forecasting strategies were evaluated.

**Last-observation baseline**

Uses the most recent observed daily sales as the forecast for each future day.

**Seasonal-naive baseline**

Repeats the sales pattern from the preceding seven days, assuming weekly seasonality continues.

**Four-week weekday-average baseline**

Calculates the average sales for each weekday over the preceding four weeks and repeats that seven-day pattern across the forecast horizon.

The four-week weekday-average approach was the strongest of these simple baselines in the initial validation experiment. It also provided an interpretable forecasting strategy for intermittent-demand series.

### 2. Time-Series Feature Engineering

A supervised learning dataset was created by converting historical sales into features and next-day targets.

The principal features included:

| Feature Group | Features | Purpose |
|---|---|---|
| Product and store metadata | Item, store, department, category and state | Learn systematic differences across retail series |
| Calendar | Day of week, month and year | Represent recurring and longer-term calendar patterns |
| Lagged demand | Lags of 1, 7, 14 and 28 days | Capture recent sales and weekly patterns |
| Rolling statistics | Rolling means over 7, 14 and 28 days | Represent recent demand levels |
| SNAP and event context | State-specific SNAP indicator, event name and event type | Test whether contextual information improves forecasts |

Lagged and rolling features were constructed from historical sales rather than future observations. Temporal cutoffs were applied during training-data construction to prevent future targets from entering the training set for a given forecast origin.

The initial supervised training table contained approximately **3.72 million training examples**. The final training table was extended to approximately **4.02 million examples** by incorporating additional historical target days available in the evaluation training file.

### 3. LightGBM Regression

LightGBM was selected as the main machine learning model because it efficiently handles large tabular datasets, nonlinear relationships and interactions between numerical and categorical features.

The model learned from pooled examples across item-store series instead of requiring a separate model for every product and store.

The main modelling configuration included:

- 800 boosting rounds as the initial maximum.
- Learning rate of 0.05.
- 31 leaves per tree.
- Row and feature subsampling.
- L2 regularisation.
- Early stopping during backtesting.
- A fixed random seed for reproducibility.

The final forecasting configuration used **271 boosting rounds**, based on the median best iteration from the rolling backtest experiments.

### 4. Recursive Multi-Step Forecasting

The model predicts one day ahead at a time. To generate a 28-day forecast, each prediction is added to the history used to construct the next day's lag and rolling features.

This recursive strategy allows a one-step model to generate a full multi-day forecast while keeping the same feature definitions used during training.

Because future actual sales are unavailable during the forecast horizon, later predictions depend partly on earlier model predictions. Forecast errors can therefore propagate across the horizon.

Predicted sales were constrained to be non-negative.

### 5. Hybrid Model Selection

The project tested a hybrid strategy based on the observed activity of each series.

For each item-store series, the active share was calculated as:

\[
\text{Active Share}_i =
\frac{\sum_{t=1}^{T}\mathbb{1}(y_{i,t}>0)}{T}
\]

where \(y_{i,t}\) represents the observed sales for series \(i\) on day \(t\), and \(\mathbb{1}(\cdot)\) is an indicator equal to one when sales are positive.

The routing rule was:

- **Active share greater than 0.30:** use LightGBM.
- **Active share less than or equal to 0.30:** use the four-week weekday-average baseline.

The 30% threshold was selected during exploratory backtesting. It should therefore be interpreted as a promising heuristic rather than a universally optimal threshold.

This approach reflects the project's main modelling hypothesis: forecasting methods should be selected with the characteristics of the demand series in mind rather than assuming a single model is equally suitable for every product.

### 6. SNAP and Calendar-Event Experiment

The hybrid model was extended with two additional types of information:

- State-specific SNAP indicators.
- Calendar event names and event types.

The objective was to test whether these variables captured additional variation in purchasing behaviour beyond historical sales and basic calendar features.

The extended model produced a small improvement over the original hybrid. The result supports retaining the contextual features in the final configuration, while recognising that their incremental contribution was modest.

### 7. Selling-Price Experiment

Selling-price information and derived price-change indicators were also investigated.

The price-augmented model did not produce a meaningful improvement over the original hybrid during the tested backtests. Price features were therefore not retained in the final model.

This experiment helped keep the final approach focused on features that demonstrated measurable value rather than increasing model complexity without a corresponding improvement.

## Model Evaluation

Forecasting performance was assessed using mean absolute error (MAE), root mean squared error (RMSE), weighted absolute percentage error (WAPE) and forecast bias.

- **MAE:** average absolute difference between actual and predicted units.
- **RMSE:** penalises larger individual errors more heavily than MAE.
- **WAPE:** total absolute error divided by total actual sales, expressed as a percentage.
- **Bias:** average signed prediction error, defined here as predicted sales minus actual sales. Positive values indicate overforecasting; negative values indicate underforecasting.

Lower MAE, RMSE and WAPE indicate better performance. Bias closer to zero indicates less systematic overforecasting or underforecasting.

### Rolling Backtest Results

Three rolling forecast origins were used to compare models over 28-day horizons. Training examples for each origin were restricted to target days strictly earlier than the validation window.

| Model | MAE ↓ | RMSE ↓ | WAPE ↓ |
|---|---:|---:|---:|
| Four-week weekday baseline | 1.0755 | 2.3309 | 77.85% |
| LightGBM | 1.0766 | 2.1722 | 77.93% |
| Original hybrid | 1.0385 | 2.1977 | 75.18% |
| **SNAP/event hybrid** | **1.0370** | **2.1867** | **75.07%** |

*Results are from the pooled rolling backtest. Values are rounded.*

The results reveal several important points:

1. **LightGBM alone was not uniformly superior.** It achieved lower RMSE than the simple baseline but slightly higher MAE and WAPE.
2. **The hybrid reduced aggregate forecasting error.** Its MAE was approximately 3.5% lower than that of the four-week weekday baseline.
3. **SNAP and event features added a small improvement.** The extended hybrid reduced MAE from 1.0385 to 1.0370 and WAPE from approximately 75.18% to 75.07%.
4. **The improvement was incremental rather than dramatic.** The experiment supports the hybrid design but does not establish that the contextual features will improve forecasts in every future period.

### Leakage-Free Validation

After selecting the final feature configuration and forecasting strategy, a validation model was trained using historical targets through day `d_1913`. It then forecast the following 28 days, `d_1914` to `d_1941` (25 April to 22 May 2016).

| Metric | Validation Result |
|---|---:|
| MAE | 1.0657 |
| RMSE | 2.1635 |
| WAPE | 73.86% |
| Bias | -0.0222 |

The small negative bias indicates slight average underforecasting under the project's bias convention.

These metrics are reported separately from the pooled rolling backtest because they come from a different forecast period and evaluation setup. They should not be interpreted as directly interchangeable measurements.

### Interpretation

The results suggest that the strongest contribution was not simply adding a more complex model. It was combining a global machine learning model with a weekday-aware baseline according to the observed demand pattern.

The additional SNAP and event variables improved the rolling-backtest results slightly. Further evaluation would be required to establish whether that gain is stable across different retail periods and forecasting settings.

## Final Forecasting Pipeline

The final pipeline was trained using the extended historical data available through `d_1941`.

It generated forecasts for the next 28 days, from **23 May to 19 June 2016**, for all 30,490 item-store series.

The final output is:

`m5_hybrid_snap_event_submission.csv`

The file contains **60,980 rows and 29 columns**:

- 30,490 rows for the validation submission.
- 30,490 rows for the evaluation submission.
- One `id` column and 28 forecast columns (`F1`–`F28`).

The output was checked against the original M5 sample submission to verify that its dimensions, column order, row identifiers and forecast values matched the required structure.

### Important Distinction

The validation forecast can be evaluated against known sales for `d_1914`–`d_1941`.

The final evaluation forecast for `d_1942`–`d_1969` is a genuine forward forecast from the available training history. The actual sales for that evaluation horizon are not provided in the supplied M5 evaluation data. Consequently, the final 28-day predictions have been generated and structurally validated, but their accuracy on that horizon cannot be calculated from the available labels.

## Deliverables

The project produces the following outputs:

| Deliverable | Description |
|---|---|
| Data audit and exploratory analysis | Checks of data integrity, sparsity and weekly patterns |
| Baseline forecasts | Reference forecasts for comparison |
| LightGBM model | Global machine learning model for daily demand |
| Hybrid forecasting pipeline | Demand-pattern-based routing between LightGBM and the weekday baseline |
| SNAP/event experiment | Evaluation of additional calendar context |
| Rolling backtest comparison | Model evaluation across multiple temporal windows |
| Validation forecast visualisations | Examples comparing actual and predicted demand |
| Aggregate final forecast visualisation | Total forecast units across all series by calendar date |
| Final submission CSV | Complete 28-day forecast in M5 submission format |

The generated figures include:

- `rolling_backtest_mae.png`
- `validation_forecast_examples.png`
- `final_28_day_forecast.png`

## Technology Stack

| Technology | Role |
|---|---|
| Python | Core programming language |
| Pandas | Data preparation, joins and time-series manipulation |
| NumPy | Numerical computation and forecast arrays |
| LightGBM | Gradient-boosted tree regression |
| Scikit-learn | Evaluation utilities and supporting machine learning tools |
| Matplotlib | Forecast and model-comparison visualisations |
| Google Colab | Development and model training environment |

## How to Reproduce the Project

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Replace the repository URL with the actual GitHub address after creating the repository.

### 2. Create an Environment

Python 3.11 is a reasonable starting point for reproducing the workflow. Use the package versions recorded in the project environment where available.

```bash
python -m venv .venv
```

Activate the environment.

**Windows**

```bat
.venv\Scripts\activate
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install numpy pandas matplotlib scikit-learn lightgbm jupyter
```

### 3. Obtain the Dataset

Download the M5 Forecasting Accuracy data from [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data).

Place the required CSV files in the data directory expected by the notebook. The raw dataset is excluded from the repository to avoid committing large source files and to keep the project reproducible from the original source.

### 4. Run the Notebook

Open the main project notebook in Jupyter or Google Colab and execute the workflow in order:

1. Load and audit the data.
2. Perform exploratory analysis.
3. Build and evaluate the baseline forecasts.
4. Construct lag, rolling and calendar features.
5. Train and evaluate LightGBM.
6. Run the rolling backtests and hybrid experiments.
7. Train the final model and generate the forecasts.
8. Create the visualisations and submission CSV.

**Reproducibility note:** update any dataset paths in the notebook to match your local environment. Exact dependency versions and notebook filenames should be documented in the repository if they are available.

## Limitations and Future Improvements

This project demonstrates a complete forecasting workflow, but several limitations should be acknowledged.

### 1. The Routing Threshold Was Selected During Exploratory Backtesting

The 30% active-share threshold was evaluated using the same rolling windows that informed model selection. Its performance is therefore exploratory rather than an unbiased estimate from a completely untouched test period.

A stronger next step would be to tune the threshold on dedicated development windows and evaluate it once on a separate, later holdout period.

### 2. Contextual Features Produced Only a Modest Improvement

SNAP and event features improved the pooled rolling-backtest metrics slightly. Further testing is needed to determine whether this gain persists across different forecast origins.

### 3. Recursive Forecasts Can Accumulate Errors

Predictions are fed back into subsequent feature calculations. Errors from early forecast days can affect later predictions, particularly for series with volatile demand.

### 4. Zero Sales Do Not Always Mean Zero Demand

A recorded zero may reflect a lack of purchases, but it can also be associated with product availability or stockouts. The supplied data does not establish the cause of every zero observation.

### 5. The Final Evaluation Horizon Has No Observed Outcomes in the Supplied Data

The final forecasts are structurally validated, but the available evaluation file does not contain the actual sales needed to calculate final-horizon errors.

### 6. Generalisation to Other Retail Markets Remains Untested

The model was developed on historical United States Walmart sales. Its performance cannot be assumed to transfer directly to Kenyan or other retail markets without appropriate local data, adaptation and validation.

### Potential Future Improvements

- Evaluate the hybrid against additional intermittent-demand forecasting methods.
- Tune routing thresholds on dedicated development periods.
- Explore direct multi-horizon forecasting to reduce recursive error propagation.
- Investigate stock availability and stockout indicators where reliable data exists.
- Assess the value of price and promotion information with more targeted experiments.
- Evaluate forecast accuracy by department, category, store and demand-frequency group.
- Test the workflow on a separate retail dataset to measure cross-market generalisation.

## Key Takeaways

This project demonstrates a practical approach to applied time-series machine learning:

- **Data quality comes first.** Reliable identifiers, dates and temporal cutoffs are essential for trustworthy forecasts.
- **Baselines are valuable.** A simple weekday-aware method established a strong benchmark that the machine learning model needed to justify exceeding.
- **Model choice should reflect data behaviour.** A hybrid approach performed better in the rolling backtest than either standalone method on its own.
- **Feature engineering should be evidence-driven.** SNAP and event features improved the results slightly, while the tested price features did not justify their additional complexity.
- **Temporal validation matters.** Multiple forecast origins provide a more informative assessment than a random train-test split for forecasting problems.
- **A working model is not the entire deliverable.** The project includes data checks, evaluation, visualisations and a submission file formatted for the competition task.

The broader lesson is that effective forecasting depends on matching model assumptions to the structure of the data, evaluating models under realistic temporal constraints and being transparent about what the results do and do not establish.

---

## Author

**Brian Mumo Mutuku**

Data science, machine learning and quantitative modelling.

- GitHub: [github.com/mumobytes](https://github.com/mumobytes)
- LinkedIn: [linkedin.com/in/mumo-mutuku](https://www.linkedin.com/in/mumo-mutuku/)

*This project was developed as a practical investigation into retail demand forecasting, with emphasis on data quality, temporal evaluation, feature engineering and demand-aware model selection.*
