# Experimental Setup Guide (README)

## 1. Hardware Environment
The evaluation experiments were systematically executed on a dedicated Windows 11 workstation outfitted with an Intel/AMD CPU (specified as XXX CPU in the manuscript) and 16GB of system RAM to ensure a standardized and stable computational environment.

## 2. Datasets Overview
To comprehensively evaluate the performance and generalization capabilities of the proposed DownsClean framework under a variety of data characteristics, the experimental suite incorporates 10 widely utilized, real-world multivariate time series datasets spanning multiple industrial and economic domains.

| Dataset | Rows | Columns | Description |
| :--- | :--- | :--- | :--- |
| **IDF** | 100K | 63 | Detailed temperature variation logs of lubricating oil across various components within a wind turbine system. |
| **PSM** | 132,481 | 25 | Server Performance Monitoring data, capturing internal multi-dimensional state metrics of large-scale computational nodes. |
| **SWaT** | 14,996 | 26 | Secure Water Treatment dataset, representing industrial control signals and operational state variables for anomaly detection. |
| **Stock** | 12,824 | 1 | A continuous financial series capturing stock market variations and business continuous trends. |
| **ETT** | 7,080 | 8 | Electricity Transformer Temperature dataset, a standard benchmark tracking oil temperature and load parameters. |
| **TOTALSA** | 593 | 1 | Total Vehicle Sales in the US market, reflecting seasonal adjustments and macro-economic trends across economic cycles. |
| **Exchange** | 7,588 | 8 | International financial foreign exchange data tracking multi-country currency exchange rates relative to the USD. |
| **Rice** | 3,808 | 6 | Morphological and structural characteristic records of rice populations for classification and trend modeling. |
| **Abalone** | 4,177 | 8 | Physical measurement profiles (length, diameter, weight, etc.) used to model and predict the age configurations of abalone. |

## 3. Evaluation Metrics
In order to eliminate any evaluation bias resulting from a single mathematical formulation, the framework is rigorously benchmarked across five complementary metrics covering squared errors, absolute variations, and percentage scaling limits:

* **Mean Squared Error (MSE)**: Quantifies the average squared discrepancy between predicted and true values. It assigns a higher penalty to large deviations. A perfect model yields an MSE of 0.
  $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

* **Root Mean Squared Error (RMSE)**: The square root of MSE, mapping the error metric back to the original scale of the target variable for intuitive physical interpretation.
  $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

* **Root Mean Squared Logarithmic Error (RMSLE)**: Evaluates the error on a logarithmic scale, which makes it highly robust to outliers and more sensitive to relative scale variations rather than absolute values.
  $$RMSLE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\log(1 + y_i) - \log(1 + \hat{y}_i))^2}$$

* **Mean Absolute Error (MAE)**: Computes the average absolute differences, providing a linear reflection of the model's uniform prediction accuracy without compounding extreme outliers.
  $$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

* **Mean Absolute Percentage Error (MAPE)**: Measures the percentage relative deviation. A value of 0% indicates an absolute flawless fit, whereas values exceeding 100% flag highly degenerate predictions.
  $$MAPE = \frac{1}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{\max(y_i, \hat{y}_i)}$$

## 4. Baseline Data Cleaning Algorithms
The proposed DownsClean framework is evaluated against six state-of-the-art baselines representing a diverse spectrum of cleaning paradigms:
* **Smooth (Smoothing Algorithm)**: Employs a sliding window mechanism to replace raw data points with rolling localized statistical indicators, filtering high-frequency noise while smoothing out localized spikes.
* **Speed (Velocity Constraint Cleaning)**: Leverages domain-specific physical velocity bounds to rapidly identify and isolate transient outliers triggered by signal dropout or transmission corruption.
* **MTSClean (Multivariate Time Series Cleaner)**: A sophisticated method capturing structural cross-correlations across both rows and columns via a sequential violation quantification, segment localization, and bidirectional repair architecture.
* **Kalman Filter**: Formulates the time series sequence as a continuous dynamic state-space process, executing sequential prediction and correction updates to filter out Gaussian state perturbations.
* **SHoTClean**: Incorporates a joint soft-and-hard constraint network solved via dynamic programming operators to eliminate structural anomalies while recovering underlying multivariate distributions.
* **Akane**: Guided by the principle of perplexity minimization, it uses local optimization with pruning to replace dirty data and accurately recover time series patterns under strict resource budgets.

## 5. Downstream Predictive Models
To evaluate the downstream utility and adaptability of the cleaned sequence, the optimized outputs are directly integrated into five contemporary forecasting architectures:
* **LightGBM**: A highly efficient, distributed gradient boosting framework structured on optimal decision tree growth patterns for tabular and tabularized sequential forecasting.
* **ARIMA**: The classical autoregressive integrated moving average model rooted in mathematical stationarity principles to precisely track historical linear trends and seasonal shifts.
* **Decision Tree**: An analytical machine learning model utilizing recursive feature splitting nodes to map complex non-linear boundaries with fully interpretable branch pathways.
* **DLinear**: A lightweight neural forecasting architecture utilizing moving average trend-seasonal decomposition layers linked to direct linear mappings, yielding immense computational speed and robust modeling properties.
* **Chronos**: A cutting-edge 120M-parameter language-model-based time series foundation architecture that treats sequential values as tokens, utilizing group attention mechanics for zero-shot and context-aware multi-step forecasting.
