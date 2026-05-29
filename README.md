# Dengue Case Forecasting for Colombo District (2022)



## Project Overview





This project focuses on forecasting dengue cases in the **Colombo District, Sri Lanka**, for the next **6 months (January 2022 – June 2022)** using **time series analysis techniques**. The study uses historical dengue case data from **2019, 2020, and 2021** to build a predictive model and support **data-driven public health planning**.

In addition to the forecasting model, Python-based risk categorization was used to classify provinces into High, Medium, and Low dengue risk levels. An **interactive Streamlit dashboard** was developed to visualize historical dengue trends, model outputs,risk levels and forecasted cases.

\---

## Objectives

* Analyze dengue case trends in Colombo District
* Check and transform the time series for stationarity
* Build and compare multiple ARIMA models
* Select the best model using Akaike Information Criterion (AIC)
* Forecast dengue cases for **January 2022 to June 2022**
* Create an interactive Streamlit dashboard for visualization

\---

## Dataset

The dataset contains **monthly dengue case counts** in the Colombo District for:

* **2019**
* **2020**
* **2021**

The data were used to forecast future dengue cases for the first half of **2022**.

\---



## Methodology



### 1\. Data Preprocessing

The time series was first examined for **stationarity**, which is an important assumption in ARIMA modeling.

* The original series was found to be **non-stationary**
* **First differencing** was applied to stabilize the mean
* Since the series still remained non-stationary, **second differencing** was performed
* After second differencing, the series became **stationary with no visible trend**



### 2\. Model Identification



To determine suitable model parameters:

* **Autocorrelation Function (ACF)** plots were analyzed
* **Partial Autocorrelation Function (PACF)** plots were examined

These plots helped identify potential values for:

* **AR (p)** – AutoRegressive component
* **I (d)** – Degree of differencing
* **MA (q)** – Moving Average component



### 3\. Model Selection

Several **ARIMA (AutoRegressive Integrated Moving Average)** models were fitted and compared using the **Akaike Information Criterion (AIC)**.

The model with the **lowest AIC value** was selected as the final model: ARIMA(0,2,2)

This model achieved a good balance between:

* Model complexity
* Forecasting performance
* Goodness of fit



### 4\. Diagnostic Checking

Residual diagnostics were conducted to verify model assumptions.

The residuals behaved approximately like **white noise**, indicating that the model adequately captured the underlying time series pattern.



### 5\. Forecasting

The finalized **ARIMA(0,2,2)** model was used to forecast dengue cases for:

**January 2022 – June 2022**



Forecast Interpretation:



The model predicts that dengue cases will remain approximately between 3300 - 3600 cases per month during early to mid-2022.

Key observations:

* A **slight increasing trend** is predicted
* Forecast behavior aligns with the **upward trend observed at the end of 2021**
* Results can support **public health planning and preventive measures**

\---



## Dashboard Features

An interactive dashboard was developed using **Streamlit** to provide:

* Historical dengue case visualization
* Time series trend analysis
* ACF and PACF visualizations
* Forecasted dengue cases
* Model insights and interpretation
* Interactive data exploration

\---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Streamlit
* Jupyter Notebook

\---

## How to Run the Project



1\. Download the project files from GitHub



Download or clone the repository to your computer.



2\. Open the project folder



Open the files:



\* `python\_app.py`

\* `requirements.txt`

\* `dengue\_forecasting.ipynb`

\* `dataset.csv`



3\. Install required libraries



Run the following command in the terminal:



```bash

pip install -r requirements.txt

```



4\. Run the Streamlit dashboard



Run:



```bash

streamlit run python\_app.py

```



5\. Open the dashboard



The dashboard will automatically open in your web browser.

## 

## 

