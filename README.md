# Retail Customer Intelligence System

An end-to-end data science project that analyzes retail customer behavior, segments customers using RFM analysis, and predicts whether a customer is likely to become a high-value customer using Machine Learning.

## Project Overview

Retail businesses generate large amounts of customer transaction data. The goal of this project is to transform transaction-level data into useful business insights and customer intelligence.

This project covers:

* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Business KPI analysis
* RFM customer segmentation
* Customer behavior analysis
* Machine Learning prediction
* XGBoost model training and evaluation
* Streamlit-based prediction application

## Dataset

The project uses the **Online Retail Dataset** from the UCI Machine Learning Repository.

The dataset contains transaction information from a UK-based online retail business, including:

* Invoice number
* Product description
* Quantity
* Invoice date
* Unit price
* Customer ID
* Country

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Joblib
* Streamlit
* Jupyter Notebook / Google Colab

## Project Workflow

### 1. Data Cleaning

The raw transaction data was cleaned by:

* Removing duplicate records
* Handling missing customer IDs
* Removing invalid quantities
* Removing invalid unit prices
* Converting invoice dates into datetime format

A new `Revenue` feature was created:

```text
Revenue = Quantity × UnitPrice
```

Additional time-based features were also extracted, including:

* Year
* Month
* Day
* Hour
* Day of Week

### 2. Business Analysis and EDA

The project analyzes important business metrics such as:

* Total revenue
* Number of orders
* Number of customers
* Average order value
* Monthly revenue
* Top products
* Top countries
* Customer revenue distribution
* Revenue by day of week

These analyses help understand overall sales and customer purchasing behavior.

## 3. RFM Customer Segmentation

Customers were segmented using **RFM analysis**.

### RFM Components

**Recency**
How recently a customer made a purchase.

**Frequency**
How frequently a customer placed orders.

**Monetary**
How much money a customer spent.

K-Means clustering was applied to the RFM features.

Four practical customer segments were identified:

* Regular Customers
* Inactive Customers
* VIP Customers
* Loyal High-Value Customers

These segments can help businesses understand different customer groups and design targeted strategies.

## 4. High-Value Customer Prediction

A supervised Machine Learning problem was created to predict whether a customer would become a high-value customer based on their previous purchasing behavior.

The target was created using future customer spending, while only historical information was used as input features.

### Features Used

* Total Spend
* Total Orders
* Total Items
* Average Order Value
* Average Item Price
* Recency
* Customer Lifetime

Future spending was deliberately excluded from the input features to avoid data leakage.

## 5. Machine Learning Models

The following models were evaluated:

* Logistic Regression
* Random Forest
* XGBoost

Cross-validation was used to compare model performance.

The final model was an XGBoost classifier with hyperparameter tuning using `RandomizedSearchCV`.

### Final Model Results

On the held-out test set:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 90.62% |
| Precision | 82.14% |
| Recall    | 46.94% |
| F1 Score  | 59.74% |
| ROC-AUC   | 86.67% |

The model is intended as a demonstration of customer-value prediction using historical transaction behavior.

## 6. Streamlit Application

A simple Streamlit application was created to demonstrate the trained model.

Users can enter customer information such as:

* Total Spend
* Total Orders
* Total Items
* Average Order Value
* Average Item Price
* Recency
* Customer Lifetime

The application predicts whether the customer is classified as a high-value customer and displays the predicted probability.

## Project Structure

```text
Retail-Customer-Intelligence/
│
├── Retail_Customer_Intelligence.ipynb
├── app.py
├── features.pkl
├── requirements.txt
├── xgboost_high_value_model.pkl
└── README.md
```

## How to Run the Streamlit Application

Clone the repository:

```bash
git clone <your-repository-url>
```

Move into the project directory:

```bash
cd Retail-Customer-Intelligence
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Key Skills Demonstrated

This project demonstrates practical experience with:

* Python for Data Science
* Pandas and NumPy
* Data Cleaning
* Exploratory Data Analysis
* Data Visualization
* Business KPI Analysis
* RFM Analysis
* K-Means Clustering
* Feature Engineering
* Classification
* XGBoost
* Model Evaluation
* Cross-Validation
* Hyperparameter Tuning
* Model Serialization
* Streamlit

## Future Improvements

Possible future improvements include:

* Deploying the Streamlit application online
* Adding interactive business dashboards
* Adding automated customer segmentation
* Incorporating additional customer behavior features
* Improving the prediction interface
