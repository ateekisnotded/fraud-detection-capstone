\# Fraud Detection Capstone Project



An end-to-end machine learning project for detecting fraudulent financial transactions, with an interactive Streamlit prediction app and a Power BI analytics dashboard.



\## 🚀 Live Demo



\*\*Streamlit App:\*\*  

https://fraud-detection-capstone.streamlit.app



The deployed application allows users to enter transaction details and receive a fraud-risk prediction from the trained machine learning model.



\---



\## 📌 Project Overview



Financial transaction fraud can result in significant financial losses and requires systems that can identify suspicious activity accurately.



This project develops a machine learning-based fraud detection system that:



\- analyzes transaction-level financial data

\- performs data preprocessing and feature engineering

\- compares multiple classification models

\- selects a Random Forest model for the prediction workflow

\- provides real-time predictions through Streamlit

\- presents fraud patterns and model validation through Power BI



The project is designed as an end-to-end workflow covering \*\*data analysis, machine learning, model validation, deployment, and business intelligence\*\*.



\---



\## 📊 Dataset



The dataset contains \*\*11,142 financial transactions\*\* consisting of:



\- \*\*10,000 legitimate transactions\*\*

\- \*\*1,142 fraudulent transactions\*\*

\- \*\*Fraud rate: 10.25%\*\*



Fraudulent transactions in the dataset occur in the \*\*TRANSFER\*\* and \*\*CASH\_OUT\*\* transaction types.



\---



\## ⚙️ Feature Engineering



The project includes engineered features to capture inconsistencies between transaction amounts and account balances:



\- `orig\_balance\_change`

\- `dest\_balance\_change`

\- `orig\_balance\_error`

\- `dest\_balance\_error`



Account identifiers are excluded from model training, while transaction and balance-related variables are retained for prediction.



\---



\## 🤖 Machine Learning



The following classification models were evaluated:



\- Logistic Regression

\- Random Forest

\- Gradient Boosting



The \*\*Random Forest\*\* model was selected for the deployed prediction workflow.



\### Model Performance



| Metric | Score |

|---|---:|

| Precision | 1.0000 |

| Recall | 0.9956 |

| F1 Score | 0.9978 |

| ROC-AUC | 0.999957 |



\### Confusion Matrix



| Actual / Predicted | Fraud | Legitimate |

|---|---:|---:|

| Fraud | 227 | 1 |

| Legitimate | 0 | 2001 |



The validation results show \*\*0 false positives\*\* and \*\*1 false negative\*\* on the held-out validation data.



\---



\## 🌐 Streamlit Application



The Streamlit application provides an interactive interface where users can enter:



\- transaction step

\- transaction type

\- transaction amount

\- origin account balances

\- destination account balances



The application processes the inputs using the same feature engineering and trained Random Forest pipeline and returns the model's prediction.



\---



\## 📈 Power BI Dashboard



The project also includes a one-page Power BI dashboard for fraud analytics and model validation.



\### Dashboard sections include:



\- Total Transactions

\- Fraud Transactions

\- Fraud Rate

\- Total Fraud Amount

\- Fraud Amount by Transaction Type

\- Fraud Activity Over Time

\- Fraud vs Legitimate Transactions

\- Confusion Matrix

\- Precision

\- Recall

\- F1 Score

\- ROC-AUC

\- False Positives

\- False Negatives

\- Model Accuracy



Power BI file:



`Fraud Detection Power BI Dashboard.pbix`



\---



\## 🗂️ Project Structure



```text

Fraud Detection Capstone Project/

│

├── app.py

├── requirements.txt

├── Fraud\_Analysis\_Dataset.xlsx

├── Fraud\_Detection\_Capstone\_Final.ipynb

├── Fraud Detection Power BI Dashboard.pbix

├── run\_app.bat

├── README.md

│

├── models/

│   └── random\_forest.joblib

│

├── src/

│   ├── preprocessing.py

│   ├── train.py

│   └── predict.py

│

└── output/

&#x20;   └── pdf/

