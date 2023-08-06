# Customer Churn

## Imbalanced Customer Churn using Logistic Regression, Random Forest Model and Support Vector Machine to predict if the customer will churn.

## 1. The Data
[The Data](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset)

This dataset is randomly collected from an Iranian telecom companyâ€™s database over a period of 12 months. A total of 3150 rows of data, each representing a customer, bear information for 14 columns. The attributes that are in this dataset
are call failures, frequency of SMS, number of complaints, number of distinct calls, subscription length, age group, the charge amount, type of service, seconds of use, status, frequency of use, and Customer Value.

All of the attributes except for attribute churn is the aggregated data of the first 9 months. The churn labels are the state of the customers at the end of 12 months. The three months is the designated planning gap.

These are the 14 columns in the dataset:

1. Anonymous Customer ID (I removed this column for the train test split as it is not useful for machine learning).
2. Call Failures: number of call failures
3. Complains: binary (0: No complaint, 1: complaint)
4. Subscription Length: total months of subscription
5. Charge Amount: Ordinal attribute (0: lowest amount, 9: highest amount)
6. Seconds of Use: total seconds of calls
7. Frequency of use: total number of calls
8. Frequency of SMS: total number of text messages
9. Distinct Called Numbers: total number of distinct phone calls
10. Age Group: ordinal attribute (1: younger age, 5: older age)
11. Tariff Plan: binary (1: Pay as you go, 2: contractual)
12. Status: binary (1: active, 2: non-active)
13. Churn: binary (1: churn, 0: non-churn) - Class label
14. Customer Value: The calculated value of customer

## 2. Data Wrangling
