# Predicting Customer Churn for a Telecom Company

![image](https://github.com/GHASS19/Customer-Churn/assets/86930309/c6bd7956-06ef-47e0-9ac1-46d51a3feb37)

## Imbalanced Target Variable of Customer Churn using Logistic Regression, Random Forest Model and Support Vector Machine to predict if the customer will churn.

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

During this stage of the project I discover the data columns in depth. I found the min, max, mean, standard deviation of each column. I found the value counts, unique values for a few columns. Most importantly we found that we have an imbalanced target varible, Customer Churn. More of the customers did not churn then did a big margin. 2655 did not churn compared to 495 who did. This is good for the company but now we need to account for this when do the train test split for our four models. 

Fortunetaly we did not have any NaNs in the dataset. Typically this is an important aspect of a data science project and what to do with the null values. Should I make them into the median, mean, 0 values, etc. is what we have to determine to make the data ready for machine learning to get an accurate prediction of our target variable.
