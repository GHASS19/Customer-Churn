# Predicting Customer Churn for a Telecom Company

![image](https://github.com/GHASS19/Customer-Churn/assets/86930309/c6bd7956-06ef-47e0-9ac1-46d51a3feb37)

## Imbalanced Target Variable using Logistic Regression, Random Forest Model and Support Vector Machine to predict if the customer will churn.

The Telecom Company wants to reduce their misclassification costs.  They want to minimize their false positives (predicting churn when the customer doesn’t actually churn) which might result in unnecessary retention efforts. They are also concerned with false negatives (failing to predict churn when the customer actually churns) which could lead to revenue loss. Thus we will find the best model with the highest AUC-ROC and F-1 Scores.

## 1. The Data
[The Data](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset)

This dataset is randomly collected from an Iranian telecom companyâ€™s database over a period of 12 months. A total of 3150 rows of data, each representing a customer, bear information for 14 columns. 
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
[Data Wrangling](https://colab.research.google.com/drive/1TEwnGjlsV8_GV6qv2hPs97oRMLc4yPNa#scrollTo=aI7LZ3hotZkc)

During this stage of the project I discovered the data columns in depth. I found the min, max, mean, standard deviation of each column. I found the value counts, unique values for a few columns. Most importantly we found that we have an imbalanced target variable, Customer Churn. More of the customers did not churn than did by a big margin. 2655 did not churn compared to 495 who did. This is good for the company but we need to account for this when we do the train/test split for our four models. 

Fortunately we did not have any NaNs in the dataset. Typically this is an important aspect of a data science project and what to do with the null values. Whether I should make them into the median, mean, 0 values, etc. is what we have to determine to make the data ready for machine learning to get an accurate prediction of our target variable.

During the Data Wrangling stage we found:
1. We have a wide range of call failures for each customer due to the high standard deviation of 7.263.
2. We have more no complains than complains due to the mean being low at .0765.
3. The average age of a customer was 30.99 and the customer value range was from 0-2165.28.
4. On average the customers used text messages more than calls.
5. We have an unbalanced target variable of Churn where we have many more customers who did not churn, 2655 compared to 495 who actually did churn.

## 3. Exploratory Data Analysis

Here is some interesting findings:

3.1 First thing I did in EDA was a heatmap to see what correlations we had that were positive or negative.

**High Correlation**

a. .96, Age & Age Group. five different groups and five ages in this database.

b. .95, Seconds of Use & Frequency of use. This makes sense as more calls and seconds of calls go hand in hand.

c. .92, Customer Value & Frequency of SMS. This was the most interesting correlation. As the company values customers who use text messages.

**Low Correlation**

a. -.46, Status & Seconds of use. Obviously there is no correlation if the status of a customer is inactive, (2 for inactive and 1 for active) and seconds of telephone calls.

b. -.45, Status & Frequency of Use. This makes sense just like the previous correlation.

c. -.41, Status & Distinct Called Numbers. Customers cannot dial more numbers if they are inactive.

3.2 I created a graph of the unbalanced target variable, Churn. This showed us just how big of a difference we had in the customers who did and did not churn.

![image](https://github.com/GHASS19/Customer-Churn/assets/86930309/162145cd-b130-4925-a975-c115172345be)

3. Customers that churned did not use the phone company that much compared to those who keep using their services at the end of 12 months.
4. From the EDA we found that no customers in the oldest and youngest groups canceled their membership. Very helpful for the company as they can target people in these two age groups. Also they could find out why customers in age groups 2-4 discontinued their services and see if they can fix the issue in a manner that is good for the business.
5. Those that used the phone company to make phone calls did so using it for a longer time then the customers who cancelled their subscription at the end of 12 months.
6. A very interesting find is that customers who had a contractual tariff plan, (2) did not churn at the end of the 12 months.
7. The frequency which the customers use text messages has many high outliers but has a low median near 10 texts. The frequency of calls has some high outliers beyond the maxim as well. The median for calls per customer is around 50.

## 4. Data Preprocessing & Training

4.1 I Handled class imbalance with class weights for Logistic Regression and Random Forest. I did a 80/20 train/test split with four Classification Models to predict the customer churn:

a. Logistic Regression

b. Random Forest Model

c. Gradient Boosting

d. Support Vector Machine

4.2 I used a standard scaler and class weight of balanced for Logistic Regression and Random Forest Regression to improve convergence, performance, and interpretability. For the Gradient Boosting and Support Vector Machine I used SMOTE for the class imbalance.

4.3 I compared the four models' metric scores using cross validation and grid search.  

4.4 After evaluating the models, it was found that Gradient Boosting emerged as the best model for predicting customer churn in the telecom company's dataset. The key reasons supporting this conclusion are as follows:

**High Accuracy:** Gradient Boosting achieved the highest accuracy of 94.60%, indicating that it correctly predicted customer churn in a significant proportion of cases.

**High Precision and Recall:** The model exhibited a high precision of 88.00%, signifying that when it predicted a customer would churn, it was accurate about 88.00% of the time. Additionally, the recall score of 80.00% indicated that the model could correctly identify 80.00% of actual churned customers.

**Best F1-Score:** The F1-Score of 83.81% was the highest among all models, implying an optimal balance between precision and recall. This is crucial in an imbalanced dataset like this, where both false positives and false negatives need to be minimized.

**Competitive ROC-AUC:** While Gradient Boosting ROC-AUC score of 88.85% was slightly lower than the Support Vector Machine, it still demonstrated a strong ability to distinguish between churned and non-churned customers.

4.5 In this project, we focused on predicting customer churn for an Iranian telecom company. With an emphasis on ROC-AUC and F1 scores, we evaluated four models to strike a balance between identifying churn instances and minimizing false positives. After rigorous analysis, Gradient Boosting emerged as the top-performing model, excelling in both ROC-AUC (0.888) and F1-score (0.838).

## 5. Recommendations and Going Forward

Based on our findings, we recommend that the telecom company leverages the predictive power of the Gradient Boosting model to enhance customer churn management. By proactively identifying potential churners, the company can tailor retention strategies effectively. This includes personalized offers, targeted communication, and improved customer support for at-risk customers. Regular model updates and continuous monitoring of performance will be essential to ensure the model's effectiveness over time. Additionally, investing in data quality and gathering more features related to customer behavior and preferences can further improve predictive accuracy and optimize retention efforts.

Overall, the Gradient Boosting model outperformed the other models in various metrics, making it the top choice for predicting customer churn in this telecom company's dataset. The model's ability to strike a balance between precision and recall, along with its high accuracy, makes it a valuable tool for the company to identify and retain potentially churned customers, thus helping to optimize their business strategies and improve customer retention efforts.
