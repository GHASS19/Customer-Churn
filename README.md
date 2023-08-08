# Predicting Customer Churn for a Telecom Company

![image](https://github.com/GHASS19/Customer-Churn/assets/86930309/c6bd7956-06ef-47e0-9ac1-46d51a3feb37)

## Imbalanced Target Variable using Logistic Regression, Random Forest Model and Support Vector Machine to predict if the customer will churn.

The Telecom Company wants to reduce their misclassification costs.  They want to minimize their false positives (predicting churn when the customer doesn’t actually churn) which might result in unnecessary retention efforts, while false negatives (failing to predict churn when the customer actually churns) could lead to revenue loss. 

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

Fortunately we did not have any NaNs in the dataset. Typically this is an important aspect of a data science project and what to do with the null values. Should I make them into the median, mean, 0 values, etc. is what we have to determine to make the data ready for machine learning to get an accurate prediction of our target variable.

During the Data Wrangling stage we found:
1. We have a wide range of call failures for each customer due to the high standard deviation of 7.263.
2. We have more no complains than complains due to the mean being low at .0765.
3. The average age of a customer was 30.99 and the customer value range was from 0-2165.28.
4. On average the customers used text messages more than calls.
5. We have an unblanced target variable of Churn where we have many more customers who did not churn, 2655 compared to 495 who actually did churn.

## 3. Exploratory Data Analysis

Here is some interesting findings:

1. First thing I did in EDA was a heatmap to see what correlations we had that were positive or negitive.

**High Correlation**

a. .96, Age & Age Group. five different groups and five ages in this database.

b. .95, Seconds of Use & Frequency of use. This makes sense as more calls and seconds of calls go hand in hand.

c. .92, Customer Value & Frequency of SMS. This was the most interesting correlation. As the company values customers who use text messages.

**Low Correlation**

a. -.46, Status & Seconds of use. Obviously there is no correalation if the status of a customer is inactive, (2 for inactive and 1 for active) and seconds of telaphone calls.

b. -.45, Status & Frequency of Use. This makes sense just like the previous correlation.

c. -.41, Status & Distinct Called Numbers. Customers cannot dial more numbers if they are inactive.

2. I created a graph of the unbalanced target variable, Churn. This showed us just how big of a difference we had in the customers who did and did not churn.

![image](https://github.com/GHASS19/Customer-Churn/assets/86930309/162145cd-b130-4925-a975-c115172345be)

3. Customers that churned did not use the phone company that much compared to those who keep using their services at the end of 12 months.
4. From the EDA we found that no customers in the oldest and youngest groups cancelled their membership. Very helpful for the company as they can target people in these two age groups. Also they could find out why customers in age groups 2-4 discontinued their services and see if they can fix the issue in manner that is good for the business.
5. Those that use the used the phone company to make phone calls did so using it for a longer time then the customers who cancelled their subscribtion at the end of 12 months.
6. A very interesting find is that customers who had a contractual tariff plan, (2) did not churn at the end of the 12 months.
7. The frenquency which the customers use text messages has many high outliers but has a low median near 10 texts. The frequency of calls has some high outliers beyond the maxium as well. The median for calls per customers is around 50.

## 4. Data Preprocessing & Training

1. I Handled class imbalance with class weights for Logistic Regression and Random Forest. I did a 80/20 train/test split with four Classification Models to predict the customer churn:

a. Logistic Regression
b. Random Forest Model
c. Gradient Boosting
d. Support Vector Machine

I used a standard scaler and class weight of balanced for Logistic Regression and Random Forest Regression to improve convergence, performance, and interpretability. For the Gradient Boosting and Support Vector Machine I used SMOTE fo the class imbalance.

2. I compared the four models metric scores to determine which is the best before I try cross validation and grid search on the models.  

3. These are the five metric scores to measure which model is the best:

**Accuracy*: This metric measures the overall correctness of the model's predictions. It is the ratio of correctly predicted samples to the total number of samples. While accuracy is easy to interpret, it may not be the best metric when classes are imbalanced (i.e., there are significantly more non-churn customers than churn customers).

**Precision**: Precision is the ratio of true positive predictions to all positive predictions (true positives + false positives). It indicates the proportion of correctly predicted churn cases among all predicted churn cases. High precision is valuable when the cost of false positives (predicting churn when the customer doesn't churn) is high.

**Recall**: (Sensitivity or True Positive Rate): Recall is the ratio of true positive predictions to all actual positive samples (true positives + false negatives). It shows the proportion of correctly predicted churn cases among all actual churn cases. High recall is valuable when the cost of false negatives (missing churn customers) is high.

**F1-score**: The F1-score is the harmonic mean of precision and recall. It provides a balanced measure between precision and recall and is useful when you want to find a good trade-off between these two metrics.

**ROC-AUC Score**: The Receiver Operating Characteristic Area Under the Curve (ROC-AUC) is a metric that evaluates the model's ability to discriminate between the two classes. It considers the trade-off between true positive rate (recall) and false positive rate. A higher ROC-AUC score indicates better discrimination ability.

4. This is the metrics for the models:

Metrics for Logistic Regression:
   Accuracy  Precision    Recall  F1-Score   ROC-AUC
0  0.831746   0.511236  0.827273  0.631944  0.829983


Metrics for Random Forest:
   Accuracy  Precision    Recall  F1-Score   ROC-AUC
1  0.936508   0.836538  0.790909  0.813084  0.879108


Metrics for Gradient Boosting:
   Accuracy  Precision    Recall  F1-Score   ROC-AUC
2  0.904762   0.678571  0.863636      0.76  0.888549


Metrics for Support Vector Machine:
   Accuracy  Precision    Recall  F1-Score   ROC-AUC
3  0.866667   0.570652  0.954545  0.714286  0.901311

5. Based on these evaluation metrics, it seems that the Random Forest model performs the best in terms of accuracy, precision, and F1-score. It achieves the highest accuracy (0.936508) among all models and relatively good precision and recall values, resulting in a good F1-score (0.813084).

An F1 score of .81 is quite good for a customer churn project, especially when dealing with an imbalanced target variable. In projects with imbalanced classes, such as customer churn prediction, where the majority of customers are likely to stay (negative class), achieving high accuracy alone might not be sufficient to evaluate model performance.

The F1 score is a suitable metric for imbalanced datasets as it considers both precision and recall, providing a balance between correctly identifying positive samples (churned customers) and minimizing false positives (incorrectly predicting a customer will churn). An F1 score of .81 indicates that the model is performing well in correctly identifying churned customers while keeping false positives relatively low. Thus I recommend we use the Random Forest Model before we do a cross validation to insure the model scores are relevent. Then grid search to find the best parameters for each model.

In most imbalanced scenarios, you would want to strike a balance between precision and recall. High recall ensures that you capture a significant portion of actual churn cases, while high precision ensures that you avoid making too many false positive predictions. Therefore, the F1-score is often considered a good metric for imbalanced datasets because it combines both precision and recall.
