# Homework 11 - Risky Business (Machine Learning Classification)

#### Introduction

This assignment uses machine learning techniques to predict credit using using data from LendingClub. The two different techniques used in this analysis is Resampling and Ensemble Learning. 


#### Data preparation

Before diving into the classification regressions. I first had to further clean the data. I used the label encoder function to convert independent variables into workable numbers. I then realized that the results were not satisfactory so I used the getdummies function to do binary encoding on the independent variables instead. The results below all refer to data that was encoded using getdummies. 

Another part of preparing the data was scaling the X_train and X_test variables. The resulting scaled independent variables are used for all the regression models below. In addition, random_state=1 was used for all models.


### Resampling

This section uses the imbalanced learn library to resample the LendingClub data and evaluate the different logistic regression classifiers used. After cleaning the lending club data, we used the Counter function to look at the number of low risk loans and high risk loans and the results are as follows:

| Samples | Count |
| ------ | ----------- |
| low_risk   | 68470 |
| high_risk | 347 |

This shows that low risk loans have much more samples compared to the high risk loans. To fix this problem, I applied both oversampling and undersampling techniques before doing a logistic regression. For oversampling, I used the Random Oversampler (ROS) and SMOTE algorithms and for undersampling I used the Cluster Centroids algorithm and for combination sampling, I used the SMOTEEN algorithm. Then I compared the regression results using the balanced accuracy score, confusion matrix, imbalanced classification report and geometric mean scores. 

After resampling, the resulting counts have changed to the following table. As you can see, for random oversampler (ROS) and SMOTE, the high risk loan counts have gone up. For cluster centroids, the low risk sample counts have gone down to match the small count of the high risk samples. Lastly, for SMOTEEN (combination), the high risk samples are similar to ROS and SMOTE but the low risk loans have not gone up as much as it had in the other two oversampling models.

| Samples | Random Oversampling | SMOTE | Cluster Centroid | SMOTEENN | 
| ------ | ----------- || ------ | ----------- |----------- |
| low_risk   | 51366 | 51366 | 246 | 51366 |
| high_risk | 51366 | 51366 | 246 | 47365 |


Below is a summary of the key metrics from each of the regressions models.The SMOTEENN model had the highest balanced accuracy score of 0.8388 compared to all other models

| Algorithm | Balanced Accuracy Score |
| ------ | ----------- |
| Random Oversampling   | 0.84 |
| SMOTE | 0.60 |
| Cluster Centroids    | 0.82 |
| SMOTEENN   | 0.84 |


The SMOTE model had the highest recall for high risk loans at 0.86 while the Cluster Centroid model had the highest recall for the low risk loans

| Algorithm | high risk loan recall | low risk loan recall |
| ------ | ----------- | ----------- |
| Random Oversampler   | 0.83 | 0.84 |
| SMOTE | 0.86 | 0.35 |
| Cluster Centroids  | 0.82 | 0.86 |
| SMOTEENN    | 0.69 | 0.74 |


The SMOTEENN model also had the highest geometric mean score at 0.8386. This score maximizes the accuracy of each of the classes.

| Algorithm | Geometric Mean |
| ------ | ----------- |
| Random Oversampler   | 0.84 |
| SMOTE | 0.55 |
| Cluster Centroids    | 0.82 |
| SMOTEENN    | 0.84 |



### Ensemble Learning

For ensemble learning, I used the Balanced Random Forest classifier and the Easy ensemble classifier to predict loan risk. Similar to what was done in the previous section, the models were evaulated using the balanced accuracy score, classification report recall scores and the geometric mean.

For all 3 metrics, the Easy Ensemble Classifier scored much higher compare to the the Balanced Random  Classifier. 


| Algorithm | Balanced Accuracy Score |
| ------ | ----------- |
| Balanced Random Forest  | 0.79 |
| Easy Ensemble Classifier | 0.93 |

---

| Algorithm | high risk loan recall | low risk loan recall |
| ------ | ----------- | ----------- |
| Balanced Random Forest   | 0.67 | 0.90 |
| Easy Ensemble Classifier | 0.92 | 0.94 |

---

| Algorithm | Geometric Mean |
| ------ | ----------- |
| Balanced Random Forest   | 0.78 |
| Easy Ensemble Classifier | 0.93 |

---


Using the feature importances function in the Balanced Random Forest classifier, the top 3 features in determining the credit risk of a borrower as shown below. We can see that the top 3 features all relate to whether a portion of the loan has already been paid which makes sense because it shows a glimpse of past behavior of the borrower.

| Feature | Importance |
| ------ | ----------- |
| total_rec_prncp  | 0.09 |
| total_pymnt_inv | 0.06 |
| total_pymnt | 0.06 |


#### Conclusions

This analysis shows the different algorithms that can be applied on imbalanced datasets. Based on the different models tested, the SMOTEENN model and Easy Ensemble Classifier seemed to work best at predicting credit risk for our LendingClub data.

#### Jupyter Notebooks for reference

- [Resampling](https://github.com/nikanikachan/HW11_Classification/blob/main/credit_risk_resampling%20_getdummies.ipynb)
- [Ensemble Learning](https://github.com/nikanikachan/HW11_Classification/blob/main/credit_risk_ensemble_getdummies.ipyn

Same analysis notebooks but in these, I used label encoder instead of get dummies fo encoding:

- [Resampling using label encoder](https://github.com/nikanikachan/HW11_Classification/blob/main/credit_risk_resampling_labelencoder.ipynb)
- [Ensemble Learning using label encoder](https://github.com/nikanikachan/HW11_Classification/blob/main/credit_risk_ensemble_labelencoder.ipynb)
