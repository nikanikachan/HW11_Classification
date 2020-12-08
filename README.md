# Homework 11 - Risky Business (Machine Learning Classification)

#### Resampling

This section uses the imbalanced learn library to resample the LendingClub data and evaluate the different logistic regression classifiers used. After cleaning the lending club data, we used the Counter function to look at the number of low risk loans and high risk loans and the results are as follows:

| Samples | Count |
| ------ | ----------- |
| low_risk   | 68470 |
| high_risk | 347 |

This shows that low risk loans have much more samples compared to the high risk loans. To fix this problem, I applied both oversampling and undersampling techniques before doing a logistic regression. For oversampling, I used the Random Oversampler and SMOTE algorithms and for Undersampling I used the cluster centroids and for combination sampling, I used the SMOTEEN algorithm. Then, compared the regression results using the balanced accuracy score, confusion matrix and imbalanced classification report. Below is a summary of the key metrics from each of the regressions models.

The SMOTEEN model had the highest balanced accuracy score of 0.8388 compared to all other models

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


The SMOTEEN model also had the highest geometric mean score at 0.8386. This score maximizes the accuracy of each of the classes.


| Algorithm | Geometric Mean |
| ------ | ----------- |
| Random Oversampler   | 0.84 |
| SMOTE | 0.55 |
| Cluster Centroids    | 0.82 |
| SMOTEENN    | 0.84 |

Jupyter Notebook with above results (using getdummies for binary encoding):
Jupyter Notebook with same models but using integer encoding with labelencoder: 


#### Ensemble Learning

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

Using the Balanced Random Forest classifier, the top 3 features are:


In addition to the above metrics, I looked at the most important independed variables. They turned out to be x,y,z


Jupyter Notebook with above results (using getdummies for binary encoding):
Jupyter Notebook with same models but using integer encoding with labelencoder: 

Use the above to answer the following:

> Which model had the best balanced accuracy score?
>
> Which model had the best recall score?
>
> Which model had the best geometric mean score?
>
> What are the top three features?
