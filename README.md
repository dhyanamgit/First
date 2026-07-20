# CREDIT CARD FRAUD DETECTION MODEL.

This project is a credit card detection dataset that has been analysed. It also includes a machine learning model that would help predict unseen data with around 94% accuracy.

Requirements:
1) Pandas
2) Numpy
3) Matplotlib
4) Sklearn

This dataset has been taken from the "Credit Card Fraud Detection Dataset" available on Kaggle. 

## <h2> DATA USAGE: </h2>
Q) The data has been undersampled instead of oversampling. Why? <br><b>
A) This is because there is a massive difference in the number of legitimate and fraud transcations in the data. If we oversample the data, then the predictions and accuracy of the model will be hampered as we would have to create unreal data for almost all the columns. Thus, undersampling is better in this case.</b>

```
# 492 fraud transactions exist
# More than 280 k legitimate transactions exist

Thus, we need to undersample the data using:

variable.sample()
```

## <h2> CODE FOR VISUALIZATION </h2>
* We first analyse the data using ```value_counts()``` , ```isnull()``` , and ```describe()```.
* We then analyse the data and plot some graphs to understand correlations between different unnamed features and their relevance to the predictions.
* OPTIONAL- We sort out the most relevant features and make sure to give them higher precendence during predictions.
* We can skip step 3 as LogisticRegression does this very thing to make predictions on numerical, categorical datasets.

## <h2> CREATING THE MODEL </h2>
* We first import the logistic regression library using
```
 from sklearn.linear_model import LogisticRegression
```
* Separate class column from other dataset. Let the excluded dataset with the features be stored in X and the class column in Y.
* We then define the training and testing data using the train_test_split command
```
 from sklearn.model_selection import train_test_split
 X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_sixe = 0.2, random_state = 3)
```
* We then initialize ``` LogisticRegression() ```




