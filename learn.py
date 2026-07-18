from sklearn.model_selection import train_test_split   # Imports train test split command of model selection sublibraray of scikit learn.
from sklearn.linear_model import LogisticRegression    # Imports the logistic regression model to predict the results.
from sklearn.metrics import accuracy_score             # Imports the accuracy-score command of the metrics sublibrary of scikit learn.
import pandas as pd                                    # Imports the pandas library
data=pd.read_csv("creditcard.csv")                     # Reads the enlisted csv file.
'''print(data.isna().sum().sum())
print(data.shape)
print(data.head())                                       |
print(data['Amount'].value_counts())                     |
print(data['Time'].value_counts())                       |   
print(data['V1'].value_counts())
print(data['V2'].value_counts())
print(data['V3'].value_counts())                         P
print(data['V4'].value_counts())                         A 
print(data['V5'].value_counts())                         R
print(data['V6'].value_counts())                         T 
print(data['V7'].value_counts())
print(data['V8'].value_counts())                         T 
print(data['V9'].value_counts())                         O
print(data['V10'].value_counts())
print(data['V11'].value_counts())                        A
print(data['V12'].value_counts())                        N
print(data['V13'].value_counts())                        A
print(data['V14'].value_counts())                        L
print(data['V15'].value_counts())                        Y
print(data['V16'].value_counts())                        S
print(data['V17'].value_counts())                        E
print(data['V18'].value_counts())  
print(data['V19'].value_counts())                        D
print(data['V20'].value_counts())                        A 
print(data['V21'].value_counts())                        T
print(data['V22'].value_counts())                        A
print(data['V23'].value_counts())
print(data['V24'].value_counts())                        U
print(data['V25'].value_counts())                        S
print(data['V26'].value_counts())                        I
print(data['V27'].value_counts())                        N
print(data['V28'].value_counts())                        G 
print(data['Amount'].describe())
print(data['Time'].describe())                           D
print(data['V1'].describe())                             E
print(data['V2'].describe())                             S 
print(data['V3'].describe())                             C
print(data['V4'].describe())                             R
print(data['V5'].describe())                             I
print(data['V6'].describe())                             B
print(data['V7'].describe())                             E
print(data['V8'].describe())
print(data['V9'].describe())                             A
print(data['V10'].describe())                            N
print(data['V11'].describe())                            D
print(data['V12'].describe()) 
print(data['V13'].describe())                            V
print(data['V14'].describe())                            A
print(data['V15'].describe())                            L
print(data['V16'].describe())                            U
print(data['V17'].describe())                            E 
print(data['V18'].describe())
print(data['V19'].describe())                            C
print(data['V20'].describe())                            O
print(data['V21'].describe())                            U
print(data['V22'].describe())                            N 
print(data['V23'].describe())                            T 
print(data['V24'].describe())                            S
print(data['V25'].describe())
print(data['V26'].describe())
print(data['V27'].describe())                            |
print(data['V28'].describe())                            | 
print(data['Class'].describe())                          |


print(data[data['Class'] == 1]['Amount'].describe())    # Describes and counts both categories of classes {Fraud and Legitimate} separately.
print(data[data['Class'] == 0]['Amount'].describe())
print(data['Class'].value_counts())'''

legit = data[data.Class==0]         # Puts all the legit transactiomns inside the legit variable.
fraud = data[data.Class==1]         # Puts all the fraud transactions inside the fraud variable.
#print(legit.shape, fraud.shape)    # Prints the shape of both the above initialized variables.

legit.Amount.describe()             # Describes the 'amount' of legit transactions.
fraud.Amount.describe()             # Describes the 'amount' of fraud transactions.

legit_sample = legit.sample(n = 492) # Samples 492 legit transactions for creating the model and stores them in the legit_sample variable.
new_df = pd.concat([legit_sample, fraud], axis = 0) # Creates a clone of the dataset along with legit_sample and fraud as another column to form the complete data needed for creating the model.
print(new_df.shape) # Prints the shape of the new complete dataset

#print(new_df['Class'].value_counts()) 
#print(new_df.groupby('Class').mean())

X = new_df.drop(columns = 'Class')  # Drops the column "Class" from the complete dataset and stores that version in X variable for training.
Y = new_df['Class']                 # Defines the variable Y as the class column for testing and accuracy calculation.

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state= 2) # Splits the data into training and testing data in order by defining 4 variables for the same along with some additional parameters.
print(X.shape, X_train.shape, X_test.shape) # Prints the shape of the training and testing data variables.

model = LogisticRegression() # Initializes logistic regression for creating the model.
model.fit(X_train, Y_train)  # Trains the data with the training data.
X_train_prediction = model.predict(X_train)  # Predicts the training data with only the data and not the answers.  
X_test_prediction = model.predict(X_test)    # Predicts the unseen testing data for measuring accuracy and real-world readiness.
 
training_data_accuracy = accuracy_score(X_train_prediction, Y_train) # Measures the accuracy when the model predicts the data if was trained on.
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)       # Measures the accuracy when the model is tested/evaluated on the new and unseen testing data. 

print(training_data_accuracy)  # Prints the training data prediction accuracy.
print(test_data_accuracy)      # Prints the testing data prediction accuracy.