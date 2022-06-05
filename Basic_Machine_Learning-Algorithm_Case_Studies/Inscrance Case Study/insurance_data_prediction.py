                                   ### Case Study : Insurance Data Prediction ###

#############################################################

# Name          : Renuka Gaikwad
# ML Type       : Supervised Learning
# Classifier    : Logistic Regression
                  #Data set contain person age and insurance is purchased or not
                  #Calculate F1 score and display confusion_matrix
# DataSet       : insurance_data.csv
#############################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def insurance_logestic(data_file_path):
    
    df = pd.read_csv(data_file_path)
    print("-----------------------------------------------------------------------------")
    print("\n First entries : ")
    print("-----------------------------------------------------------------------------")
    print(df.head(5))
    print("-----------------------------------------------------------------------------")

    # show data on graph
    plt.scatter(x=df.Age, y= df.bought_insurance, marker="+", color="red")
    plt.show()
    
    # split the data
    x_train, x_test, y_train, y_test = train_test_split(df[["Age"]], df.bought_insurance, train_size= 0.3)
    
    print("\n Indepdent variable of training x_train: ")
    print(x_train)
    print("-----------------------------------------------------------------------------")
    
    print("\n Dedepdent variable of training y_train: ")
    print(y_train)
    print("-----------------------------------------------------------------------------")
    
    print("\n Indepdent variable of Test x_test: ")
    print(x_test)
    print("-----------------------------------------------------------------------------")
    
    print("\n Dedepdent variable of Test y_test: ")
    print(y_test)
    print("-----------------------------------------------------------------------------")
    
    model = LogisticRegression();

    # train the model
    model.fit(x_train, y_train)
    
    # predict the output
    y_predicted = model.predict(x_test)

    print("\n predict outut y_predicted: ")
    print("-----------------------------------------------------------------------------")
    print(y_predicted)
    print("-----------------------------------------------------------------------------")
    
    # Probability
    prob = model.predict_proba(x_test)

    print(f"\n Probability is Result : 0 and 1 are ")
    print("-----------------------------------------------------------------------------")
    print(prob)
    print("-----------------------------------------------------------------------------")
 
    # confusion_matrix
    print("\n confusion_matrix is :")
    print("-----------------------------------------------------------------------------")
    print(confusion_matrix(y_test, y_predicted))
    print("-----------------------------------------------------------------------------")
        
    
    # classification report
    class_report = classification_report(y_test, y_predicted)
    print("\n Calssification reports of Logistic regression is :")
    print("-----------------------------------------------------------------------------")
    print(class_report)
    print("-----------------------------------------------------------------------------")
    
     
    

# Main Entry function
def main():
    print("-----------------------------------------------------------------------------")
    print("Insurance Case Study By Renuka Gaikwad  ")
    print("-----------------------------------------------------------------------------")
    print("Machine Learning Type   : Supervised Machine Learning")
    print("-----------------------------------------------------------------------------")
    print("Algorithm : Logistic Regression")
    
    data_file_path = "insurance_data.csv" 
    insurance_logestic(data_file_path)

# Starter
if __name__ == "__main__":
    main()