           
                           ###Case Study : Breast Cancer Predictions : Using SVM (Support Vector Machine) ###


#############################################################

# Name          : Renuka Gaikwad
# ML Type       : Supervised Learning
# Classifier    : Support Vector Machine
# DataSet       : sklearn.datasets ==> breast_cancer

#############################################################

from sklearn import svm
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def svm_lienar():

    #Load dataset from sklearn.datasets 
    cancer = load_breast_cancer()

    # print  features
    print("\n")
    print("Features ", cancer.feature_names)
    print("-----------------------------------------------------------------------------")
    print("First 5 records are : ")
    print("-----------------------------------------------------------------------------")
    print(cancer.data[0:5])
    print("-----------------------------------------------------------------------------")

    # print Target / Label
    print("Target ", cancer.target_names)
    print("-----------------------------------------------------------------------------")

    # 0='malignant' 1='benign'
    print(cancer.target[:5])
    print("-----------------------------------------------------------------------------")

    #split the data
    x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, train_size=0.5, random_state=44)

    print("kernel   =  \"linear\" ")
    print("----------------------------------------------------------------------------")

    #Create model
    model_svc_linear = svm.SVC(kernel = "linear")

    # Train the model
    model_svc_linear.fit(x_train, y_train)

    # Predict the response
    y_predict = model_svc_linear.predict(x_test)

    # Evaluate the model
    
    print_accurarcy(y_true=y_test, y_pred=y_predict)

    
    print("kernel  =  \"rbf\" ")

    print("-----------------------------------------------------------------------------")

    #Create model
    model_svc_rbf = svm.SVC(kernel="rbf")

    # Train the model
    model_svc_rbf.fit(x_train, y_train)

    # Predict the response
    y_predict = model_svc_rbf.predict(x_test)

    # Evaluate the model
    print_accurarcy(y_true=y_test, y_pred=y_predict)

    print("------------------------------------------------------------------------------")

def print_accurarcy(y_true, y_pred):
    # Evaluate the model
    # calcualate accurarcy
    
    acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)

    print("Accuarcy score : {:.4} ".format(acc_score))

    print("------------------------------------------------------------------------------")
    
    # get Precision and Recall
    precision = metrics.precision_score(y_true=y_true, y_pred=y_pred)

    print("precision is : {:.4} ".format(precision))

    print("------------------------------------------------------------------------------")

    recall = metrics.recall_score(y_true=y_true, y_pred=y_pred)
    print("recall is : {:.4} ".format(precision))

    print("------------------------------------------------------------------------------")

# Main Entry function
def main():
    
    print("Breast Cancer Case Study By Renuka Gaikwad ")
    print("------------------------------------------------------------------------------")
    print("Machine Learning Type   : Supervised Machine Learning")
    print("------------------------------------------------------------------------------")
    print("Algorithm : SVM (Support Vector Machine")
    print("------------------------------------------------------------------------------")
    print("\n")

    svm_lienar()


if __name__ == "__main__":
    main()