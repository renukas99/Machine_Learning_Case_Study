                            ### Case Study : Play Predictor By Using KNN Classifier With The Accuracy###

#############################################################

# Name          : Renuka Gaikwad

# ML Type       : Supervised Learning
# Classifier    : KNN (KNeighborsClassifier)
                  #Calculate the accuracy of [PlayPredictor] : KNN model
                  #For Knn (KNeighborsClassifier), accuracy changes based on value [n_neighbors] parameter
                  #Change N_Neighbors value from 2 to 10 and calculate accuracy
                  #For testing user 0.3 data set.Training is = 80%
#To split the data use     : sklearn.model_selection ==> train_test_split
#To calculate accuracy use :   sklearn.metrics ==> accuracy_score
# DataSet       : Play Predictor
# Features      : Weather & Temperature
# Label         : Yes and No

#############################################################

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd


def play_predictor_accuracy():
    
    print("-----------------------------------------------------------------------------------------------------------------")
    print("play_predictor_accuracy ")
    print("-----------------------------------------------------------------------------------------------------------------")
    csv_data = pd.read_csv("Play_Predictor.csv")
    print(csv_data.head(n=5))

    print("-----------------------------------------------------------------------------------------------------------------")
    label_encoder = preprocessing.LabelEncoder()
    csv_data["weather_enc"] = label_encoder.fit_transform(csv_data["Weather"])
    csv_data["temp_enc"] = label_encoder.fit_transform(csv_data["Temperature"])
    csv_data["play_enc"] = label_encoder.fit_transform(csv_data["Play"])
    print(csv_data.head(n=5))
    print("-----------------------------------------------------------------------------------------------------------------")

    data = csv_data[["weather_enc", "temp_enc"]].values
    target = csv_data[["play_enc"]].values

    # split the data set
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.3)

    # classifier = KNeighborsClassifier(n_neighbors=3)
    # model = classifier.fit(train_data, train_target.ravel())
    # predictions = model.predict(test_data)
    # acc_score = accuracy_score(test_target, predictions)
    # print(f"Model Accuracy is : {acc}")

    accuracy_result = {}
    for k in range(2, 11):
        # Algorithm
        classifier = KNeighborsClassifier(n_neighbors=k)

        # train the algorithm
        model = classifier.fit(train_data, train_target.ravel())

        # predict the outputs
        predictions = model.predict(test_data)

        # calculate the accuracy
        acc_score = accuracy_score(test_target, predictions)
        # print(f"Model Accuracy is : {acc}")
        accuracy_result[k] = acc_score

    return accuracy_result


def main():

    # Note : train_test_split method ==> by default shuffle the data. So every time accuracy is different thas why use the for loop for check the different accuracy
    for i in range(3):
        accuracy_result = play_predictor_accuracy()
        
        print(f"accuracy_result = {accuracy_result}")
        

if __name__ == "__main__":
    main()