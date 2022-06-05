                          ### Case Study : Diabetes Predictions - Using Decision Tree ##

#############################################################

# Name          : Renuka Gaikwad
# ML Type       : Supervised Learning
# Classifier    : Decision Tree
                  #Change the paramaters and check accuracy
                  #check feature importance 
# DataSet       : diabetes.csv

#############################################################

from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from matplotlib import pyplot as plt
import pandas as pd

def diabetes_predictor():

# Step 1 : get the features and labels from data

    # 1.1 : Read csv
    csv_file = "diabetes.csv"
    data_df = pd.read_csv(csv_file)
    print("-----------------------------------------------------------------------------")
    print(data_df.isna().sum())
    print("-----------------------------------------------------------------------------")
    print(data_df.columns)
    
    y = data_df["Outcome"]
    x = data_df.loc[:, data_df.columns != "Outcome"]
    data_df.drop(columns =["Outcome"], inplace=True)
    x = data_df
    
    print("-----------------------------------------------------------------------------")
    print(x.head(5))
    print("-----------------------------------------------------------------------------")
    print(y.head(5))
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
    
    print("-----------------------------------------------------------------------------")
    print("---x_train----", type(x_train))
    print("-----------------------------------------------------------------------------")
    print("---y_train----", type(y_train))

################################################################################################################################    
    model_1 = DecisionTreeClassifier()
    model_1.fit(x_train, y_train)
    model_1_predicted = model_1.predict(x_test)
    
    train_accuracy = model_1.score(x_train, y_train)
    test_accuracy = model_1.score(x_test, y_test)
    print("-----------------------------------------------------------------------------")
    print("*model_1 : train_accuracy : {:.3f}".format(train_accuracy))
    print("-----------------------------------------------------------------------------")
    print("*model_1 : test_accuracy : {:.3f}".format(test_accuracy))
    print("-----------------------------------------------------------------------------")
    features_imp =     model_1.feature_importances_
    print("features_imp : " , features_imp)
    draw_features_importance(model_1, x, "Model_1 feature Importance : max_depth = none")

##################################################################################################################################   
    model_2 = DecisionTreeClassifier(max_depth = 3)
    model_2.fit(x_train, y_train)
    model_2_predicted = model_2.predict(x_test)

    train_accuracy = model_2.score(x_train, y_train)
    test_accuracy = model_2.score(x_test, y_test)
    print("-----------------------------------------------------------------------------")
    print("*model_2 : train_accuracy : {:.3f}".format(train_accuracy))
    print("-----------------------------------------------------------------------------")
    print("model_2 : test_accuracy : {:.3f}".format(test_accuracy))
    draw_features_importance(model_2, x, "model_2 feature Importance max_depth = 3")
    

def draw_features_importance(model, x_df, title : str = "Features Importance"):
    features_names = model.feature_names_in_
    features_imp = model.feature_importances_
    
    plt.figure()
    plt.barh(features_names, features_imp)
    
    plt.xlabel("feature Importance")
    plt.ylabel("feature Name")
    plt.title(title)
    
    plt.show()

# Main Entry function
def main():
    print("-----------------------------------------------------------------------------")
    print("Diabetes Model by Renuka Gaikwad ")
    print("-----------------------------------------------------------------------------")
    print("Machine Learning Type   : Supervised Machine Learning")
    print("-----------------------------------------------------------------------------")
    print("--- Algorithm : Decision Tree ---- ")
    print("-----------------------------------------------------------------------------")
   
    diabetes_predictor()
    
if __name__ == "__main__":
    main()