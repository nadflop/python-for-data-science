import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt


def main():
    #Importing dataset
    diamonds = pd.read_csv('diamonds.csv')

    #Feature and target matrices
    X = diamonds[['carat', 'depth', 'table', 'x', 'y', 'z', 'clarity', 'cut', 'color']]
    y = diamonds[['price']]

    #Training and testing split, with 25% of the data reserved as the test set
    X = X.to_numpy()
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.25, random_state=101)

    #Normalizing training and testing data
    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    print('X_train: {},\n trn_mean: {},\n trn_std: {}'.format(X_train, trn_mean, trn_std))
    print()
    
    X_test = normalize_test(X_test, trn_mean, trn_std)
    print('X_train: {}\n '.format(X_test))
    
    #Define the range of lambda to test
    lmbda = np.logspace(-1,2,num=101)
    #lmbda = [1,100]
    
    MODEL = []
    MSE = []
    for l in lmbda:
        #Train the regression model using a regularization parameter of l
        model = train_model(X_train,y_train,l)

        #Evaluate the MSE on the test set
        mse = error(X_test,y_test,model)

        #Store the model and mse in lists for further processing
        MODEL.append(model)
        MSE.append(mse)

    #Plot the MSE as a function of lmbda
    plt.plot(lmbda, MSE)
    plt.xlabel('lmbda')
    plt.ylabel('MSE')
    plt.title('MSE as a function of lmbda')
    plt.show()
    
    #Find best value of lmbda in terms of MSE
    ind = MSE.index(min(MSE))
    [lmda_best,MSE_best,model_best] = [lmbda[ind],MSE[ind],MODEL[ind]]

    print('Best lambda tested is ' + str(lmda_best) + ', which yields an MSE of ' + str(MSE_best))
    
    #problem 7
    #using coefficient and intercept from model_best
    
    new_diamond = np.array([0.25,3,3,5,60,55,4,3,2])
    norm_diamond = (new_diamond - trn_mean.T) / (trn_std.T)
    print('coefficients {}'.format(model_best.coef_))
    print('Normalized diamond data: {}'.format(norm_diamond))
    #y = mx + c
    predicted_price = (np.dot(model_best.coef_, norm_diamond.T)) + model_best.intercept_
    print('Predicted price for diamond is: {}'.format(predicted_price))
    
    return model_best
    
    

#Function that normalizes features in training set to zero mean and unit variance.
#Input: training data X_train
#Output: the normalized version of the feature matrix: X, the mean of each column in
#training set: trn_mean, the std dev of each column in training set: trn_std.
def normalize_train(X_train):
    
    #fill in
    #for each column, calculate column's mean and std deviation 
    #subtract the mean and std deviation and subtract that mean from each
    #element in the column and divide result by column's std deviation
    
    mean = np.empty((len(X_train[0]), 1))
    std = np.empty((len(X_train[0]),1))
    X = np.empty(np.shape(X_train))
    
    for col in range(len(X_train[0])): #get the no of column
        mean[col] = np.mean(X_train[:,col])
        std[col] = np.std(X_train[:,col])
        X[:,col] = (X_train[:,col] - np.mean(X_train[:,col])) / (np.std(X_train[:,col]))
        
    return X, mean, std


#Function that normalizes testing set according to mean and std of training set
#Input: testing data: X_test, mean of each column in training set: trn_mean, standard deviation of each
#column in training set: trn_std
#Output: X, the normalized version of the feature matrix, X_test.
def normalize_test(X_test, trn_mean, trn_std):

    #fill in
    X = np.empty(np.shape(X_test))
    
    for col in range(len(X_test[0])): #get the no of column
        X[:,col] = (X_test[:,col] - trn_mean[col]) / trn_std[col]

    return X



#Function that trains a ridge regression model on the input dataset with lambda=l.
#Input: Feature matrix X, target variable vector y, regularization parameter l.
#Output: model, a numpy object containing the trained model.
def train_model(X,y,l):

    #fill in
    model = Ridge(alpha=l, fit_intercept = True)
    model.fit(X,y)

    return model


#Function that calculates the mean squared error of the model on the input dataset.
#Input: Feature matrix X, target variable vector y, numpy model object
#Output: mse, the mean squared error
def error(X,y,model):

    #Fill in
    
    y_pred = model.predict(X)
    mse = metrics.mean_squared_error(y, y_pred)
    return mse

if __name__ == '__main__':
    main()