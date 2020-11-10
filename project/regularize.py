import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def regExt(X,y):
    #Training and testing split, with 10% of the data reserved as the test set
    X, _, _ = normalize_train(X)
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.09, random_state=101)

    #Normalizing training and testing data
    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    
    #Define the range of lambda to test
    lmbda = np.logspace(-1,2,num=101)
    #lmbda = np.linspace(.00001, 2, 500).tolist()
    #lmbda = [round(i, 2) for i in lmbda]
    #lmbda = np.linspace(-12, 100, 1000).tolist()
    #lmbda = [round(i, 2) for i in lmbda]
    
    MODEL = []
    MSE = []
    for l in lmbda:
        #Train the regression model using a regularization parameter of l
        model = Ridge(alpha=l, fit_intercept = True).fit(X_train,y_train)
        #Evaluate the MSE on the test set
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        #Store the model and mse in lists for further processing
        MODEL.append(model)
        MSE.append(mse)
    #find the best lmbda
    ind = MSE.index(min(MSE))
    
    #return best model, min MSE, lmbda_best, lmbda, MSE
    return MODEL[ind], min(MSE), lmbda[ind], lmbda, MSE
    
    

#Function that normalizes features in training set to zero mean and unit variance.
#Input: training data X_train
#Output: the normalized version of the feature matrix: X, the mean of each column in
#training set: trn_mean, the std dev of each column in training set: trn_std.
def normalize_train(X_train):
    
    #fill in
    #for each column, calculate column's mean and std deviation 
    #subtract the mean and std deviation and subtract that mean from each
    #element in the column and divide result by column's std deviation
    #try:
    mean = np.empty((len(X_train[0]), 1))
    std = np.empty((len(X_train[0]),1))
    X = np.empty(np.shape(X_train))
    for col in range(len(X_train[0])): #get the no of column
        mean[col] = np.mean(X_train[:,col])
        std[col] = np.std(X_train[:,col])
        X[:,col] = (X_train[:,col] - np.mean(X_train[:,col])) / (np.std(X_train[:,col]))
    '''
    except:
        mean = np.empty((len(X_train), 1))
        std = np.empty((len(X_train),1))
        X = np.empty(np.shape(X_train))
        for col in range(len(X_train)): #get the no of column
            mean[col] = np.mean(X_train[col])
            std[col] = np.std(X_train[col])
            X[:,col] = (X_train[col] - np.mean(X_train[col])) / (np.std(X_train[col]))
    '''
        
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

if __name__ == '__main__':
    ...
    #main()