import numpy as np
import matplotlib.pyplot as plt

#Return fitted model parameters to the dataset at datapath for each choice in degrees.
#Input: datapath as a string specifying a .txt file, degrees as a list of positive integers.
#Output: paramFits, a list with the same length as degrees, where paramFits[i] is the list of
#coefficients when fitting a polynomial of d = degrees[i].
def main(datapath, degrees):
    paramFits = []

    #fill in
    #read the input file, assuming it has two columns, where each row is of the form [x y] as
    #in poly.txt.
    #iterate through each n in degrees, calling the feature_matrix and least_squares functions to solve
    #for the model parameters in each case. Append the result to paramFits each time.
    with open(datapath, "r") as f:
        data = [line.split() for line in f.read().splitlines()]

    x = [item[0] for item in data]
    y = [item[1] for item in data]

    for i in degrees:
        X = feature_matrix(x, i)
        X_poly = least_squares(X, y)
        paramFits.append(X_poly)

    return paramFits


#Return the feature matrix for fitting a polynomial of degree d based on the explanatory variable
#samples in x.
#Input: x as a list of the independent variable samples, and d as an integer.
#Output: X, a list of features for each sample, where X[i][j] corresponds to the jth coefficient
#for the ith sample. Viewed as a matrix, X should have dimension #samples by d+1.
def feature_matrix(x, d):

    #fill in
    #There are several ways to write this function. The most efficient would be a nested list comprehension
    #which for each sample in x calculates x^d, x^(d-1), ..., x^0.
    X = [[pow(float(i),di) for di in range(d,-1,-1)] for i in x]
    
    return X


#Return the least squares solution based on the feature matrix X and corresponding target variable samples in y.
#Input: X as a list of features for each sample, and y as a list of target variable samples.
#Output: B, a list of the fitted model parameters based on the least squares solution.
def least_squares(X, y):
    X = np.array(X, dtype='float64')
    y = np.array(y, dtype='float64')

    #fill in
    #Use the matrix algebra functions in numpy to solve the least squares equations. This can be done in just one line.
    temp = (np.linalg.inv(X.T @ X) @ X.T) @ y.T
    B = temp.tolist()
    return B

if __name__ == '__main__':
    datapath = 'poly.txt'
    degrees = [2, 4]

    paramFits = main(datapath, degrees)
    print(paramFits)
    print('\n')

    #problem 2
    degrees = [1,2,3,4,5]
    paramFits = main(datapath, degrees)
    print(paramFits)
    print('\n')
    
    #problem 3
    with open(datapath, "r") as f:
        data = [line.split() for line in f.read().splitlines()]

    x = [float(item[0]) for item in data]
    y = [float(item[1]) for item in data]

    plt.scatter(x,y, c='black', label='Given')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    
    markers = [',', '+', 'x', '.', '*']
    
    x.sort()
    y_hat = list()
    for item in paramFits:
        degree = len(item) - 1
        X = feature_matrix(x, degree)
        y_hat = (np.array(X,dtype='float64') @ np.array(item,dtype='float64'))
        mse = 0.0
        for i in range(len(y_hat)):
            mse+=(np.mean((y_hat[i]-y[i])**2))
        mse /= len(item)
        print('MSE for d = {} is {}\n'.format(degree, mse))
        plt.plot(x,y_hat,marker=markers[degree-1],label= 'd = '+str(degree))
    
    plt.legend(fontsize=12)
    plt.title('Plot of fitted model')
    plt.show()
    
    #problem 4
    y_pred = 0.00
    x_data = 2
    for i in range(0,6):
        count = paramFits[4][i] * (x_data ** (5-i))
        y_pred += paramFits[4][i] * (x_data ** (5-i))
    print('y_pred if x = 2 and d = 3 is {}\n'.format(y_pred))
    