import numpy as np
import scipy.stats as stats

def z_test(file):
    with open(file, "r") as f:
        data = [float(line) for line in f.read().splitlines()]
    #Problem 1 Question 1 and Question 2
    mu = 0.75
    sample_size = len(data)
    sample_mean = np.mean(data)
    sd = np.std(data, ddof=1) #ddof 1 gives an unbiased
    #estimator of the variance of the sample population

    std_error = sd/np.sqrt(sample_size)
    z_score = (sample_mean - mu) / std_error
    p = 2* stats.norm.cdf(-abs(z_score))

    #Problem 1 Question 3
    #to get standard error less than 0.05
    new_p = 0.05
    z_alt = stats.norm.ppf(new_p/2)
    new_std_error = (sample_mean-mu)/z_alt
    min_sample_size = (sd/new_std_error)**2

    return (sample_size,sample_mean,sd,std_error,z_score,p, new_std_error,min_sample_size)

def two_z_test(file1,file2):
    #Problem 1 Question 4 & 5
    n0,mean0, sd0,se0, sc0, pval0, _, _ = z_test('eng1.txt')
    n1,mean1, sd1,se1, sc1, pval1, _, _ = z_test('eng0.txt')
    
    mu = 0
    test_point = mean0 - mean1
    sd = np.sqrt( ((sd0**2)/n0) + ((sd1**2)/n1) )
    z_score = (test_point - mu)/sd
    p = 2* stats.norm.cdf(-abs(z_score))

    return (n0,n1,mean0,mean1,se0,se1,sc0,sc1,pval0,pval1,z_score,p)


if __name__ == '__main__' :
    n1,mean1, sd1, se1, sc1, pval1,new_sd, min_size = z_test('eng1.txt')
    print("Problem 1 Q1 & Q2 answers:")
    print("Sample size = {}, sample mean = {}, standard error = {}, standard cost = {}, p value = {}\n".format(n1,mean1, sd1, se1, sc1, pval1))

    print("Problem 1 Q3 answer when p = 0.05:")
    print("New standard error = {}, minimum sample size = {}\n".format(new_sd,min_size))
    
    n0,n1,mean0,mean1,se0,se1,sc0,sc1,pval0,pval1,z_score,p = two_z_test('eng1.txt','eng0.txt')
    print("Problem 1 Q4 & Q5 answers:")
    print("This is for sample from eng1.txt")
    print("Sample size = {}, sample mean = {}, standard error = {}, standard cost = {}, p value = {}\n".format(n1,mean1, se1, sc1, pval1))
    print("This is for sample from eng0.txt")
    print("Sample size = {}, sample mean = {}, standard error = {}, standard cost = {}, p value = {}\n".format(n0,mean0, se0, sc0, pval0))
    print("Z-score = {}, p-value for the two-z-sample-test = {}".format(z_score,p))



