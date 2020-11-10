import numpy as np
import scipy.stats as stats

def confidence_test1(data, c):
    #Problem 2 Question 1 and 2
    n = len(data)
    x = np.mean(data)
    std_dv = np.std(data, ddof=1)
    std_error = std_dv/np.sqrt(n)
    
    #using t_test
    t_c = stats.t.ppf(1-(1-c)/2, df=n-1)
    mu_t = ( (x-((t_c*std_error)/np.sqrt(n)) ), (x+((t_c*std_error)/np.sqrt(n)) ))

    return x, std_error, t_c, mu_t
    
def confidence_test2(data, c):
    #Problem 2 Question 3
    n = len(data)
    x = np.mean(data)
    std_dv = 16.836
    std_error = std_dv/np.sqrt(n)
    
    #using z_test
    z_c = stats.norm.ppf(1-(1-c)/2)
    mu_z = ( (x-((z_c*std_error)/np.sqrt(n)) ), (x+((z_c*std_error)/np.sqrt(n)) ))

    return x, std_error, z_c, mu_z

def confidence_test3(data, c):
    #Problem 2 Question 4
    n = len(data)
    x = np.mean(data)
    std_dv = np.std(data, ddof=1)
    std_error = std_dv/np.sqrt(n)
    
    #using t_test since we don't know the population standard deviation
    #need to solve for c
    t_c = x / (std_dv/np.sqrt(n))
    p = 2* stats.t.cdf(-abs(t_c), df=n-1)
    c = 1 - p
    mu_t = ( (x-((t_c*std_error)/np.sqrt(n)) ), (x+((t_c*std_error)/np.sqrt(n)) ))

    return x, std_error, p, c, t_c, mu_t

if __name__ == '__main__' :
    points = [3, -3, 3, 12, 15, -16, 17, 19, 23, -24, 32]
    mean, std_err, t_c, mu_t = confidence_test1(points,0.95)
    print("Problem 2 Question 1 answers:")
    print("mean: {} standard error: {}".format(mean,std_err))
    print("t-value: {}".format(t_c))
    print("confidence interval for 95% is {}\n".format(mu_t))
    
    mean, std_err, t_c, mu_t = confidence_test1(points,0.90)
    print("Problem 2 Question 2 answers:")
    print("mean: {} standard error: {}".format(mean,std_err))
    print("t-value: {}".format(t_c))
    print("confidence interval for 90% is {}\n".format(mu_t))
    
    mean, std_err, z_c, mu_z = confidence_test2(points,0.95)
    print("Problem 2 Question 3 answers:")
    print("mean: {} standard error: {}".format(mean,std_err))
    print("z-value: {}".format(z_c))    
    print("confidence interval: {}\n".format(mu_z))

    mean, std_err, p, c, t_c, mu_t = confidence_test3(points,0.95)
    print("Problem 2 Question 4 answers:")
    print("mean: {} std_err: {}".format(mean,std_err))
    print("t-value: {}".format(t_c))  
    print("p-value: {}".format(p))     
    print("confidence value: {}".format(c))
    print("confidence interval: {}\n".format(mu_t))

