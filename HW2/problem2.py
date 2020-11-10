import numpy as np
import matplotlib.pyplot as plt


def norm_histogram(hist):
    """
    takes a histogram of counts and creates a histogram of probabilities
    :param hist: list
    :return: list
    """
    total_count = sum(hist)
    prob = list()

    for freq in hist:
        prob.append(freq/total_count)

    return prob    

def computeJ(histo, width):
    """
    takes histogram of counts, uses norm_histogram to convert to probabilties, t    then calculates computeJ for one bin width
    :param histo: list 
    :param width: float
    :return: float
    """ #J = 2/(m-1)w - ((m+1)/(m-1)w)(p1+p2+...+pn)
    prob = norm_histogram(histo)
    m = sum(histo)
    p_ss = sum(map(lambda x: x*x, prob)) #square of sum of prob
    result = (2/((m-1)*width)) - (((m+1)/((m-1)*width))*p_ss)    

    return result


def sweepN (data, minimum, maximum, min_bins, max_bins):
    """
    find the optimal bin
    calculate computeJ for a full sweep [min_bins to max_bins]
    :param data: list
    :param minimum: int
    :param maximum: int
    :param min_bins: int
    :param max_bins: int
    :return: list
    """
    result = list()
    plt.hist(data, 5,(minimum,maximum))[0],(maximum-minimum)/5
    for i in range(min_bins,max_bins+1):
        #create histogram
        hist = plt.hist(data, i, (minimum,maximum))[0]
        w = (maximum - minimum)/i
        result.append(computeJ(hist,w))
    
    return result


def findMin (l):
    """
    generic function that takes a list of numbers and returns smallest number in that list its index.
    return optimal value and the index of the optimal value as a tuple.
    :param l: list
    :return: tuple
    """
    result = min(l)
    i = l.index(result)
    return (result, i)


if __name__ == '__main__':
    data = np.loadtxt('input.txt')  # reads data from inp.txt
    lo = min(data)
    hi = max(data)
    bin_l=1
    bin_h=100
    js = sweepN(data, lo, hi, bin_l,bin_h)

    # the values bin_l and bin_h represent the lower and higher bound of the range of bins.
    # They will change when we test your code and you should be mindful of that.
    
    print(findMin(js))

    # Include code here to plot js vs. the bin range