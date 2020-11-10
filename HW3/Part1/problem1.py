def composeMap(fun1, fun2, l) :
    """
    Returns a new list r where each element in r is fun2(fun1(i)) for the
    corresponding element i in l
    :param fun1: function
    :param fun2: function
    :param l: list
    :return: list
    """
    #Fill in
    temp_r = [fun1(x) for x in l]
    r = [fun2(x) for x in temp_r]
    return r

#Returns a new list r where each element in r is fun(fun(i)) for the 
#corresponding element in l
def doubleMap(fun, l) :
    """
    Returns a new list r where each element in r is fun(fun(i)) for the
    corresponding element i in l
    :param fun: function
    :param l: list
    :return: list
    """
    #Fill in
    return composeMap(fun1=fun, fun2=fun, l=l)


def compose(fun1, fun2) :
    #Fill in
    """
    Returns a new function f. f should take a single input i, and return
    fun2(fun1(i))
    :param fun1: function
    :param fun2: function
    :return: function
    """
    def retfun(i) :
        # Fill in
        return fun2(fun1(i))
    return retfun


def repeater(fun, num_repeats) :
    """
    Returns a new function f. f should take a single input i, and return
    fun applied to i num_repeats times. In other words, if num_repeats is 1, f
    should return fun(i). If num_repeats is 2, f should return fun(fun(i)). If
    num_repeats is 0, f should return i.
    :param fun: function
    :param num_repeats: int
    :return: function
    """
    def retfun(x) :
        #Fill in
        result = x
        for _ in range(num_repeats):
            result = fun(result)
        return result
    return retfun
    
if __name__ == '__main__' :
    def test1(x) :
        return x * 2
    
    def test2(x) :
        return x - 3
        
    data = [2, 5, -10, -7, -7, -3, -1, 9, 8, -6]

    print(composeMap(test1, test2, data))
    print(composeMap(test2, test1, data))
    print(doubleMap(test1, data))
    
    f1 = compose(test1, test2)
    
    print(f1(4))
    
    print(list(map(f1, data)))
    
    f2 = compose(test2, test1)

    print(f2(4))

    print(list(map(f2, data)))
    n = 1
    z = repeater(test1, 0)
    once = repeater(test1, n)
    twice = repeater(test1, 2)
    thrice = repeater(test1, 3)
    
    print("repeat 0 times: {}".format(z(5)))
    n = 2
    print("repeat 1 time: {}".format(once(5)))
    print("repeat 2 times: {}".format(twice(5)))
    print("repeat 3 times: {}".format(thrice(5)))