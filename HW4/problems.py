import re

def problem1(searchstring):
    """
    Match phone numbers.
    :param searchstring: string
    :return: True or False
    """
    no = re.search(r'^\d{3}\-\d{3}\-\d{4}|\(\d{3}\)\s\d{3}-\d{4}|^\d{3}\-\d{4}', searchstring)
    if no != None:
        #print(no.group())
        return True
    return False
        
def problem2(searchstring):
    """
    Extract street name from address.
    :param searchstring: string
    :return: string
    """
    address = re.search(r'\d+\s([A-Za-z\s]+)\s(Rd|Dr|Ave|St)',searchstring)
    if address != None:
        #print(address.group(1))
        return address.group(1)
    
def problem3(searchstring):
    """
    Garble Street name.
    :param searchstring: string
    :return: string
    """
    address = re.search(r'\d+\s([A-Za-z\s]+)\s(Rd|Dr|Ave|St)',searchstring)
    if address != None:
        s = address.group(1)[::-1]
        #print(s)
        x = re.sub(address.group(1),s,searchstring)
        #print(x)
        return x


if __name__ == '__main__' :
    print(problem1('765-494-4600')) #True
    print(problem1(' 765-494-4600 ')) #False
    print(problem1('(765) 494 4600')) #False
    print(problem1('(765) 494-4600')) #True    
    print(problem1('494-4600')) #True
    
    print(problem2('The EE building is at 465 Northwestern Ave.')) #Northwestern
    print(problem2('Meet me at 201 South First St. at noon')) #South First
    print(problem2('Pete lives on 123 Have No Clue Ave.')) #South First

    print(problem3('The EE building is at 465 Northwestern Ave.'))
    print(problem3('Meet me at 201 South First St. at noon'))