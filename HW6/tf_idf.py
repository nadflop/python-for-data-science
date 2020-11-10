from helper import remove_punc
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

#Clean and lemmatize the contents of a document
#Takes in a file name to read in and clean
#Return a list of words, without stopwords and punctuation, and with all words stemmed
# NOTE: Do not append any directory names to doc -- assume we will give you
# a string representing a file name that will open correctly
def readAndCleanDoc(doc) :
    #1. Open document, read text into *single* string
    with open(doc, "r") as f:
        text = f.read()
    #2. Tokenize string using nltk.tokenize.word_tokenize
    #nltk.download('punkt')
    word_token = word_tokenize(text)
    #3. Filter out punctuation from list of words (use remove_punc)
    word_filtered = remove_punc(word_token)
    #4. Make the words lower case
    word_lower = [words.lower() for words in word_filtered]
    #5. Filter out stopwords
    #nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    doc_clean = [words for words in word_lower if words not in stop_words]
    #6. Stem words
    ps = PorterStemmer()
    words = [ps.stem(w) for w in doc_clean]
    
    return words
    
#Builds a doc-word matrix for a set of documents
#Takes in a *list of filenames*
#
#Returns 1) a doc-word matrix for the cleaned documents
#This should be a 2-dimensional numpy array, with one row per document and one 
#column per word (there should be as many columns as unique words that appear
#across *all* documents. Also, Before constructing the doc-word matrix, 
#you should sort the wordlist output and construct the doc-word matrix based on the sorted list
#
#Also returns 2) a list of words that should correspond to the columns in
#docword
def buildDocWordMatrix(doclist) :
    #1. Create word lists for each cleaned doc (use readAndCleanDoc)
    
    #list of words that should correspond to the columns in docword
    wordlist = set() 
    doc_word = list()
    doc_clean = [readAndCleanDoc(doc) for doc in doclist]

    for doc in doc_clean:
        for word in doc:
            #make sure the word is unique
            wordlist.add(word)
    
    wordlist = list(wordlist)
    wordlist.sort()
    
    for doc in doc_clean:
        temp = np.zeros(shape=len(wordlist))
        #for each word in the doc, you ask if it appears 
        #in the doc you're dealing with, and how many times
        for word in doc:
            position = wordlist.index(word)
            temp[position] += 1
        doc_word.append(temp)
    
    docword = np.array(doc_word)
            
    #2. Use these word lists to build the doc word matrix
    
    return docword, wordlist
    
#Builds a term-frequency matrix
#Takes in a doc word matrix (as built in buildDocWordMatrix)
#Returns a term-frequency matrix, which should be a 2-dimensional numpy array
#with the same shape as docword
def buildTFMatrix(docword) :
    #fill in
    #tf(t,d) = count of t in d/ no of words in d
    #[:, np.newaxis] 
    #newaxis used to increase the dimension to one more dimension
    #make it as a column vector by inserting an axis along
    #second dimension
    #axis = 1 since we want to work along the row
    tf = docword / (np.sum(docword,axis=1))[:,None]
    
    return tf
    
#Builds an inverse document frequency matrix
#Takes in a doc word matrix (as built in buildDocWordMatrix)
#Returns an inverse document frequency matrix (should be a 1xW numpy array where
#W is the number of words in the doc word matrix)
#Don't forget the log factor!
def buildIDFMatrix(docword) :
    #fill in
    #idf = log(N/(df+1))
    #N = count of document
    #df = occurence of words in documents (all documents)
    #axis = 0 since we want to find the total occurence of that word in ALL doc
    idf = np.log10(docword.shape[0]/(np.sum(docword>0,axis=0))).reshape(1,docword.shape[1])

    return idf
    
#Builds a tf-idf matrix given a doc word matrix
def buildTFIDFMatrix(docword) :
    #fill in
    #tf-idf = tf(t,d) * log(n/(DF+1))
    tfidf = buildTFMatrix(docword) * buildIDFMatrix(docword)

    return tfidf
    
#Find the three most distinctive words, according to TFIDF, in each document
#Input: a docword matrix, a wordlist (corresponding to columns) and a doclist 
# (corresponding to rows)
#Output: a dictionary, mapping each document name from doclist to an (ordered
# list of the three most common words in each document
def findDistinctiveWords(docword, wordlist, doclist) :
    distinctiveWords = dict()
    #fill in
    #you might find numpy.argsort helpful for solving this problem:
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
    
    for doc_name in doclist:
        #slice the elements from the beginning to index 3 (not included)
        #argsort returns the indexes of the sorted array in ascending order
        #so need to reverse the array returned
        indexes = np.argsort(buildTFIDFMatrix(docword)[doclist.index(doc_name),:])[::-1][:3]
        distinctiveWords[doc_name] = np.array(wordlist)[indexes]
        
    
    return distinctiveWords


if __name__ == '__main__':
    from os import listdir
    from os.path import isfile, join, splitext
    
    ### Test Cases ###
    directory='lecs'
    path1 = join(directory, '4_vidText.txt')
    path2 = join(directory, '13_vidText.txt')
    
    # Uncomment and recomment ths part where you see fit for testing purposes
    
    print("*** Testing readAndCleanDoc ***")
    print(readAndCleanDoc(path1)[0:5])
    print("*** Testing buildDocWordMatrix ***") 
    doclist =[path1, path2]
    docword, wordlist = buildDocWordMatrix(doclist)
    
    print(docword.shape)
    print(len(wordlist))
    print(docword[0][0:10])
    print(wordlist[0:10])
    print(docword[1][0:10])
    
    print("*** Testing buildTFMatrix ***") 
    tf = buildTFMatrix(docword)
    print(tf[0][0:10])
    print(tf[1][0:10])
    print(tf.sum(axis =1))
    
    print("*** Testing buildIDFMatrix ***") 
    idf = buildIDFMatrix(docword)
    print(idf[0][0:10])
    
    print("*** Testing buildTFIDFMatrix ***") 
    tfidf = buildTFIDFMatrix(docword)
    print(tfidf.shape)
    print(tfidf[0][0:10])
    print(tfidf[1][0:10])
    
    print("*** Testing findDistinctiveWords ***")
    print(findDistinctiveWords(docword, wordlist, doclist))
    
