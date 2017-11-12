import numpy as np 
#split a set
def partition(a):
    return {c: (a == c).nonzero()[0] for c in np.unique(a)}

#Ent
def entropy(s):
    res = 0
    val, counts = np.unique(s, return_counts = True)
    freqs = counts.astype('float')/len(s)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res


def mutal_information(y, x):
    res = entropy(y)
    # We partition x, acording to attribute value x_i
    val, counts = np.unique(x, return_counts = True)
    freqs = counts.astype('float')/len(x)
    # We calculate a weighted average of the entropy
    #zip() has been transformed from a list into a iterable object
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])
    return res

from pprint import pprint

#set will ignore the repeated number
def is_pure(s):
    return len(set(s)) == 1

def recursive_split(x, y):
    #if there could be no split, just return mutal information
    if is_pure(y):
        return y[0]

    #We get attribute that gives the highest mutal information
    gain = np.array([mutal_information(y, x_attr) for x_attr in x.T])
    selected_arr = np.argmax(gain)

    #If there is no gain at all, nothing has to be done, just return the original set
    if np.all(gain < 1e-6):
        var, count = np.unique(y, return_counts = True)
        return var[np.argmax(count)]

    #We split using the selected attribute
    sets = partition(x[:, selected_arr])
    res = {}
    for k,v in sets.items():
        y_subset = y.take(v, axis = 0)
        x_subset = x.take(v, axis = 0)
    #Nested dictionary to store the selected_arr and the labels
        res[(selected_arr, k)] = recursive_split(x_subset, y_subset)
    return res

#Predict function
def predict(res, x):
    return



#To train 
x1 = [0, 1, 1, 2, 2, 2]
x2 = [0, 0, 1, 1, 1, 0]
y = np.array([0, 0, 0, 1, 1, 0])
X = np.array([x1, x2]).T 
pprint(recursive_split(X, y))
#To test





