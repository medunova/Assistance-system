"""
Created on Thu Mar  8 21:53:19 2018

@author: Anet
"""
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from configuration import reshape_range

def solve_lda (x_train,y_train,x_test):
    """
    This function will classifies features vectors by LDA
    """
    x_train = np.reshape(x_train,reshape_range)

    clf = LinearDiscriminantAnalysis()
    clf.fit(x_train, y_train)
    LinearDiscriminantAnalysis(solver='svd', store_covariance=False, tol=0.0001)
    
    
    x_event = []
    for i in range(len(x_test)):
        
        x_test[i] = np.reshape(x_test[i],(1,-1))
        x_event.append(clf.predict(x_test[i]))
      
    x_event = np.array(x_event)
    return x_event
    

