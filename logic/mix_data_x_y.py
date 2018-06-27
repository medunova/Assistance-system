'''
Created on 11. 3. 2018

@author: Aneta Medunova
'''
import random

def mix_data(x, y):
    """
    The function randomly snaps the data set and evaluates it so that no connection is broken.
    """
    if(len(x) != len(y)):
        print("Data not consistence: ")
    
    else:
        for i in range(1,len(x)):
            j = random.randrange(0, len(x),2);
            ax = x[i - 1];
            x[i - 1] = x[j];
            x[j] = ax;
            
            ay = y[i - 1];
            y[i - 1] = y[j];
            y[j] = ay;
            
            
            