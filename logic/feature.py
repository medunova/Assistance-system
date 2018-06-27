'''
Created on 9. 3. 2018

@author: Aneta Medunova
'''
import numpy as np
import math
from configuration import samples_to_average
from configuration import electrods_count
from configuration import min_of_sample
from configuration import max_of_sample


def feature_vector(epochs):
    """
    Function will create features vectors from entered epochs.
    ______________________________________
    
    Return:
    features:
        features vector.
    """
#    Baseline correction average of first 100   
#    print(np.mean((epochs.get_data()[0][0][:100])))
    
    features = []    

# five electrods 
    for i in range(electrods_count):
        
        for j in range(min_of_sample, max_of_sample, samples_to_average):
            from_index = j
            to_index = j + samples_to_average
            features.append(np.mean((epochs.get_data()[0][i][from_index:to_index])))
     
    counterPower = 0
    
    for k in range(len(features)):
        number = features[k]
        counterPower = counterPower + (number**2)
      
    counterPower = math.sqrt(counterPower)

    for l in range(len(features)):
              
        features[l]/counterPower  

    return features

