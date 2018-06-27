'''
Created on 4. 4. 2018

@author: Anet
'''
import matplotlib.pyplot as plt

def display_history(hist):
    """
    This function just plot history of training model
    
    Parameters
    ----------
    hist:
        history of model training
        
    Return
    """
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()