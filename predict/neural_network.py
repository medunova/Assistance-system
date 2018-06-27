'''
Created on 23. 3. 2018

@author: Anet
'''

from keras.layers import Dense, Dropout
from keras.models import load_model

import numpy as np
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from configuration import false, true, zero, reshape_range
import display.plot_training as plot_training

def train_nn(x,y, model):
    """
    This function creates the model and its training

    """
    x_train = []
    x_valid = []
    y_train = []
    y_valid = []
    
#     separate validation set and training set
    for i in range(len(x)):
        
        if(i%5 == zero):
            x_valid.append(x[i])
            if(y[i] == false):
                y_valid.append(false)
            else:
                y_valid.append(true)
            
        else:
            x_train.append(x[i])
            if(y[i] == false):
                y_train.append(false)
            else:
                y_train.append(true)

   
    x_train = np.reshape(x_train,reshape_range)
    x_valid = np.reshape(x_valid,reshape_range)
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    
    print("\n\n\n")
    print("Do you want to show figure od features vectors to train?")
    show = input("Show figures? 1/0: ")
    if(show == '1'):
        plt.plot(np.mean(x_train[y_train==1], axis=0))
        plt.plot(np.mean(x_train[y_train==0], axis=0))
        plt.title('Training epochs')
        plt.legend(['target', 'non-target'], loc='lower left')
    
        plt.show()
    
        plt.plot(np.mean(x_valid[y_valid==1], axis=0))
        plt.plot(np.mean(x_train[y_train==0], axis=0))
        plt.title('Validation epochs')
        plt.legend(['target', 'non-target'], loc='lower left')
    
        plt.show()
        
    input_dim = x_train.shape[1]
    
    
#     model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, input_dim=input_dim, activation='sigmoid'))
    
    model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    model.summary()

    earlyStopping = EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='auto')

    model_history = model.fit(x_train, y_train, epochs=300, batch_size=4,shuffle = True, callbacks=[earlyStopping], validation_data=(x_valid,y_valid))
    loss_and_metrics = model.evaluate(x_valid, y_valid, batch_size=4)
   
    model.save('mymodel.h5')
    plot_training.display_history(model_history)
    print(loss_and_metrics)
 
    return model


def solve_nn(x_predict, model):
    """
    This function will classifies features vectors by NN
    Parameters:
        x_predict:
            Data set to predict
        trained: Boolean
            If was trained new model or load save model
    Return:
    classes:
        Evaluation of features vectors.
    """

    x_predict = np.reshape(x_predict,reshape_range)
     
    classes = model.predict(x_predict, batch_size=8)
    
    for i in range(len(classes)):
        for j in range(len(classes[i])):
            classes[i][j] = round(classes[i][j])
   
    
    return classes
    

    