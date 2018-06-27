"""
Created on Wed Feb 14 22:22:09 2018

@author: Anet
print(__doc__)
"""
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)
import sklearn; print("Scikit-Learn", sklearn.__version__)
import keras; print("Keras", keras.__version__)

import mne; print("MNE", mne.__version__)
import numpy as np
from mne import io
import platform; print(platform.platform())
import logic.epochs_methods as epoch_met
import i_o.input_test_data as load_file_names
import predict.lda as lda
from builtins import print
import logic.mix_data_x_y as mix
import predict.neural_network as neural_network
from keras.models import Sequential
from configuration import high_filter_frequency
from configuration import low_filter_frequency  
from configuration import instruction_files_to_pred, matrix_files_to_pred
from configuration import event_id_instruction
from configuration import event_id_matrix
from configuration import epoch_tmax
from configuration import epoch_tmin
from configuration import baseline_max
from configuration import baseline_min
from configuration import instruction_files_count,colors,linestyles,linestyles_matrix,colors_matrix
from configuration import increase
from configuration import conditions,conditions_matrix
from configuration import chan
from configuration import first_item
from configuration import second_item
from configuration import false, true, saved_model_path
from display import print_guess
from logic import feature_vector
from numpy.random import seed
from keras.models import load_model
from datashape.coretypes import null
import os
seed(1)


#turn off log
mne.set_log_level('ERROR')
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"]="4"

# Set path to raw data folder 

DATA_FOLDER =os.getcwd()+'/raw_data/'
print(DATA_FOLDER)


# Set EEG event list - instruction



##############################################
#
#             Data loading
#
##############################################

# mapu, ve ktere jsou ulozeny nazvy trenovacich souboru a jejich targetove/non-targetove znacky
files_training_map = load_file_names.load_training_data_names()

# mapu, ve ktere jsou ulozeny nazvy testovacich souboru a jejich targetove nazvy
files_testing_map = load_file_names.load_predicting_data_names()



data_training_count = len(files_training_map)
data_predicting_count = len(files_testing_map)

##############################################
#
#            Loading data to train_nn
#
##############################################
raw = []
for i in range(data_training_count):
    path = DATA_FOLDER + (files_training_map[i][first_item])
    raw.append(io.read_raw_brainvision(vhdr_fname=path, preload=True))
    raw[i].filter(low_filter_frequency,high_filter_frequency)

##############################################
#
#             Loading data to predict
#
##############################################
raw_to_predict = []
true_prediction = []
for i in range(data_predicting_count):
    path = DATA_FOLDER + (files_testing_map[i][first_item])
    true_prediction.append(files_testing_map[i][second_item])
    true_prediction[i] = true_prediction[i].strip()
    raw_to_predict.append(io.read_raw_brainvision(vhdr_fname=path, preload=True))
    raw_to_predict[i].filter(low_filter_frequency,high_filter_frequency)

    
##############################################
#
#             Epochs creating
#
##############################################
    
# Vytvori epochy pro klasifikaci
event_to_predict = []
epochs_to_predict = []
for i in range(data_predicting_count):
    event_to_predict.append(raw_to_predict[i]._events)
 
    if(i < instruction_files_to_pred):

        epochs_to_predict.append(mne.Epochs(raw_to_predict[i],event_to_predict[i], event_id=event_id_instruction, tmin=epoch_tmin, tmax=epoch_tmax,baseline=(baseline_min, baseline_max), preload=True))
    else:
        epochs_to_predict.append(mne.Epochs(raw_to_predict[i],event_to_predict[i], event_id=event_id_matrix, tmin=epoch_tmin, tmax=epoch_tmax,baseline=(baseline_min, baseline_max), preload=True))
    
    

# Plot raw data bylo 40 a 6
"""
for i in range(instruction_files_count): 
   raw[i].plot(lowpass=50, n_channels=5,title="Raw data", block=False)
"""
##############################################
#
#             
#
##############################################




# Set color of events
"""
for i in range(instruction_files_count):
    mne.viz.plot_events(event_to_predict[i],raw_to_predict[i].info['sfreq'], raw_to_predict[i].first_samp, color=color)
"""

#extract epochs

events_train = []
epochs = []
epochs_targets = []
epochs_non_targets = []



# Vytvori epochy, z vytvorenych Epoch potom vybere ty targetove a ulozi je do epochs_target

for i in range(data_training_count):
    events_train.append(raw[i]._events)
    
    if(i < instruction_files_count):
        epochs.append(mne.Epochs(raw[i],events_train[i], event_id=event_id_instruction, tmin=epoch_tmin, tmax=epoch_tmax,baseline=(baseline_min, baseline_max), preload=True))
        instruction = 1
    else:
        epochs.append(mne.Epochs(raw[i],events_train[i], event_id=event_id_matrix, tmin=epoch_tmin, tmax=epoch_tmax,baseline=(baseline_min, baseline_max), preload=True))
        instruction = 0
        
    epochs_targets.append(epoch_met.filter_epochs_target(epochs[i], events_train[i], files_training_map[i][1], instruction))  
    epochs_non_targets.append(epoch_met.filter_epochs_target(epochs[i], events_train[i], files_training_map[i][2], instruction))  
    

"""
for i in range(instruction_files_count):
    mne.viz.plot_epochs(epochs_to_predict[i], title="Epoch to predict: ", n_epochs=8, block=False)
"""

# epochs.plot(title="Events epochs", n_epochs=(len(epochs.events)),event_colors=color)
# mne.viz.plot_epochs(epochs, title="Events epochs", n_epochs=15,event_colors=color)



# Create evoked structure

evoked_dict = [[]]
# jen pro instruction
for i in range(instruction_files_to_pred):
    evoked_dict.append('')
    evoked_dict[i] = dict()
    for condition in conditions:
        evoked_dict[i][condition] = epochs_to_predict[i][condition].average()
       
evoked_dict_matrix = [[]]
# jen pro instruction
for i in range(matrix_files_to_pred):
    evoked_dict_matrix.append('')
    evoked_dict_matrix[i] = dict()
    for condition in conditions_matrix:
        evoked_dict_matrix[i][condition] = epochs_to_predict[instruction_files_to_pred+i][condition].average()

print("\n\n\n")
print("Do you want to show epochs of individual incentives in figures?")
show = input("Show figures? 1/0: ")
if(show == '1'):
# Plot instructions 
    for i in range(instruction_files_to_pred):
        mne.viz.plot_compare_evokeds(evoked_dict[i], title="ERP chart instructions", colors=colors, linestyles=linestyles, gfp=False)
# Plot matrix
    for i in range(matrix_files_to_pred):
        mne.viz.plot_compare_evokeds(evoked_dict_matrix[i], title="ERP chart matrix", colors=colors_matrix, linestyles=linestyles_matrix, gfp=False)
 

# Extrakce priznaku

labels = epochs[0].events[:, -1]

#feature extraction
    
target_features = []
non_target_features = []
x = []


test_sample_count = 5


y = []


# Prepare data to training 
target_nontarget_epochs = epochs_targets + epochs_non_targets


for i in range(len(target_nontarget_epochs)):
    #count of target epochs     
    for j in range(len(target_nontarget_epochs[i])):
    
        pick_epochs = target_nontarget_epochs[i][j].pick_channels(chan)
        x.append(feature_vector(pick_epochs))
        if(i < epochs_targets.__len__()):
            y.append(true)
        else:
            y.append(false)


# Prepare data to predict 
x_pred = []

y = np.array(y)

for i in range(data_predicting_count):
    x_pred.append([])
    for j in range(len(epochs_to_predict[i])):
        pick_epoch_to_predict = epochs_to_predict[i][j].pick_channels(chan)
        x_pred[i].append(feature_vector(pick_epoch_to_predict))




mix.mix_data(x, y)

# X = np.reshape(X,(-1, 100))

##############################################
#
#             Predicting
#
##############################################

#plotting means of training data
#plt.plot(np.mean(X[y==1], axis=0))
#plt.plot(np.mean(X[y==0], axis=0))
#plt.show()


# plotting tests epochs
# for i in range(len(X_pred)):
#     name = str(i)+'.png'
#     plt.plot(X_pred[i])
#     plt.savefig(name)



x_event_lda = []
x_event_neural = []
print("\n\n\n\n")
print("You can choose between a new model training or choose already trained model. \nWhen choosing new model training there is a risk \nof bad treined model due random setting of parameters.")
print("If you want to load model from file: 1")
print("If you want to train_nn new model  : 0")
print()

model_load = input("Load model? 1/0: ")
if(model_load == '1'):
    model = load_model(saved_model_path) 
       
else:
    if(model_load == '0'):
        model = Sequential()
        model = neural_network.train_nn(x, y, model)
    else:
        print("Invalid option")


for i in range(data_predicting_count):
    x_event_lda.append(lda.solve_lda(x,y,x_pred[i]))
    x_event_neural.append(neural_network.solve_nn(x_pred[i],model))

print("\n\n")
print("The test results are as follows: ")
for i in range(data_predicting_count):
    print()
    print("##########################################")
    print()
  
  
    if(i < instruction_files_to_pred):
        
        print(i+increase,".) Expected solve_lda: ",true_prediction[i])
        print()
        
        instruction = 1
        print("LDA: ")
        print_guess(x_event_lda[i], epochs_to_predict[i], true_prediction[i],null,instruction)
        print("Neural network: ")
        print_guess(x_event_neural[i], epochs_to_predict[i], true_prediction[i],null,instruction)
        
    else:
        if(i%2==0):
            print(i+increase,".) Expected solve_lda: ",true_prediction[i],true_prediction[i+increase])
            
            print()
        
            instruction = 0
            print("LDA: ")
            print_guess(x_event_lda[i], epochs_to_predict[i], true_prediction[i],true_prediction[i+increase],instruction)
            print("Neural network: ")
            print_guess(x_event_neural[i], epochs_to_predict[i], true_prediction[i],true_prediction[i+increase],instruction)
        
    print()




