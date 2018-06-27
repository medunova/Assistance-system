'''
Created on 9. 3. 2018

@author: Aneta Medunova
'''
from configuration import instruction_files_count, instruction_files_to_pred, data_to_test_path, data_to_train_path

def load_training_data_names():
    """
    Function loading names of training data sets.
    """
    training_files = []
#     s = open('../subject_target_train',"r")
    s = open(data_to_train_path,"r")
    count = 0;
    for line in s:
        if(count < instruction_files_count):
            training_files.append(line.split())
            count += 1
        else:
            training_files.append(line.split())
    
    return training_files

def load_predicting_data_names():
    """
    Function loading names of predicting data sets.
    """
    testing_files = []
#     s = open('../subject_target_test',"r")
    s = open(data_to_test_path,"r")
    count = 0
    for line in s:
        if(count < instruction_files_to_pred):
            testing_files.append(line.split())
            count += 1
        else:
            testing_files.append(line.split())

    return testing_files

