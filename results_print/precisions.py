'''
Created on 26. 6. 2018

@author: Anet
'''

from configuration import false
from configuration import expected_targets_matrix,\
    expected_targets_instructions, hundred, rounded, zero


"""
"""
def accuracy (mapResults, true_label_first, true_label_second, number_of_all, instruction):
    
    if(instruction == false):
        true = count_results_matrix(mapResults)
        true_positive = mapResults.count(true_label_first) + mapResults.count(true_label_second)
        
    else:
        true = count_results_instruction(mapResults)
        true_positive = mapResults.count(true_label_first)

    
    false_positive = true - true_positive
    true_negative = number_of_all - false_positive - true_positive
    accuracy = (true_positive + true_negative)/number_of_all
    
    accuracy_round = round((accuracy*hundred), rounded)
    
    print("Accuracy: ",accuracy_round,"%")
    
"""
"""
def precision(mapResults,true_label_first, true_label_second, instruction):
    
    if(instruction == false):
        true = count_results_matrix(mapResults)
        true_positive = mapResults.count(true_label_first) + mapResults.count(true_label_second)
        
    else:
        true = count_results_instruction(mapResults)
        true_positive = mapResults.count(true_label_first)
    
    
    false_positive = true - true_positive
   
    if((true_positive + false_positive) != zero):
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = zero
        
    precision_round = round((precision*hundred),rounded)
    print("Precision is: ",precision_round,"%")

"""
"""
def recall(mapResults,true_label_first, true_label_second, instruction):
    
    if(instruction == false):
        expected_targets = expected_targets_matrix
        true_positive = mapResults.count(true_label_first) + mapResults.count(true_label_second)
    else:
        expected_targets = expected_targets_instructions
        true_positive = mapResults.count(true_label_first)

    false_negative = expected_targets - true_positive
    
    if((true_positive + false_negative) != zero):
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = zero
        
    recall_round = round((recall*hundred),rounded)
    
    print("Recall is: ",recall_round,"%")
    
    
"""

"""
def count_results_matrix(mapResults):
    
    count = mapResults.count('R1') + mapResults.count('R2') + mapResults.count('R3') + mapResults.count('C1') + mapResults.count('C2') + mapResults.count('C3')
    return count
    
def count_results_instruction(mapResults):
    
    count = mapResults.count('door') + mapResults.count('window') + mapResults.count('radio') + mapResults.count('lamp') + mapResults.count('phone') + mapResults.count('tv') + mapResults.count('food') + mapResults.count('toilet') + mapResults.count('helps')
    return count
    
    
    
    
    
    
    
    