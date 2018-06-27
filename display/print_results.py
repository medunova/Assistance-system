from results_print import precisions
from configuration import true
from configuration import mark_length
from configuration import labels
from configuration import labels_matrix
from configuration import mark_length_matrix

def print_guess(guess_results, epochs_to_predict, true_label_first,true_label_second, instruction):
    """
    This function call other functions for procentual result print - accuracy, precision, recall
    
    Parameters
    ----------
    guess_results:
    epochs_to_predict:
    true_label_first:
    true_label_second:
    instruction:
        
    Return
    """  
    
    mapResults = []
    for i in range(len(guess_results)):
        if(guess_results[i] == true):
            result = ''
            result = ''.join(epochs_to_predict[i].event_id)
            mapResults.append(result)
    
    if(instruction == 1):
        for j in range(mark_length):
#             print(config.labels[j]," :",mapResults.count(config.labels[j]))
            print ('%-8s : %3d' % (labels[j], mapResults.count(labels[j])))
           
    else:
        for j in range(mark_length_matrix):
            print(labels_matrix[j]," :",mapResults.count(labels_matrix[j]))

    print()

  
    precisions.accuracy(mapResults,true_label_first,true_label_second, len(epochs_to_predict),instruction)
    precisions.precision(mapResults, true_label_first,true_label_second,instruction)
    precisions.recall(mapResults, true_label_first,true_label_second,instruction)
    
    print()


    
    