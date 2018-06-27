'''
Created on 9. 3. 2018

@author: Aneta Medunova
'''
from configuration import true
  
def filter_epochs_target(epochs, events, target, instruction):
    """
    This function looking for stimuls which are entered into the entrance at as parameter target.
    ___________________________
    Return:
    epochs_t:
        An array of epochs that are only target (nontarget)
    """
    event_id_instr = ['{\'door\': 1}','{\'window\': 2}','{\'radio\': 3}','{\'lamp\': 4}','{\'phone\': 5}','{\'tv\': 6}','{\'food\': 7}','{\'toilet\': 8}','{\'helps\': 9}']
    event_id_matrix = ['{\'R1\': 1}','{\'R2\': 2}','{\'R3\': 3}','{\'C1\': 4}','{\'C2\': 5}','{\'C3\': 6}']

    epochs_t = []
    target = int(target)
    target -= 1
        
    for j in range(len(epochs)):
    
        epoch_event_id = epochs[j].event_id
        epoch_event_id = str(epoch_event_id)
       
        if(instruction == true):
            target_event = event_id_instr[target]
        else:
            target_event = event_id_matrix[target]
        
        if(epoch_event_id == target_event):
            epochs_t.append(epochs[j])
    
    return epochs_t
