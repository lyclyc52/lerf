import torch
import torch.nn.functional as F
from torch import tensor 


def pt(obj):
    if type(obj) is list or type(obj) is tuple:
        shape_list = []
        for i in obj:
            shape_list.append(pt(i))
        return shape_list
    else:
        return obj.shape
        

def print_shape(obj):
    shape = pt(obj)
    print(shape)
    
    
