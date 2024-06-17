import time
import numpy as np 

class info:

    def iter(object) -> str:
    
        if hasattr(object, '__iter__'):
            if hasattr(object, '__next__'):
                return('Iterator')
            else:
                return('Iterable')
        
        if hasattr(object, '__getitem__'):
            return('Iterable')
    
        return('Not Iterable')
    
    def func_time(func) -> str:
        
        start_time = time.time()
        result = func()
        end_time = time.time()
        
        return('\ntime : %.3f ms' %(round((end_time - start_time)*1000, 3)))


start_time = time.time()

def count_elements(arr, condition):
    mask = condition(arr)
    count = np.sum(mask)
    return count
