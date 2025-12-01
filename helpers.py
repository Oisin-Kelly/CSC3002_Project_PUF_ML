import numpy as np

def get_XY(data):
    X, Y = [], []
    
    for d in data:
        Y.append(d.pop())
        X.append(d)
    
    return np.asarray(X), np.asarray(Y)

def transform_features(C):
    # convert into 1 and -1
    C = 2. * C - 1
    C = np.fliplr(C)
    C = np.cumprod(C, axis=1,  dtype=np.int8)
    return C

def save_to_memmap(array, filepath, dtype = "int8"):
    memmap_array = np.memmap(filepath, dtype=dtype, mode='w+', shape=array.shape)
    memmap_array[:] = array[:] 
    memmap_array.flush()
