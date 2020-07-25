import numpy as np

def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def ind2sub(array_shape, ind):
    '''
    ind2sub is used to determine the equivalent subscript values
    corresponding to a given single index into an array.
 
    [I,J] = ind2sub(SIZ,IND) returns the arrays I and J containing the
    equivalent row and column subscripts corresponding to the index
    matrix IND for a matrix of size SIZ.  
    For matrices, [I,J] = ind2sub(SIZE(A),FIND(A>5)) returns the same
    values as [I,J] = FIND(A>5).
 
    [I1,I2,I3,...,In] = ind2sub(SIZ,IND) returns N subscript arrays
    I1,I2,..,In containing the equivalent N-D array subscripts
    equivalent to IND for an array of size SIZ.


    Parameters
    ----------
    array_shape : TYPE
        DESCRIPTION.
    ind : TYPE
        DESCRIPTION.

    Returns
    -------
    rows : TYPE
        DESCRIPTION.
    cols : TYPE
        DESCRIPTION.

    '''
    if len(array_shape) == 1:
        assert array_shape[0] >  max(ind)
        return np.array(list(np.unravel_index(ind, array_shape)))[0]
    
    elif len(array_shape) == 2: 
        indices = np.array(list(np.unravel_index(ind, array_shape)))
        i = indices[0, :]
        j = indices[1, :]
        
        return i, j
    
    elif len(array_shape) == 3:
        i = indices[0, :]
        j = indices[1, :]
        k = indices[2, :]
        
        return i, j, k
    else:
        raise NotImplemented
        
def main():
    ind = [2, 3, 4, 5, 9]
    sz = (10,)
    i = ind2sub(sz, ind)
    
    print(i)

if __name__ == "__main__":
    main()