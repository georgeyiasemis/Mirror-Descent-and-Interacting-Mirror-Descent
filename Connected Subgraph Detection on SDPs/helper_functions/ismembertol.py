import numpy as np
from scipy.spatial.distance import cdist

def ismembertol(A, B, tol=0.0):
    '''
    ismembertol(A,B,tol) returns an array of the same size as A
    containing logical 1 (true) where the elements of A are within the
    tolerance TOL of the elements in B; otherwise, it contains logical 0
    (false). ismembertol scales the TOL input based on the magnitude of the
    data, so that two values u and v are within tolerance if:
      abs(u-v) <= TOL*max(abs([A(:);B(:)]))

    Parameters
    ----------
    A : ndarray
        1-D array.
    B : ndarry
        1-D array.
    tol : float
        tolerance.

    Returns
    -------
    bools : bool Array
        same size as A.

    '''
    assert len(np.shape(A.squeeze())) <= 1 and len(np.shape(B.squeeze())) <= 1
    tol = tol * max(np.max(A),np.max(B))
    bools = []
    b = np.unique(B)
    if len(np.shape(A.squeeze())) == 0:
        bools.append((np.abs((np.unique(b) - A)) <= tol).any())
    else:
        for elem in A.squeeze():
            bools.append((np.abs((np.unique(b) - elem)) <= tol).any())
    return np.array(bools).astype(int)



def ismembertol_custom(A, B, tol):
    '''
    ismembertol(A,B,TOL) returns an array of the same size as A
    containing logical 1 (true) where the elements of A are within the
    tolerance TOL of the elements in B; otherwise, it contains logical 0
    (false). ismembertol scales the TOL input based on the magnitude of the
    data, so that two values u and v are within tolerance if:
      d <= TOL*max(abs([A(:);B(:)]))
    where d is the pairwise set distances of A and B

    Parameters
    ----------
    A : ndarray
        1-D array.
    B : ndarry
        1-D array.
    tol : float
        tolerance.

    Returns
    -------
    bools : bool Array
        same size as A.

    '''
    assert len(np.shape(A.squeeze())) <= 1 and len(np.shape(B.squeeze())) <= 1
   
    a = A.reshape(-1,1)
    b = B.reshape(-1,1)
    
    d = cdist(a,b)
    
    ind = np.sum(d <= tol*np.nanmax(np.abs(np.concatenate((a, b)))),axis=1) > 0 
    
    return ind.astype(int)

