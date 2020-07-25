import numpy as np
from scipy.spatial.distance import cdist

def ismembertol_custom(A, B, tol=0.0):
    '''
    ismembertol(A,B,TOL) returns an array of the same size as A
    containing logical 1 (true) where the elements of A are within the
    tolerance TOL of the elements in B; otherwise, it contains logical 0
    (false). ismembertol scales the TOL input based on the magnitude of the
    data, so that two values u and v are within tolerance if:
      abs(u-v) <= TOL*max(abs([A(:);B(:)]))

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    B : TYPE
        DESCRIPTION.
    tol : TYPE
        DESCRIPTION.

    Returns
    -------
    bools : TYPE
        DESCRIPTION.

    '''
    assert len(np.shape(A.squeeze())) <= 1 and len(np.shape(B.squeeze())) <= 1

    a = A.reshape(-1,1)
    b = B.reshape(-1,1)

    d = cdist(a,b)
    # print(d[:10,:10])
    # print(tol*np.nanmax(np.abs(np.concatenate((a, b)))))
    ind = np.sum(d <= tol*np.nanmax(np.abs(np.concatenate((a, b)))),axis=1) > 0

    return ind.astype(int)
# A=np.array([1,2,3,1,5,63,22,2,1,3,5,44,3,1,9,4.5,3,2,4.7])
# B=np.array([0.3,20,1,3,5,3,6])

# print(ismembertol_custom(A,B,1e-10))
