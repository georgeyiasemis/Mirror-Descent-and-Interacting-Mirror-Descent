import numpy as np

def triuToFullIdx(coords_i, coords_j, tri_i, tri_j, A):

    n = A.shape[0]

    map = np.zeros((coords_i.shape[0]))
    triIdx = {}

    for w, (i,j) in enumerate(zip(tri_i, tri_j)):
        triIdx[n*i + j] = w

    for w, (i,j) in enumerate(zip(coords_i, coords_j)):

        if i < j:
            map[w] = triIdx[n*i + j]
        else:
            map[w] = triIdx[n*j + i]
    return map.astype(int)

def precomputeDists(tri_i, tri_j, map, y):

    m = tri_i.shape[0]
    d = np.zeros((m,))

    for w, (i,j) in enumerate(zip(tri_i, tri_j)):
        d[w] = ((y[i,:] - y[j,:]) ** 2).sum()
    return d[map]

def accumarray(subs, val, sz):

    A = np.zeros(sz)
    for i in range(A.shape[0]):
        A[i] = val[subs==i].sum()
    return A
if __name__ == '__main__':

    subs = np.array([1,2,4,2,4]) - 1
    sz=(4,1)
    val = np.array([101,102,103,104,105])
    print(accumarray(subs,val,sz))
    # A = np.eye(10)
    # A[1,3] = 1
    # A[3,1] = 1
    # A[5,2] = 1
    # A[2,5] = 1
    # A[9,3] = 1
    # A[3,9] = 1
    # coords_i, coords_j = np.where(A)
    # print(coords_i+1)
    # print(coords_j+1)
    #
    # tri_i, tri_j = np.where(np.triu(A))
    # print(tri_i+1)
    # print(tri_j+1)
    # print(triuToFullIdx(coords_i, coords_j, tri_i, tri_j, A) + 1)
