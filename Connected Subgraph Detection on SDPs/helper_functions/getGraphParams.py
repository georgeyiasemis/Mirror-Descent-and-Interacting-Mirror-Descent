import numpy as np

def getGraphParams(Adj, p):
    p = p-1
    n = Adj.shape[0]
    # Unnormalized graph Laplacian
    d = np.sum(Adj, axis=1)
    D = np.diag(d)
    L = D - Adj
    
    rows = 0
    DI = np.zeros((1, n))
    for i in range(n-1):
        for j in range(i+1, n):
            if Adj[i,j] == 1.0:
                DI[rows,i] = 1
                DI[rows,j] = -1
                rows += 1
    
    # Adjacency List #
    AL = {}
    for i in range(n):
        AL[i] = np.where(Adj[i,:] == 1)[0].astype(int)
    
    ## compact primal optimization;  fn: node indicator; fe: edge indicator
    # pre-processing,  must be done if anchor is changed
    DI_p = DI[np.abs(DI[:,p]) == 0,:]
    params = {}
    params['m'] = DI_p.shape[0]
    
    DI_star = - np.eye(n)
    
    DI_star[:,p] = DI_star[:,p] + np.ones((1,n))
    Np = np.zeros((n))
    
    Np[np.where(Adj[p,:] > 0)[0].astype(int)] = 1
    DI_pg = np.matmul(np.diag(Np), DI_star)
  
    params['DI_M'] = np.concatenate((DI_star, DI_p))
    params['DI_AoM'] = np.concatenate((DI_pg, DI_p))
    params['DI_p1'] = DI_p > 0.5
    params['DI_p2'] = DI_p < - 0.5
    
    return params

if __name__ == '__main__':
    print(getGraphParams(np.eye(100), 10))