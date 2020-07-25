import numpy as np
import sys
sys.path.insert(0, '../')
from helper_functions.load_mat import load_mat
from helper_functions.ismembertol import ismembertol, ismembertol_custom
import networkx as nx
import matplotlib.pyplot as plt
rng = np.random.default_rng()


xval = 4
nSamp = 1
pop_vec, yy_1, cons, S = load_mat()

n = len(cons)

l_0 = 5e-5
lam_0 = pop_vec * l_0
l_1 = xval * l_0
lam_1 = lam_0
lam_1[S] = pop_vec[S] * l_1

yy_0 = rng.poisson(lam_0, (lam_0.shape[0], nSamp))
yy_1 = rng.poisson(lam_1, (lam_1.shape[0], nSamp))

Adj = np.zeros((n, n))
yy = np.empty((n, nSamp))
fLat, fLon = np.zeros((n,1)), np.zeros((n,1))
tol = 1e-10
for i in range(n):
    cons[i]['rate'] = yy_1[i,:]/cons[i]['pop']
    yy[i] = cons[i]['rate']
    cons[i]['diseased'] = 0.5 * ismembertol(np.array([i]), S)
    cons[i]['idx'] = i/n
    fLat[i] = cons[i]['Lat'][:-1].mean()
    fLon[i] = cons[i]['Lon'][:-1].mean()
    for j in range(i+1, n):
        Adj[i,j] = (ismembertol_custom(cons[i]['Lat'], cons[j]['Lat'], tol) & \
                    ismembertol_custom(cons[i]['Lon'], cons[j]['Lon'], tol)).any().astype(int)
Adj[8,11] = 1
Adj[11,17] = 1
Adj += Adj.T
# Anchor
s = 89


# Define graph
G = nx.from_numpy_matrix(Adj)
pos = {i : (fLon[i].item(), fLat[i].item()) for i in range(n)}
G.pos = pos

colors = np.zeros((n,)) + 100
# Anomalous cluster color
colors[S] = 50
colors[s] = 30


fig, ax = plt.subplots(figsize=(20,10))
colors = np.array(colors)
x = np.array([val[0] for (_, val) in pos.items()])
y = np.array([val[1] for (_, val) in pos.items()])



# Plot nodes
ax.scatter(x, y, c=colors, s=300, edgecolors='k', alpha=1)
# Plot edges
for i,j in enumerate(G.edges):

    X = np.array((pos[j[0]][0], pos[j[1]][0]))
    Y = np.array((pos[j[0]][1], pos[j[1]][1]))
    ax.plot(X, Y, c='blue', alpha=0.7)

ax.axis('off')
plt.savefig("./county/county_graph.png", dpi=500)
plt.show()
