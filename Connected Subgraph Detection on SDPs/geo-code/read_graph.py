import numpy as np

def readGeoGraph(path):
    pts = []
    edges = []
    with open(path) as f:
        line = f.readline()
        n, _, m = tuple(line.split())

        A = np.zeros((int(n),int(n)))

        for i in range(int(n)):
            line = f.readline()
            pts.append([float(p.strip()) for p in line.split('\t')])


        for j in range(int(m)):
            line = f.readline()
            edges.append([int(e.strip()) for e in line.split("\t")])

    # Adjecency matrix
    # A[i, j] = 0 if nodes i and j are connected (directed graph)
    for edge in edges:
        A[edge[0],edge[1]] = 1

    # Make it undirected --> symmetric
    A += A.T
    return (A, np.array(pts))

if __name__ == "__main__":

    path = "./example_10k_3d.graph"

    A, pts = readGeoGraph(path)
    print(A.shape)
    print(pts.shape)
