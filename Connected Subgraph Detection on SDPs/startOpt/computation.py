import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d

def PolygonArea(corners):
    # Only for ordered vertices
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area


def computeAUC(scores_noise, scores_signal, gammas):
    """Computes and plots AUC scores.

    Parameters
    ----------
    scores_noise : array
        Array of shape (n_samp, n_gammas) containing false positives of
        classifation.
    scores_noise : array
        Array of shape (n_samp, n_gammas) containing true positives of
        classifation.
    gammas : float or array
        Conducatance parameters used for optimisation.

    Returns
    -------
    aucs : list
        list of shape (n_gammas) containg the AUC score for each gamma.

    """

    gamma_pts = 500
    aucs = np.zeros((gammas.shape[0]))
    n, m = scores_signal.shape

    for gind in range(gammas.shape[0]):
        th_range = np.linspace(scores_noise[:, gind].mean() / m,
            scores_signal[:, gind].mean() * m,
            gamma_pts)
        tps = np.zeros((gamma_pts))
        fps = np.zeros((gamma_pts))


        for thind in range(gamma_pts):
            th = th_range[thind]
            tps[thind] = (scores_signal[:, gind] > th).mean()
            fps[thind] = (scores_noise[:, gind] > th).mean()

        tps = np.concatenate((tps, np.array([0, 1, 0])),0)
        fps = np.concatenate((fps, np.array([0, 1, 1])),0)

        # Create 2-D ndarray of points
        points = np.stack((fps,tps),1)

        points = np.stack((np.concatenate((fps, np.array([1])), 0),
                            np.concatenate((tps, np.array([0])), 0)),1)

        hull = ConvexHull(points)
        # Only vertices to get PolygonArea
        polygon_coords = [(x,y) for (x,y) in zip(points[hull.vertices][:,0], points[hull.vertices][:,1])]

        # Area of polygon = auc = Area enclosed by vertices
        auc = PolygonArea(polygon_coords)

        aucs[gind] = auc

    # Plot and save ROC-curve
    plt.figure()
    plt.plot(points[hull.vertices][:,0], points[hull.vertices][:,1], 'o')
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

    return aucs
