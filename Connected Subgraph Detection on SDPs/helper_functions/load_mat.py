import numpy as np
from os.path import dirname, join as pjoin
import sys
import scipy.io as sio


def load_mat():

    path = "../helper_functions/county/map_two_clusters_final.mat"
    M = sio.loadmat(path)

    con = M['cons']
    cons = {}
    for i in range(con.shape[0]):
        cons[i] = {}
        cons[i]['Geometry'] = con[i]['Geometry'][0].squeeze()
        cons[i]['BoundingBox'] = con[i]['BoundingBox'][0].squeeze()
        cons[i]['Lon'] = con[i]['Lon'][0].squeeze()
        cons[i]['Lat'] = con[i]['Lat'][0].squeeze()
        cons[i]['NAME'] = con[i]['NAME'][0].squeeze()
        cons[i]['STATE_NAME'] = con[i]['STATE_NAME'][0].squeeze()
        cons[i]['STATE_FIPS'] = con[i]['STATE_FIPS'][0].squeeze()
        cons[i]['CNTY_FIPS'] = con[i]['CNTY_FIPS'][0].squeeze()
        cons[i]['FIPS'] = con[i]['FIPS'][0].squeeze()
        cons[i]['pop'] = con[i]['pop'][0].squeeze()
        cons[i]['SDP'] = con[i]['SDP'][0].squeeze()
        cons[i]['rate'] = con[i]['rate'][0].squeeze()

    pop_vec = M['pop_vec']
    yy_1 = M['yy_1']
    # S = M['S']
    S = np.array([68,65,104,92,105,94,48,89,67,58,103,61,87,107,83,91])

    return pop_vec, yy_1, cons, S
