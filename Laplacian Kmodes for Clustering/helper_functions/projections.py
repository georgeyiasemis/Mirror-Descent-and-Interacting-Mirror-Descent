import torch

def projection_simplex(u, r=1):
    '''
    Project u unto the the r-simplex, i.e.
    out = min_v || v - u ||_2^2, subject to sum(v_i)_i=1^n = r, v_i >= 0
    '''
    u = u.squeeze()
    assert len(u.shape) == 1, "u should be 1-D"
    assert r > 0, "Radius r of ball should be positive"

    if torch.sum(u) == r and (u >= 0).all():
        return u

    w, _ = u.sort(descending=True)
    cumsum = w.cumsum(0)
    indices,  = torch.where(w * torch.arange(1, u.shape[0]+1) > (cumsum - r))
    ind = indices[-1]
    lambd = (cumsum - r)[ind] / (ind+1).float()
    return (u - lambd).clamp(0)

if __name__ == '__main__':

    # u = torch.tensor([1/3,1/3,0, 1/2], dtype=torch.float64)
    u = torch.randn((10000) * 1000)
    print(projection_simplex(u, r=1).sum())
