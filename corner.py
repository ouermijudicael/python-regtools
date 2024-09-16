import numpy as np
import matplotlib.pyplot as plt

def corner(rho, eta, fig=None):
    # Find corner of discrete L-curve via adaptive pruning algorithm.

    # Parameters:
    # rho : array-like
    #     Residual norm || A x - b ||.
    # eta : array-like
    #     Solution's (semi)norm || x || or || L x ||.
    # fig : int, optional
    #     If provided, a figure will show the discrete L-curve in log-log scale and indicate the found corner.

    # Returns:
    # k_corner : int
    #     Index of the corner of the L-curve.
    # info : int
    #     Information about possible warnings.

    if len(rho) != len(eta):
        raise ValueError('Vectors rho and eta must have the same length')
    if len(rho) < 3:
        raise ValueError('Vectors rho and eta must have at least 3 elements')

    rho = np.array(rho).flatten()
    eta = np.array(eta).flatten()

    info = 0

    fin = np.isfinite(rho + eta)
    nzr = (rho * eta) != 0
    kept = np.where(fin & nzr)[0]
    if len(kept) == 0:
        raise ValueError('Too many Inf/NaN/zeros found in data')
    if len(kept) < len(rho):
        info += 1
        print('Warning: Bad data - Inf, NaN or zeros found in data. Continuing with the remaining data')

    rho = rho[kept]
    eta = eta[kept]

    if np.any(rho[:-1] < rho[1:]) or np.any(eta[:-1] > eta[1:]):
        info += 10
        print('Warning: Lack of monotonicity')

    nP = len(rho)
    P = np.log10(np.column_stack((rho, eta)))
    V = P[1:nP, :] - P[0:nP-1, :]
    v = np.sqrt(np.sum(V**2, axis=1))
    W = V / v[:, np.newaxis]
    clist = []
    p = min(5, nP-1)
    convex = 0

    Y = np.sort(v)
    I = np.argsort(v)[::-1]

    while p < (nP-1) * 2:
        elmts = np.sort(I[:min(p, nP-1)])

        candidate = Angles(W[elmts, :], elmts)
        if candidate > 0:
            convex = 1
        if candidate and candidate not in clist:
            clist.append(candidate)

        candidate = Global_Behavior(P, W[elmts, :], elmts)
        if candidate not in clist:
            clist.append(candidate)

        p *= 2

    if convex == 0:
        k_corner = []
        info += 100
        print('Warning: Lack of convexity')
        return k_corner, info

    if 1 not in clist:
        clist.insert(0, 1)

    clist = np.sort(clist)

    vz = np.where(np.diff(P[clist, 1]) >= np.abs(np.diff(P[clist, 0])))[0]
    if len(vz) > 1:
        if vz[0] == 0:
            vz = vz[1:]
    elif len(vz) == 1:
        if vz[0] == 0:
            vz = []

    if len(vz) == 0:
        index = clist[-1]
    else:
        vects = np.column_stack((P[clist[1:], 0] - P[clist[:-1], 0], P[clist[1:], 1] - P[clist[:-1], 1]))
        vects = vects / np.sqrt(np.sum(vects**2, axis=1))[:, np.newaxis]
        delta = vects[:-1, 0] * vects[1:, 1] - vects[1:, 0] * vects[:-1, 1]
        vv = np.where(delta[vz-1] <= 0)[0]
        if len(vv) == 0:
            index = clist[vz[-1]]
        else:
            index = clist[vz[vv[0]]]

    k_corner = kept[index]

    if fig:
        plt.figure(fig)
        plt.clf()
        plt.loglog(rho, eta, 'k--o')
        plt.plot(rho[index], eta[index], 'ro')  # Mark the corner point
        plt.xlabel('residual norm || A x - b ||_2')
        plt.ylabel('solution (semi)norm || L x ||_2')
        plt.title(f'Discrete L-curve, corner at {k_corner}')
        # plt.show()

    return k_corner, info

def Angles(W, kv):
    delta = W[:-1, 0] * W[1:, 1] - W[1:, 0] * W[:-1, 1]
    mm = np.min(delta)
    kk = np.argmin(delta)
    if mm < 0:
        return kv[kk] + 1
    else:
        return 0

def Global_Behavior(P, vects, elmts):
    hwedge = np.abs(vects[:, 1])
    An = np.sort(hwedge)
    In = np.argsort(hwedge)

    count = 1
    ln = len(In)
    mn = In[0]
    mx = In[-1]
    while mn >= mx:
        mx = max(mx, In[ln-count])
        count += 1
        mn = min(mn, In[count-1])
    if count > 1:
        I, J = 0, 0
        for i in range(count):
            for j in range(ln-1, ln-count, -1):
                if In[i] < In[j]:
                    I, J = In[i], In[j]
                    break
            if I > 0:
                break
    else:
        I, J = In[0], In[-1]

    x3 = P[elmts[J]+1, 0] + (P[elmts[I], 1] - P[elmts[J]+1, 1]) / (P[elmts[J]+1, 1] - P[elmts[J], 1]) * (P[elmts[J]+1, 0] - P[elmts[J], 0])
    origin = [x3, P[elmts[I], 1]]

    dists = (origin[0] - P[:, 0])**2 + (origin[1] - P[:, 1])**2
    index = np.argmin(dists)

    return index
