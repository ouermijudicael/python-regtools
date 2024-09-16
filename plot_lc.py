import numpy as np
import matplotlib.pyplot as plt

def plot_lc(rho, eta, marker='-', ps=1, reg_param=None):
    # Plot the L-curve.

    # Parameters:
    # rho : array-like
    #     Residual norm || A x - b ||.
    # eta : array-like
    #     Solution norm || x || if ps = 1, or || L x || if ps = 2.
    # marker : str, optional
    #     Marker style for the plot. Default is '-'.
    # ps : int, optional
    #     Indicates the type of solution norm. Default is 1.
    # reg_param : array-like, optional
    #     Regularization parameters corresponding to rho and eta.

    if ps < 1 or ps > 2:
        raise ValueError('Illegal value of ps')

    n = len(rho)
    np_points = 10
    ni = round(n / np_points)

    plt.loglog(rho[1:-1], eta[1:-1])
    ax = plt.axis()

    if max(eta) / min(eta) > 10 or max(rho) / min(rho) > 10:
        if reg_param is None:
            plt.loglog(rho, eta, marker)
            plt.axis(ax)
        else:
            plt.loglog(rho, eta, marker, rho[ni-1::ni], eta[ni-1::ni], 'x')
            plt.axis(ax)
            # hold_state = plt.ishold()
            # plt.hold(True)
            for k in range(ni-1, n, ni):
                plt.text(rho[k], eta[k], str(reg_param[k]))
            # if not hold_state:
                # plt.hold(False)
    else:
        if reg_param is None:
            plt.plot(rho, eta, marker)
            plt.axis(ax)
        else:
            plt.plot(rho, eta, marker, rho[ni-1::ni], eta[ni-1::ni], 'x')
            plt.axis(ax)
            # hold_state = plt.ishold()
            plt.hold(True)
            for k in range(ni-1, n, ni):
                plt.text(rho[k], eta[k], str(reg_param[k]))
            # if not hold_state:
                # plt.hold(False)

    plt.xlabel('residual norm || A x - b ||_2')
    if ps == 1:
        plt.ylabel('solution norm || x ||_2')
    else:
        plt.ylabel('solution semi-norm || L x ||_2')
    plt.title('L-curve')
    # plt.show()