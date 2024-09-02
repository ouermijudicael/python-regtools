import numpy as np

def fil_fac(s, reg_param, method='Tikh', s1=None, V1=None):
    """
    Filter factors for some regularization methods.
    
    Parameters:
    s : array_like
        Singular values.
    reg_param : array_like
        Regularization parameter.
    method : str, optional
        Regularization method. Default is 'Tikh'.
            method = 'dsvd' : damped SVD or GSVD
            method = 'tsvd' : truncated SVD or GSVD
            method = 'Tikh' : Tikhonov regularization
            method = 'ttls' : truncated TLS.
    s1 : array_like, optional
        Singular values of [A, b] for 'ttls' method.
    V1 : array_like, optional
        Right singular matrix of [A, b] for 'ttls' method.
    
    Returns:
    f : ndarray
        Filter factors.
    """
    p, ps = s.shape if s.ndim > 1 else (len(s), 1)
    lr = len(reg_param)
    f = np.zeros((p, lr))
    
    if np.min(reg_param) <= 0:
        raise ValueError('Regularization parameter must be positive')
    if method in ['tsvd', 'tgsv', 'ttls'] and np.max(reg_param) > p:
        raise ValueError('Truncation parameter too large')
    
    for j in range(lr):
        if method in ['cg', 'nu', 'ls']:
            raise ValueError('Filter factors for iterative methods are not supported')
        elif method in ['dsvd', 'dgsv']:
            if ps == 1:
                f[:, j] = s / (s + reg_param[j])
            else:
                f[:, j] = s[:, 0] / (s[:, 0] + reg_param[j] * s[:, 1])
        elif method in ['Tikh', 'tikh']:
            if ps == 1:
                f[:, j] = (s ** 2) / (s ** 2 + reg_param[j] ** 2)
            else:
                f[:, j] = (s[:, 0] ** 2) / (s[:, 0] ** 2 + reg_param[j] ** 2 * s[:, 1] ** 2)
        elif method in ['tsvd', 'tgsv']:
            if ps == 1:
                f[:, j] = np.concatenate([np.ones(reg_param[j]), np.zeros(p - reg_param[j])])
            else:
                f[:, j] = np.concatenate([np.zeros(p - reg_param[j]), np.ones(reg_param[j])])
        elif method == 'ttls':
            if s1 is not None and V1 is not None:
                coef = (V1[p, :] ** 2) / np.linalg.norm(V1[p, reg_param[j]:p + 1]) ** 2
                for i in range(p):
                    k = reg_param[j]
                    f[i, j] = s[i] ** 2 * np.sum(coef[:k] / (s1[:k] + s[i]) / (s1[:k] - s[i]))
                    if f[i, j] < 0:
                        f[i, j] = np.finfo(float).eps
                    if i > 0 and f[i - 1, j] <= np.finfo(float).eps and f[i, j] > f[i - 1, j]:
                        f[i, j] = f[i - 1, j]
            else:
                raise ValueError('The SVD of [A, b] must be supplied')
        else:
            raise ValueError('Illegal method')
    
    return f