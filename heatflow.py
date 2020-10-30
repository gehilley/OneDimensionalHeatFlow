def heat_flow(To, Tb, alpha, t, v = -1E-3, dx = 100, L = 40000, method = 'RK45', use_cfl = True):
    import numpy as np

    z = np.arange(0, L, dx)
    Go = (Tb - To) / L
    x0 = To + Go*z


    def to_integrate(t_step, x):
        dxdz = np.diff(x) / dx
        d2xdz2 = (x[0:-3] - 2*x[1:-2] + x[2:-1]) / (np.power(dx,2))
        dxdt_advect = np.zeros_like(x)
        dxdt_diff = np.zeros_like(x)
        try:
            dxdt_advect[0:-1] = v(t_step) * dxdz

        except:
            dxdt_advect[0:-1] = v*dxdz
        dxdt_diff[1:-2] = alpha*d2xdz2
        dxdt_advect[0] = 0.0

        return dxdt_advect + dxdt_diff

    from scipy.integrate import solve_ivp
    try:
        cfl_v = np.max(v(t))
        cfl_dt = 0.25 * dx / np.abs(cfl_v) if use_cfl else np.inf
    except:
        cfl_dt = 0.25 * dx / np.abs(v) if use_cfl else np.inf

    out = solve_ivp(to_integrate, (0, np.max(t)), x0, method=method, max_step = np.max(cfl_dt), t_eval=t)
    if out is None:
        print('problem')
    T = np.zeros((len(t),x0.shape[0]))

    this_y = out.y.T
    for i in range(len(t)):
        T[i,:] = this_y[i,:]
    return np.arange(0, L, dx), T