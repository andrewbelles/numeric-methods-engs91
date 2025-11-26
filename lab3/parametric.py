#
# parametric.py  Andrew Belles  Oct 4th, 2025 
#
# Computes the global approximating function for the parametric curve 
# as well as the set of cubic splines that approximate 
#
#

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from polyfit import chebyshev_roots, nevilles 

def _spline_helper(f):
    '''
    generate spline matrix and solve for coefficients c 
    we use a natural spline where splines at edge have a second derivative of zero 
    '''
    n = len(f)
    # sample uniformly across f (h_0)
    h = 1.0 / (n - 1) 

    # construct linear system
    A = np.zeros((n,n), dtype=np.float64)
    b = np.zeros((n,), dtype=np.float64)
 
    A[0, 0]   = 1.0 
    A[-1, -1] = 1.0

    for i in range(1,n-1):

        A[i, (i - 1)] = h  
        A[i, i]       = 4.0 * h
        A[i, (i + 1)] = h 

        b[i] = (6.0 / h) * (f[i + 1] - 2.0 * f[i] + f[i - 1]) 
     
    # gather coefficients 
    theta = np.linalg.solve(A, b)
    c = theta[:-1] / 2.0 
    a = f[:-1] 

    # need to compute b and d 
    d = (theta[1:] - theta[:-1]) / (6.0 * h) 
    b = (np.diff(f) / h) - (h / 6.0) * (2.0 * theta[:-1] + theta[1:]) 

    # return all n-1 polynomials in array(?) 
    return np.column_stack([a, b, c, d]) 


def spline(f, csv_path=None):
    n = len(f)
    P = _spline_helper(f)

    # save coefficients to csv 
    if csv_path: 
        df = pd.DataFrame(P, columns=["a", "b", "c", "d"])
        df.to_csv(csv_path, index=False)

    h = 1.0 / (n - 1.0) 

    s = np.linspace(0.0, 1.0, 1000, endpoint=False) # interpolating points 
    S_approx = np.zeros_like(s)

    for i, ds in enumerate(s):
        k = int(ds / h) # row of P that we use to interpolate  
        if k >= n - 1: 
            k = n - 2 

        dx = ds - k * h 
        a, b, c, d = P[k, :]
        S_approx[i] = a + b * dx + c * dx**2 + d * dx**3  

    # return full approximation from splines 
    return S_approx

def interpolate(f):
    s_optimal = chebyshev_roots(0.0, 1.0, len(f)-1)
    ticks = np.linspace(0, 1.0, 1000) # ds = 1e-3
    
    p_optimal = np.zeros((len(ticks),))
    for i, ds in enumerate(ticks):
        p_optimal[i] = nevilles(ds, s_optimal, f)
    
    return np.array(p_optimal)

def main(): 
    data = np.loadtxt("lab3_data-1.txt")

    x = data[:, 0]
    y = data[:, 1]  

    p_x = interpolate(x)
    p_y = interpolate(y)
    s = np.linspace(0, 1.0, 1000)

    f, axs = plt.subplots(1,3,figsize=(12,5))
    axs[1].plot(s, p_x, label="x(s) approximation")
    axs[1].axhline(max(p_x), label=f"max(x)={max(p_x):.4f}", 
                   color="red", ls="--", lw=0.5)
    axs[1].axhline(min(p_x), label=f"min(x)={min(p_x):.4f}",
                   color="green", ls="--", lw=0.5)
    axs[1].set_xlabel("s")
    axs[1].set_ylabel("x(s)")
    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 5)
    axs[1].legend()

    axs[2].plot(s, p_y, label="y(s) approximation")
    axs[2].set_xlabel("s")
    axs[2].set_ylabel("y(s)")
    axs[2].axhline(max(p_y), label=f"max(y)={max(p_y):.4f}", 
                   color="red", ls="--", lw=0.5)
    axs[2].axhline(min(p_y), label=f"min(y)={min(p_y):.4f}",
                   color="green", ls="--", lw=0.5)
    axs[2].set_xlim(0, 1)
    axs[2].set_ylim(0, 5)
    axs[2].legend()

    axs[0].plot(p_x, p_y, label="global approximation")
    axs[0].plot(x, y, label="original parametric", lw=0.8, ls="--", color="k")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].legend()
    f.suptitle("Global Approximation of Parametric Curve")

    f.savefig("global_parametric.png")
    plt.close() 

    # generate splines 
    s = np.linspace(0, 1.0, 1000, endpoint=False)
    x_spline = spline(x, "xcoeff.csv")
    y_spline = spline(y, "ycoeff.csv")
    
    f, axs = plt.subplots(1,3,figsize=(12,5))
    axs[1].plot(s, x_spline, label="x(s) approximation")
    axs[1].set_xlabel("s")
    axs[1].set_ylabel("x(s)")
    axs[1].axhline(max(x_spline), label=f"max(x)={max(x_spline):.4f}", 
                   color="red", ls="--", lw=0.5)
    axs[1].axhline(min(x_spline), label=f"min(x)={min(x_spline):.4f}",
                   color="green", ls="--", lw=0.5)
    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 5)
    axs[1].legend()

    axs[2].plot(s, y_spline, label="y(s) approximation")
    axs[2].axhline(max(y_spline), label=f"max(y)={max(y_spline):.4f}", 
                   color="red", ls="--", lw=0.5)
    axs[2].axhline(min(y_spline), label=f"min(y)={min(y_spline):.4f}",
                   color="green", ls="--", lw=0.5)
    axs[2].set_xlabel("s")
    axs[2].set_ylabel("y(s)")
    axs[2].set_xlim(0, 1)
    axs[2].set_ylim(0, 5)
    axs[2].legend()

    axs[0].plot(x_spline, y_spline, label="natural splines")
    axs[0].plot(x, y, label="original parametric", lw=0.8, ls="--", color="k")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].legend()
    f.suptitle("Natural Cubic Spline fit to Parametric Curve")

    f.savefig("spline_parametric.png")
    plt.close() 

    f, axs = plt.subplots(1,3,figsize=(12,5))
    axs[0].plot(x_spline, y_spline, label="natural spline", color="blue", lw=0.8) 
    axs[0].plot(p_x, p_y, label="global", color="red", lw=0.8)
    axs[0].legend()

    axs[1].plot(s, x_spline, label="x(s) spline", color="blue")
    axs[1].plot(s, p_x, label="x(s) global", color="red")
    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 5)
    axs[1].legend()

    axs[2].plot(s, y_spline, label="y(s) spline", color="blue")
    axs[2].plot(s, p_y, label="y(s) global", color="red")
    axs[2].set_xlim(0, 1)
    axs[2].set_ylim(0, 5)
    axs[2].legend()

    f.suptitle("Comparison Between Natural Spline and Global Approximation")
    f.savefig("comp_spline_global.png")
    plt.close()

if __name__ == "__main__":
    main()
