# 
# polyfit.py  Andrew Belles  Oct 3rd, 2025 
# 
# Solution to first question on lab3 
# 
# 
# 

import numpy as np 
import matplotlib.pyplot as plt 

# Function we want to interpolate 
def func(x):
    return np.sin(6.0 * x) * np.cos(np.sqrt(5.0) * x) - (x**2.0) * np.exp(-x / 5.0) 

# Project optimal chebyshev roots onto interval 
def chebyshev_roots(a, b, n):
    k = np.arange(n+1)
    x = np.cos((2.0 * k + 1) * np.pi / (2*(n+1)));
    return (a+b)/2 + (b-a)/2 * x 

# Bottom up nevilles method to compute nth order interpolated approximation at x_k  
def nevilles(xk, xarr, yarr):
    x = np.asarray(xarr, float)
    p = np.asarray(yarr, float).copy()
    
    n = len(xarr)
    for k in range(1, n):
        d = x[k:] - x[:n-k]
        p[:n-k] = ((x[k:] - xk) * p[:n-k] + (xk - x[:n-k]) * p[1:n-k+1]) / d

    return p[0]

# Fit polynomials on uniform points 
def uniform_fit(a: float, b: float, n):
    ticks = np.linspace(a, b, 1000)

    xarr  = np.linspace(a, b, n+1)
    yarr = func(xarr) 

    parr = np.zeros((len(ticks),))
    for i, xk in enumerate(ticks): 
        parr[i] = nevilles(xk, xarr, yarr)
    
    return np.array(parr)

def chebyshev_fit(a: float, b: float, n):
    x_optimal = chebyshev_roots(a, b, n)
    y_optimal = func(x_optimal)
    ticks = np.linspace(a, b, 1000)

    p_optimal = np.zeros((len(ticks),)) 

    for i, xk in enumerate(ticks):
        p_optimal[i] = nevilles(xk, x_optimal, y_optimal)

    return np.array(p_optimal)

def main(): 
    a = -2 
    b = 2 

    # both polynomials estimate well on the interval 
    ns = np.arange(5, 27) 
    
    # concate larger order polynomials to show ringing 
    N = np.concatenate((ns, np.array([60, 75, 90])))

    x = np.linspace(a, b, 1000)
    fx = func(x)

    for n in N: 
        fig = plt.figure() 
        plt.plot(x, fx, label="exact function")
        plt.xlabel("x")
        plt.ylabel("f(x)")

        uniform_poly = uniform_fit(a, b, n)
        plt.plot(x, uniform_poly, label="uniform polynomial")
        plt.legend(loc="lower left")

        plt.title(f"{n}th order uniformly interpolated polynomial")
        fig.savefig(f"uniform_fit_{n}.png")
        plt.close() 

    fig, ax = plt.subplots(2,2, figsize=(8,8))
    ax[0, 0].plot(x, uniform_fit(a, b, 5), label="deg=5")
    ax[0, 0].plot(x, fx, label="f(x)", lw=0.7, ls="--", color="k")
    ax[0, 0].legend()
    ax[0, 0].set_xlabel("x")
    ax[0, 0].set_ylabel("f(x)")

    ax[0, 1].plot(x, uniform_fit(a, b, 20), label="deg=20")
    ax[0, 1].plot(x, fx, label="f(x)", lw=0.7, ls="--", color="k")
    ax[0, 1].legend()
    ax[0, 1].set_xlabel("x")
    ax[0, 1].set_ylabel("f(x)")

    ax[1, 0].plot(x, uniform_fit(a, b, 27), label="deg=27")
    ax[1, 0].plot(x, fx, label="f(x)", lw=0.7, ls="--", color="k")
    ax[1, 0].legend()
    ax[1, 0].set_xlabel("x")
    ax[1, 0].set_ylabel("f(x)")

    ax[1, 1].plot(x, uniform_fit(a, b, 75), label="deg=75")
    ax[1, 1].plot(x, fx, label="f(x)", lw=0.7, ls="--", color="k")
    ax[1, 1].legend()
    ax[1, 1].set_xlabel("x")
    ax[1, 1].set_ylabel("f(x)")

    fig.suptitle("Ringing as Polynomial Degree Increases")
    fig.savefig("uniform_grid.png")
    plt.close() 

    fig, ax = plt.subplots(2,2, figsize=(8,8))
    ax[0, 0].plot(x, np.abs(fx - uniform_fit(a, b, 5)), label="deg=5")
    ax[0, 0].legend()
    ax[0, 0].set_xlabel("x")
    ax[0, 0].set_ylabel("|f(x) - P_5|")

    ax[0, 1].plot(x, np.abs(fx - uniform_fit(a, b, 20)), label="deg=20")
    ax[0, 1].legend()
    ax[0, 1].set_xlabel("x")
    ax[0, 1].set_ylabel("|f(x) - P_{20}|")

    ax[1, 0].plot(x, np.abs(fx - uniform_fit(a, b, 27)), label="deg=27")
    ax[1, 0].legend()
    ax[1, 0].set_xlabel("x")
    ax[1, 0].set_ylabel("|f(x) - P_{27}|")

    ax[1, 1].plot(x, np.abs(fx - uniform_fit(a, b, 75)), label="deg=75")
    ax[1, 1].legend()
    ax[1, 1].set_xlabel("x")
    ax[1, 1].set_ylabel("|f(x) - P_{75}|")

    fig.suptitle("Global Error of Uniform Polynomials for Select Degrees")
    fig.savefig("uerror_grid.png")
    plt.close() 


    for n in N: 
        fig = plt.figure() 
        plt.plot(x, fx, label="exact function")
        plt.xlabel("x")
        plt.ylabel("f(x)")

        optimal_poly = chebyshev_fit(a, b, n)
        plt.plot(x, optimal_poly, label="chebyshev polynomial")

        plt.title(f"{n}th order chebyshev polynomial")
        fig.savefig(f"chebyshev_fit_{n}.png")
        plt.close() 

    fig, ax = plt.subplots(2,2, figsize=(8,8))

    ax[0, 0].plot(x, chebyshev_fit(a, b, 5), label="deg=5")
    ax[0, 0].plot(x, fx, label="f(x)", lw=0.7, ls="--", color="k")
    ax[0, 0].legend()
    ax[0, 0].set_xlabel("x")
    ax[0, 0].set_ylabel("f(x)")

    ax[0, 1].plot(x, chebyshev_fit(a, b, 20), label="deg=20")
    ax[0, 1].plot(x, fx, label="f(x)", lw=0.7, ls="--", color="k")
    ax[0, 1].legend()
    ax[0, 1].set_xlabel("x")
    ax[0, 1].set_ylabel("f(x)")

    ax[1, 0].plot(x, chebyshev_fit(a, b, 27), label="deg=27")
    ax[1, 0].plot(x, fx, label="f(x)", lw=0.7, ls="--", color="k")
    ax[1, 0].legend()
    ax[1, 0].set_xlabel("x")
    ax[1, 0].set_ylabel("f(x)")

    ax[1, 1].plot(x, chebyshev_fit(a, b, 75), label="deg=75")
    ax[1, 1].plot(x, fx, label="f(x)", lw=0.7, ls="--", color="k")
    ax[1, 1].legend()
    ax[1, 1].set_xlabel("x")
    ax[1, 1].set_ylabel("f(x)")

    fig.suptitle("Chebyshev Polynomials' Robustness to Overfitting")
    fig.savefig("chebyshev_grid.png")
    plt.close() 

    fig, ax = plt.subplots(2,2, figsize=(8,8))
    ax[0, 0].plot(x, np.abs(fx - chebyshev_fit(a, b, 5)), label="deg=5")
    ax[0, 0].legend()
    ax[0, 0].set_xlabel("x")
    ax[0, 0].set_ylabel("|f(x) - P_5|")

    ax[0, 1].plot(x, np.abs(fx - chebyshev_fit(a, b, 20)), label="deg=20")
    ax[0, 1].legend()
    ax[0, 1].set_xlabel("x")
    ax[0, 1].set_ylabel("|f(x) - P_{20}|")

    ax[1, 0].plot(x, np.abs(fx - chebyshev_fit(a, b, 27)), label="deg=27")
    ax[1, 0].legend()
    ax[1, 0].set_xlabel("x")
    ax[1, 0].set_ylabel("|f(x) - P_{27}|")

    ax[1, 1].plot(x, np.abs(fx - chebyshev_fit(a, b, 75)), label="err deg=75")
    ax[1, 1].legend()
    ax[1, 1].set_xlabel("x")
    ax[1, 1].set_ylabel("|f(x) - P_{75}|")

    fig.suptitle("Global Error for Select Chebyshev Polynomials")
    fig.savefig("cerror_grid.png")
    plt.close() 


    N = range(5, 86, 20)
    keyN = {5, 45, 85}

    f, ax = plt.subplots(1, 2, figsize=(11, 5), sharey=True, constrained_layout=True)

    for c, title in zip(ax, ["Uniform Samples", "Chebyshev Samples"]):
        c.set_title(title)
        c.set_xlabel("x")
    ax[0].set_ylabel(r"$|f(x) - P_N(x)|$")

    for n in N:
        u_fx = uniform_fit(a, b, n)
        c_fx = chebyshev_fit(a, b, n)

        uer = np.abs(fx - u_fx)
        cer = np.abs(fx - c_fx)

        uer = np.clip(uer, 1e-16, None)
        cer = np.clip(cer, 1e-16, None)

        lw_u = 1.8 if n in keyN else 0.8
        al_u = 1.0 if n in keyN else 0.35
        lw_c = lw_u
        al_c = al_u

        ax[0].semilogy(x, uer, ls="-", lw=lw_u, alpha=al_u,
                       label=(f"deg={n}" if n in keyN else None), rasterized=True)
        ax[1].semilogy(x, cer, ls="-", lw=lw_c, alpha=al_c,
                       label=(f"deg={n}" if n in keyN else None), rasterized=True)

    for c in ax:
        c.set_ylim(1e-16, 1e2)
        c.grid(True, which="both", ls=":", alpha=0.5)

    ax[0].legend(title="Uniform", loc="upper left", frameon=False)
    ax[1].legend(title="Chebyshev", loc="upper left", frameon=False)

    f.suptitle("Interpolation Error from Uniform and Chebyshev Sampling")
    f.savefig("all_errors.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main() 
