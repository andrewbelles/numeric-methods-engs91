#
# root.py  Andrew Belles  Sept 24th, 2025 
#
# Python code to determine roots for questions 2 and 3 
# of lab2 
#
#

import numpy as np 
import matplotlib.pyplot as plt 

EPS = 1e-19 
MAX_ITER = 1e4 

# Function definitions to pass as func pointers 

def first(x: np.float128): 
    return ( x + np.cos(x) ) * np.exp(-(x**2)) + x * np.cos(x)

def first_prime(x: np.float128): 
    c = 1.0 - np.sin(x) - 2.0 * x**2 + 2.0 * x * np.cos(x)
    return ( c * np.exp(-(x**2)) + np.cos(x) - x * np.sin(x))

def first_double_prime(x: np.float128): 
    sx = np.sin(x)
    cx = np.cos(x)
    ex = np.exp(-(x**2))

    a = -2.0 * sx - 4.0 * ex * (1.0 - sx) + 4.0 * x**2 * ex * (cx + x)
    b = -2.0 * ex * (cx + x) - ex * cx - x * cx

    return a + b 


def second(x: np.float128): 
    return first(x)**2 

def second_prime(x: np.float128):
    return 2.0 * first_prime(x) * first(x)

def second_double_prime(x: np.float128): 
    return 2.0 * first_double_prime(x) * first(x) + 2.0 * first_prime(x)**2


def third(x: np.float128):
    return first(x)**3

def third_prime(x: np.float128): 
    return 3.0 * first_prime(x) * second(x)

def third_double_prime(x: np.float128): 
    a = 3.0 * first_double_prime(x) * second(x) 
    b = 6.0 * first_prime(x)**2 * first(x)
    return a + b  


# Method definitions 
# Same function ptrs (loosely, python doesn't explicitly require strict prototyping)

def secant(xn: np.float128, xprev: np.float128, func, dfunc, d2func):
    c1 = func(xn)    # f(p_{n})
    c0 = func(xprev) # f(p_{n-1})
    # If c1 and c0 are sufficiently close avoid division by zero 
    return xn - ((c1 * (xn - xprev)) / (c1 - c0))

def newtons(xn: np.float128, xprev, func, dfunc, d2func):
    # Add tolerance here to avoid a degenerate root => division by zero 
    return xn - (func(xn) )/ (dfunc(xn))

def modified_newtons(xn: np.float128, xprev, func, dfunc, d2func): 
    a =  func(xn) * dfunc(xn)
    b = (dfunc(xn)**2 - func(xn) * d2func(xn)) 
    return xn - (a / (b)) 

def cubic_method(xn: np.float128, xprev, func, dfunc, d2func):
    a = func(xn) / (dfunc(xn)) 
    b = (d2func(xn) / (2.0 * dfunc(xn)**3)) * (func(xn)**2)
    return xn - a - b 


def find_root(a: float, b: float, 
              method, func, dfunc, d2func): 
    ''' 
    Best approximates roots in bound, returns absolute error array and root
    We assume that function meets fixed point conditions guarenteeing unique soln  

    Since conditions are guarenteed inclusive on the boundary, let p0 = a
    '''

    error = []
    xprev = b
    xcurr = (a + b) / 2.0   # Start in middle of interval   
    current_error = np.abs(xcurr - xprev)  
    xnext = 0.0 
    i = 0 

    while i < MAX_ITER and current_error > EPS:
        xnext = method(xcurr, xprev, func, dfunc, d2func)
        current_error = np.abs(xnext - xcurr) 

        error.append(current_error)
        xprev = xcurr 
        xcurr = xnext 
        i += 1 

    return float(xcurr), np.asarray(error, float)


def main(): 

    labels = ["first", "second", "third"]
    colors = ["blue", "green", "red"] 
    methods = [secant, newtons, modified_newtons, cubic_method]
    method_labels = ["secant", "newtons", "modified_newtons", "cubic_method"]
    funcs = {
        "first": [first, first_prime, first_double_prime], 
        "second": [second, second_prime, second_double_prime], 
        "third": [third, third_prime, third_double_prime]
    }

    fig, axii_grid = plt.subplots(3,4,figsize=(15,15))

    for i, (axii, label, color) in enumerate(zip(axii_grid, labels, colors)): 
        # Get functions 
        f, df, d2f = funcs[label]

        # For each method 
        results = [find_root(1.0, 2.0, method, f, df, d2f) 
                   for method in methods]
        roots, errors = zip(*results)

        roots = np.asarray(roots, float)
        for (root, error) in zip(roots, errors):
            print((len(error),root))

        # Plot all methods error for function 
        for axes, method, error in zip(axii, method_labels, errors):
            n = np.arange(1, len(error)+1)
            axes.semilogy(n, error, label=f"{i+1}.) {method}", color=color) 
            axes.set_xlabel("iteration")
            axes.set_ylabel("error [log-scale]")
            axes.legend(loc='best')            

    fig.suptitle("Error for Root Approximation for Functions")
    fig.tight_layout()
    fig.savefig("root_errors.png")
    plt.close() 


if __name__ == "__main__":
    main()
