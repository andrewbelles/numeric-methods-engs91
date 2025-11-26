#!/bin/python3.14 
# 
# bioheat_pde.py  Andrew Belles  Nov 21th, 2025 
# 
# O(dt^2 + dx^2) Crank-Nicolson Scheme Solution to the 
# bioheat parabollic pde given in lab7. Matrix solution using LU decomp 
# 

import numpy as np 
import numpy.typing as npt 
import matplotlib.pyplot as plt 

from typing import List, Dict  

from gauseid_bioheat import Bioheat 

EPS = 1e-12

class PDEBioheat(): 
    '''
    Full PDE model of Bioheat diffusion equation. Solved using a Crank-Nicolson Scheme
    and LU decompositon mixed with forward/back-substitution 
    '''
    L       = np.float64(1.0)
    LAMBDSQ = np.sqrt(np.float64(2.7))
    SIG     = np.float64(100.0)
    GAMMA   = np.float64(-1.0 / L) 
    CORE    = np.float64(37.0)
    SURFACE = np.float64(32.0)
    ARTERY  = CORE
    BCLEFT  = np.float64(CORE - ARTERY)
    BCRIGHT = np.float64(SURFACE - ARTERY)

    def __init__(self, nodes: List[int], dt: np.float64): 
        self.nodes = nodes 
        self.dt = dt 
        self.sols: Dict[int, npt.NDArray] = {}


    def decompose_(self, A: npt.NDArray[np.float64]): 
        '''
        A is a tridiagonal matrix 
        '''
        N, _ = A.shape 
        L, U = np.zeros_like(A), np.zeros_like(A)
        
        # Crout LU: uii = 1  
        U += np.identity(N)

        # Off diagonal (N - 1) eq-ns  
        for i in range(N-1): 
            L[i + 1, i] = A[i + 1, i]

        # Alternate diag and off diag 
        for i in range(N-1): 
            L[i, i]   = A[i, i] - np.dot(L[i, :], U[:, i])
            U[i, i+1] = (A[i, i+1] - np.dot(L[i, :], U[:, i+1])) / L[i, i]

        L[-1, -1]   = A[-1, -1] - np.dot(L[-1, :], U[:, -1])

        return L, U

    
    def bounds_(self, N: int) -> npt.NDArray[np.float64]: 
        xar = np.linspace(0.0, self.L, N + 2)
        dx  = self.L / N 
        b   = self.dt * self.SIG * np.exp(self.GAMMA * (self.L - xar[1:-1]))  
        b[0]  += (self.dt / dx**2) * self.BCLEFT 
        b[-1] += (self.dt / dx**2) * self.BCRIGHT
        assert(b.shape == (N,))
        return b 

    
    def forward_sub_(self, L: npt.NDArray[np.float64], r: npt.NDArray[np.float64]): 
        ''' 
        Inputs: 
            L must necessarily be a lower triangular matrix
            RHS must necessarily be a vector of N elements
        '''
        N, _ = L.shape
        u    = np.zeros((N,), dtype=np.float64)

        for i in range(N):
            u[i] = (r[i] - np.dot(L[i, :i], u[:i])) / L[i, i]

        return u 


    def backward_sub_(self, U: npt.NDArray[np.float64], r: npt.NDArray[np.float64]): 

        N, _ = U.shape 
        u    = np.zeros((N,), dtype=np.float64)

        for i in reversed(range(N)): 
            u[i] = (r[i] - np.dot(U[i, i+1:], u[i+1:])) / U[i, i]

        return u 


    def solve_(self, N: int) -> npt.NDArray[np.float64]: 
        '''
        Numeric Solution for N interior nodes to the bioheat PDE from initial conditions
        to determined steady-state 
        '''
        dx = self.L / N 
        # Form matrix A and B 
        k = self.dt / (dx**2)
        khalf = 0.5 * k 
        l = 0.5 * self.LAMBDSQ * self.dt
        d = 1.0 - k - l 

        a = np.full((N,), 1.0 + k + l)
        c = np.full((N-1,), 0.5 * k) 
        
        A  = np.zeros((N,N), dtype=np.float64)
        A += np.diag(a) - np.diag(c, k=-1) - np.diag(c, k=1)

        # Perform LU decomposition on A ~ O(N)
        L, U = self.decompose_(A)
        u    = -5.0 * np.ones((N,), dtype=np.float64)
        Bu   = np.zeros((N,), dtype=np.float64)
        b    = self.bounds_(N)

        sol: List[npt.NDArray[np.float64]] = [u]

        while True: 
            # Compute B * u, w/o matmul 
            Bu[0]  = d * u[0] + khalf * u[1]
            for i in range(1, N-1): 
                Bu[i] = khalf * u[i-1] + d * u[i] + khalf * u[i+1]  
            Bu[-1] = khalf * u[N-2] + d * u[-1] 

            Bu += b

            y = self.forward_sub_(L, Bu)
            u = self.backward_sub_(U, y)
            sol.append(u)
            if np.linalg.norm(sol[-1] - sol[-2], ord=np.inf) < EPS:
                break

        return np.array(sol)

    
    def sol(self): 

        for N in self.nodes: 
            self.sols[N] = self.solve_(N)


    def plot(self, title: str, png: str, steady): 
        '''
        Plots the best solution for PDE as heat diffusion map 
        '''
        best = self.sols[self.nodes[-1]]
        nt, nx = best.shape 
        
        t = np.arange(nt) * self.dt 
        x = np.linspace(0.0, self.L, nx)

        f = plt.figure(figsize=(12,7))
        extent = x[0], x[-1], t[0], t[-1]
        plt.imshow(best, aspect="auto", origin="lower", 
                   extent=extent)

        plt.xlabel("x [m]")
        plt.ylabel("t [s]")
        plt.xlim([0.0, self.L])
        plt.colorbar(label=R"temperature $[^\circ C]$")
        plt.title(title)
        plt.tight_layout()
        f.savefig(png)
        plt.close()

        f = plt.figure(figsize=(12,7))
        plt.plot(x, best[-1, :], label="Steady-State")
        plt.plot(x, steady, label="lab7-2b", lw=1.1, color="g", ls="--", alpha=0.8)
        plt.xlim([0.0, self.L])
        plt.hlines([self.BCRIGHT, self.BCLEFT], 0.0, self.L, 
                   ls=":",
                   colors="k", 
                   lw=0.5,
                   label="boundary conditions")
        plt.xlabel("x [m]")
        plt.ylabel(R"temperature $[^\circ C]$")
        plt.title(title + " - Steady-State Solution")
        plt.legend() 
        f.savefig(png.removesuffix(".png") + "-steadystate.png")
        plt.close() 

def main(): 

    pde_model = PDEBioheat([5, 10, 20, 40, 80, 160, 320], dt=np.float64(1e-3))
    pde_model.sol()
    
    model = Bioheat([320], True)
    model.sol() 

    pde_model.plot(
        "Bioheat Solution - Crank-Nicolson",
        "pde-bioheat-cn.png",
        model.solutions[320]
    )

    # Compare absolute error w.r.t to previous lab's steady state
    pde = pde_model.sols[320] 
    error = np.abs(pde[-1, :] - model.solutions[320])
    x = np.linspace(0.0, PDEBioheat.L, pde.shape[1])
    
    f = plt.figure(figsize=(12,7))
    plt.plot(x, error, label="rel error")
    plt.xlabel("x [m]")
    plt.ylabel(R"$|\tilde{T}^p - \tilde{T}|$")
    plt.title("absolute error of steady-state solution versus previous lab")
    plt.legend() 
    f.savefig("abserr.png")
    plt.close() 




if __name__ == "__main__": 
    main() 
