#python3 -m venv venv_sussman
#from the bash shell:
#source venv_sussman/bin/activate  (this step must be done upon reboot)
#python3 -m pip install qiskit
#pip3 install qiskit-aer
#pip3 install matplotlib
#python3 hadamard_gate.py
#Transpilation is the process of rewriting a given input circuit to 
#match the topology of a specific quantum device, and optimize the 
#circuit instructions for execution on noisy quantum computers.
#import numpy as np
#from qiskit import QuantumCircuit
#from qiskit import transpile
#from qiskit.circuit import Parameter, ParameterVector
#from qiskit_aer import AerSimulator as Aer
#from qiskit.quantum_info import SparsePauliOp
#from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
#from qiskit_aer.primitives import Estimator
#from scipy.optimize import minimize
#import random,math
#from qiskit.quantum_info import Statevector, state_fidelity
#from qiskit.circuit.library import CRXGate, CRYGate, CRZGate,TwoLocal
#import json
#import multiprocessing
#import os

import torch
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data
from functools import partial
#import time
#import sys
#import os

# this is Feng Bao's code together with Z. Zhang and G. Zhang
# compact version
def cond_alpha(t):
    # conditional information
    # alpha_t(0) = 1
    # alpha_t(1) = esp_alpha \approx 0
    return 1 - (1-eps_alpha)*t

def cond_sigma_sq(t):
    # conditional sigma^2
    # sigma2_t(0) = 0
    # sigma2_t(1) = 1
    # sigma(t) = t
    return t

# drift function of forward SDE
def f(t):
    # f=d_(log_alpha)/dt
    alpha_t = cond_alpha(t)
    f_t = -(1-eps_alpha) / alpha_t
    return f_t

def g_sq(t):
    # g = d(sigma_t^2)/dt -2f sigma_t^2
    d_sigma_sq_dt = 1
    #1-2(1/(t-1))t=(t+1)/(1-t)
    g2 = d_sigma_sq_dt - 2*f(t)*cond_sigma_sq(t)
    return g2

def g(t):
    return np.sqrt(g_sq(t))

# generate sample with reverse SDE
def reverse_SDE(x0, score_likelihood=None, time_steps=100,
                drift_fun=f, diffuse_fun=g, alpha_fun=cond_alpha, sigma2_fun=cond_sigma_sq,  save_path=False):
    # x_T: sample from standard Gaussian
    # x_0: target distribution to sample from

    # reverse SDE sampling process
    # N1 = x_T.shape[0]
    # N2 = x0.shape[0]
    # d = x_T.shape[1]

    # Generate the time mesh
    dt = 1.0/time_steps

    # Initialization
    # ensemble_size, n_dim, and device are globals defined in main.
    xt = torch.randn(ensemble_size,n_dim, device=device)
    # we go backwards in time starting with the normal distribution N(0,1)
    t = 1.0

    # define storage
    if save_path:
        path_all = [xt]
        t_vec = [t]

    # forward Euler sampling
    for i in range(time_steps):
        # prior score evaluation
        #if no parameter provided, then use cond_alpha=1-(1-eps)t
        alpha_t = alpha_fun(t)
        #if no parameter provided, then use cond_sigma_sq=t
        sigma2_t = sigma2_fun(t)


        # Evaluate the diffusion term
        # if no parameter provided then use g=sqrt(g_sq)=
        # sqrt(1-2(-(1-eps)/(1-(1-eps)t))t)
        diffuse = diffuse_fun(t)

        # Evaluate the drift term
        # drift = drift_fun(t)*xt - diffuse**2 * score_eval

        # Update
        # drift_fun=-1/(1-t) if eps=0 and no parameters specified.
        if score_likelihood is not None:
            xt += - dt*( drift_fun(t)*xt + diffuse**2 * ( (xt - alpha_t*x0)/sigma2_t) - diffuse**2 * score_likelihood(xt, t) ) \
                  + np.sqrt(dt)*diffuse*torch.randn_like(xt)
        else:
            xt += - dt*( drift_fun(t)*xt + diffuse**2 * ( (xt - alpha_t*x0)/sigma2_t) ) + np.sqrt(dt)*diffuse*torch.randn_like(xt)

        # Store the state in the path
        if save_path:
            path_all.append(xt)
            t_vec.append(t)

        # update time
        t = t - dt

    if save_path:
        return path_all, t_vec
    else:
        return xt

# the lorenz drift
def lorenz96_drift(x):
    return (torch.roll(x, -1)- torch.roll(x, 2))*torch.roll(x, 1) - x + F

if __name__ == "__main__":
    ####################################################################
    ####################################################################
    # filtering settings
    # lorenz system
    F = 8
    # n_dim = 1000000
    n_dim=100

    nx=n_dim

    SDE_sigma = 0.1

    # filtering setup
    dt = 0.005
     #filtering_step=number of steps in the data assimilation window.
    filtering_steps = 50

    nt=filtering_steps

    X = np.zeros ( [ nx, nt + 1 ] )
    Y = np.zeros ( [ nx, nt + 1 ] )
    Z = np.zeros ( [ nx, nt + 1 ] )

    # observation sigma
    obs_sigma = 0.05

    ####################################################################
    # EnSF setup
    # define the diffusion process
    eps_alpha = 0.05

    # ensemble size
    ensemble_size = 250

    # forward Euler step
    # euler_steps=number of steps for the backwards SDE.
    euler_steps = 100

    # damping function(tau(0) = 1;  tau(1) = 0;)
    def g_tau(t):
        return 1-t

    # saving file name
    exp_name = 'L96_results'


    # computation setting
    torch.set_default_dtype(torch.float16) # half precision
    #device = 'cuda'
    device = 'cpu'

    ####################################################################
    ####################################################################

    # initial state
    # rand = uniform distribution
    # randn= N(0,1)
    # state_target is used for verification purposes; state_target is the
    # "exact" solution.
    state_target = 10*torch.rand(n_dim, device=device)

    # filtering initial ensemble
    # x_state = state_target + torch.randn(ensemble_size, n_dim, device=device)*0.5
    # x_state is the result of the new "score function" data assimilation method
    x_state = torch.randn(ensemble_size, n_dim, device=device)  # pure Gaussian initial

    x_est = torch.mean(x_state,dim=0)

    j_time=0
    for i in range ( 0, nx ):
        X[i,j_time]=i
        Y[i,j_time]=j_time
        Z[i,j_time]=x_est[i]

    # get state memory size
    mem_state = state_target.element_size() * state_target.nelement()/1e+6
    mem_ensemble = mem_state * ensemble_size
    print(f'single state memory: {mem_state:.2f} MB')
    print(f'state ensemble memory: {mem_ensemble:.2f} MB')

    torch.cuda.empty_cache()
    # filtering cycles
    for i in range(filtering_steps):
        print(f'step={i}:')
        #t1 = time.time()

        # prediction step ############################################
        # state forward in time
        x_state += dt*lorenz96_drift(x_state) + np.sqrt(dt)*SDE_sigma*torch.randn_like(x_state)

        # ensemble prediction
        state_target += dt*lorenz96_drift(state_target) + np.sqrt(dt)*SDE_sigma*torch.randn_like(state_target)

        # update step ################################################
        # get observation
        obs = torch.atan(state_target) + torch.randn_like(state_target)*obs_sigma

        # define likelihood score
        # S=grad log p(x)
        # e.g. p(x)=(1/sqrt(sigma 2 pi))e^{-(x-mu)^2/sigma}
        # then S=A+B(x-mu
        def score_likelihood(xt, t):
            # obs: (d)
            # xt: (ensemble, d)
            #obs_sigma=0.05, g_tau(t)=1-t
            score_x = -(torch.atan(xt) - obs)/obs_sigma**2 * (1./(1. + xt**2))
            tau = g_tau(t)
            return tau*score_x

        # generate posterior sample
        x_state = reverse_SDE(x0=x_state, score_likelihood=score_likelihood, time_steps=euler_steps)

        # get state estimates
        x_est = torch.mean(x_state,dim=0)

        j_time=j_time+1
        for i in range ( 0, nx ):
          X[i,j_time]=i
          Y[i,j_time]=j_time
          Z[i,j_time]=x_est[i]

        # get rmse
        rmse_temp = torch.sqrt(torch.mean((x_est - state_target)**2)).item()

        # get time
        if x_state.device.type == 'cuda':
            torch.cuda.current_stream().synchronize()
        #t2 = time.time()
        print(f'\t RMSE = {rmse_temp:.4f}')
        #print(f'\t time = {t2-t1:.4f} ')

        # break
        if rmse_temp > 1000:
            print('diverge!')
            break

    fig = plt.figure ( )
    ax = fig.add_subplot ( projection = '3d' )
    surf = ax.plot_surface ( X, Y, Z, cmap = cm.coolwarm, \
     linewidth = 0, antialiased = False )
    ax.set_xlabel ( '<--X-->' )
    ax.set_ylabel ( '<--T-->' )
    ax.set_zlabel ( '<--U(X,T)-->' )
    fig.colorbar ( surf, shrink = 0.5, aspect = 10 )
    plt.savefig ( 'L96_results.png' )
    plt.show ( )
    
