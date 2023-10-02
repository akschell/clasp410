# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:19:57 2023

@author: annas
"""
import numpy as np
import scipy
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def dNdt_comp(t, N, a=1, b=2, c=1, d=3):
    '''
    This function calculates the Lotka-Volterra competition equations for two
    species.
    
    Current population numbers are given in N, where the first element
    represents species N1 population and the second represents species N2
    population. Coefficients a,b,c, and d are given.
    
    The function uses these values to calculate the change of species N1 with
    time (dN1dt), and the change in population of species N2 with time (dN2dt). 
    The function returns time (to be compliant with scipy module), dN1dt, and 
    dN2dt. 

    Parameters
    ----------
    t : float
        The current time (not used here)
    N : two-element list
        Current values of N1 and N2 as a list [N1, N2]
    a, b, c, d: float, defaults = 1,2,1,3
        The value of Lotka-Volterra coefficients

    Returns
    -------
    dN1dt, dN2dt : floats
        The time derivatives of 'N1' and 'N2' using the Lotka-Volterra
        competition modeling equations.

    '''
    dN1dt = a*N[0]*(1-N[0]) - b*N[0]*N[1]
    dN2dt = c*N[1]*(1-N[1]) - d*N[1]*N[0]
    
    return dN1dt, dN2dt

def predpray(t, N, a=1, b=2, c=1, d=3):
    '''
    This function calculates the Lotka-Volterra predator-prey equations for
    two species.
    
    Current population numbers are given in N, where the first element
    represents species N1 population and the second represents species N2
    population. Coefficients a,b,c, and d are given.
    
    The function uses these values to calculate the change of species N1 with
    time (dN1dt), and the same for species N2 (dN2dt). 
    

    Parameters
    ----------
    t : float.
        The current time (not used here)
    N : two-element list
        Current values of N1 and N2 as a list [N1,N2]
    a, b, c, d: floats. Defaults = 1, 2, 1, 3.
        The value of the Lotka-Volterra coefficients
    

    Returns
    -------
    dN1dt, dN2dt : floats
        The time derivatives of 'N1' and 'N2' using the Lotka-Volterra
        predator-prey modeling equations.

    '''
    dN1dt = a*N[0] - b*N[0]*N[1]
    dN2dt = -c*N[1] + d*N[0]*N[1]
    
    return dN1dt, dN2dt

def euler_solve(f, N1_init = 0.3, N2_init = 0.6, dT = 0.01, tf = 100, 
                a=1, b=2, c=1, d=3):
    '''
    This function is a Euler Solver, which uses a Taylor series expansion
    about an initial time step to solve for a function result-- either the
    competition modeling equations or the predator-prey modeling equations--
    at the next time step. 
    
    First, an array of time points is created from 0 to tf (final time point)
    being incremented by dT (time step). Two empty arrays are created-- one
    for the N1 population and the other for the N2 population-- that will hold
    the Euler Solver solutions for each population. 
    
    A "for" loop is created using a forward difference derivative approximation.
    dN1 and dN2 are unpacked using a function placeholder starting at time 0
    and using the empty solution arrays. The a=a, b=b, etc., coefficients are
    included for later on when we will be testing to see how varying these
    values changes the behavior of the predator/prey system.
    
    For each iteration, the solver fills in both solution arrays using the
    last solution and adding the time step multiplied by the change in
    species population. It returns the current populations of species N1 
    and N2, along with the time array used.
    
    
    Parameters
    ----------
    N1_init, N2_init: floats. Defaults = 0.3, 0.6
        Initial population values for species N1 and species N2.
    dT : float, default = 0.01.
        Time step in years.
    tf : float, default = 100.
        Final time point in years.
    a,b,c,d: floats, defaults = 1,2,1,3
        The values of the Lotka-Volterra coefficients.

    Returns
    -------
    time : float
        The time array used in the Euler solver.
    N1_sol, N2_sol : floats.
        The current populations of species N1 and N2.

    '''
    time = np.arange(0, tf, dT)
    #Create a solution array:
    N1_sol = np.zeros(time.size)
    N1_sol[0]= N1_init 
    
    N2_sol = np.zeros(time.size)
    N2_sol[0] = N2_init
    
    
    #iterate through our solver
    for i in range(time.size-1):
        dN1, dN2 = f(0, [N1_sol[i], N2_sol[i]], a=a, b=b, c=c, d=d)
        N1_sol[i+1] = N1_sol[i] + dT*dN1
        N2_sol[i+1] = N2_sol[i] + dT*dN2
        
    return time, N1_sol, N2_sol

def solve_rk8(f, N1_init =0.3, N2_init=0.6, dt = 10, tf = 100.0,
              a=1, b=2, c=1, d=3):
    '''
    
    Parameters
    ----------
    f : function
        Placeholder for a function used in the RK8 solver.
    N1_init, N2_init: floats, defaults= 0.3, 0.6
        The initial populations of species N1 and N2.
    dt : float, default = 10
        The time step/increment.
    tf : float, default = 100.0
        The value of the last time step (length of time interval).
    a, b, c, d: floats, defaults = 1,2,1,3
        The values of the Lotka-Volterra coefficients.

    Returns
    -------
    time : float
        The time array used in the RK8 solver.
    N1, N2: floats
        The current population numbers for species N1 and N2.

    '''
    result = solve_ivp(f, [0,tf], [N1_init, N2_init], args = [a,b,c,d], 
                       method = 'DOP853', max_step = dt)
    #Perform the integration
    time, N1, N2 = result.t, result.y[0,:], result.y[1,:]
    
    #return values to caller
    return time, N1, N2

def q1_plot1():
    '''
    This function creates one of the plots in Question 1-- the competition
    model plot. The function unpacks the time, N1, and N2 variables using the
    Euler solver for the competition modeling function (dNdt_comp) using a
    time step dT = 1. Time, N1, and N2 variables are also unpacked using the
    RK8 solver for the competition modeling function.

    The resulting plot shows the populations of predator and prey species
    changing with time, using both the Euler solver method and the RK8 method.
    '''
    #plotting competition model with dT = 1 yr
    ct, cN1, cN2 = euler_solve(dNdt_comp, dT = 1)
    crk8t, crk8N1, crk8N2 = solve_rk8(dNdt_comp)
    
    fig, ax = plt.subplots(1,1)
    
    plt.plot(ct, cN1, label = 'N1 (Euler)', color = 'b')
    plt.plot(ct, cN2, label = 'N2 (Euler)', color = 'r')
    plt.plot(crk8t, crk8N1, label = 'N1 (RK8)', color = 'b', ls = '--')
    plt.plot(crk8t, crk8N2, label = 'N2 (RK8)', color = 'r', ls = '--')
    
    ax.legend(loc = 'best')
    plt.title('Lotka-Volterra Competition Model')
    plt.xlabel('Time (years)')
    plt.ylabel('Population/Carrying Capacity')

def q1_plot2():
    '''
    This function creates one of the plots in Question 1-- the predator-prey
    model plot. The function unpacks the time, N1, and N2 variables using the
    Euler solver for the predator-prey modeling function (predpray) using a
    time step dT = 0.05. Time, N1, and N2 variables are also unpacked using the
    RK8 solver for the predator-prey modeling function.

    The resulting plot shows the populations of predator and prey species in
    the predator-prey model using both the Euler solver method and the RK8 
    method.

    '''
    #plotting predator-prey model with dT = 0.05 yr
    t, N1, N2 = euler_solve(predpray, dT = 0.05)
    rk8t, rk8N1, rk8N2 = solve_rk8(predpray)
    
    fig, ax = plt.subplots(1,1)
    
    plt.plot(t, N1, label = 'N1 (Euler)', color = 'b')
    plt.plot(t, N2, label = 'N2 (Euler)', color = 'r')
    plt.plot(rk8t, rk8N1, label = 'N1 (RK8)', color = 'b', ls = '--')
    plt.plot(rk8t, rk8N2, label = 'N2 (RK8)', color = 'r', ls = '--')
    
    ax.legend(loc = 'best')
    plt.title('Lotka-Volterra Predator-Prey Model')
    plt.xlabel('Time (years)')
    plt.ylabel('Population/Carrying Capacity')
        
def q2_plot1():
    '''
    This function creates one of the plots for Question 2 in the lab report;
    it only considers the competition model equations. In this function, the
    N1_init value is being varied to see how initial conditions affect the
    final result and general behavior of the two species.
    
    An array of values is created for N1_init between 0 and 1, incrementing by
    0.1. Then, a "for" loop is created-- for every index in this array
    representing a different N1_init value, a different plot is generated. 
    An arbitrary N2_init value is chosen for the function-- the experiment was
    repeated 10x with different N2_init values each time. It was deemed easier
    to manually loop the N2_init values rather than looping both the N1_init
    and N2_init values simultaneously to produce 100 plots. 
    
    As each plot is generated, its behavior is assessed and recorded. If the 
    steady-state result of the plot shows an equilibrium behavior, this is
    recorded as a "favorable" outcome.

    '''
    
    #Create ranges of values
    N1_init = np.arange(0,1,0.1)
    
    for i in N1_init:
        ct, cN1, cN2 = euler_solve(dNdt_comp, N1_init = i, N2_init = 0.6, dT = 1)
        crk8t, crk8N1, crk8N2 = solve_rk8(dNdt_comp, N1_init = i, N2_init = 0.6, dt = 1)
        
        fig, ax = plt.subplots(1,1)
        
        plt.plot(ct, cN1, label = 'N1 (Euler)', color = 'b')
        plt.plot(ct, cN2, label = 'N2 (Euler)', color = 'r')
        plt.plot(crk8t, crk8N1, label = 'N1 (RK8)', color = 'b', ls = '--')
        plt.plot(crk8t, crk8N2, label = 'N2 (RK8)', color = 'r', ls = '--')
    
    ax.legend(loc = 'best')
    plt.title('Lotka-Volterra Competition Model')
    plt.xlabel('Time (years)')
    plt.ylabel('Population/Carrying Capacity')
    plt.show()

def q2_plota():
    '''
    This function also focuses on the competition model equations from
    Question 2-- this time, rather than looking at initial conditions, the 
    coefficient values are being individually tested. Thus, the "q2_plota", 
    "q2_plotb", "q2_plotc", and "q2_plotd" all follow the same format, where
    the last letter indicates the coefficient that is being tested.
    
    Similarly to the initial conditions loop, an array is created for one of
    the coefficients. Then, a "for" loop is created. Within the Euler solver
    and RK8 solvers, the coefficient value is set to "i", which is the index 
    of the array created within the function. Thus, a new competition model 
    plot is created for each value of the coefficient that is being tested.
    '''
    #looping through different values of a
    a = np.arange(0,5,1)
    
    for i in a:
        ct, cN1, cN2 = euler_solve(dNdt_comp, dT = 1, a=i)
        crk8t, crk8N1, crk8N2 = solve_rk8(dNdt_comp, dt = 1, a=i)
        
        fig, ax = plt.subplots(1,1)
        
        plt.plot(ct, cN1, label = 'N1 (Euler)', color = 'b')
        plt.plot(ct, cN2, label = 'N2 (Euler)', color = 'r')
        plt.plot(crk8t, crk8N1, label = 'N1 (RK8)', color = 'b', ls = '--')
        plt.plot(crk8t, crk8N2, label = 'N2 (RK8)', color = 'r', ls = '--')
        
        ax.legend(loc = 'best')
    plt.title('Lotka-Volterra Competition Model')
    plt.xlabel('Time (years)')
    plt.ylabel('Population/Carrying Capacity')
    plt.show()
    
def q2_plotb():
    '''
    See q2_plota description.
    '''
    #looping through different values of b
    b = np.arange(0,5,1)
    for i in b:
        ct, cN1, cN2 = euler_solve(dNdt_comp, dT = 1, b=i)
        crk8t, crk8N1, crk8N2 = solve_rk8(dNdt_comp, dt = 1, b=i)
        
        fig, ax = plt.subplots(1,1)
        
        plt.plot(ct, cN1, label = 'N1 (Euler)', color = 'b')
        plt.plot(ct, cN2, label = 'N2 (Euler)', color = 'r')
        plt.plot(crk8t, crk8N1, label = 'N1 (RK8)', color = 'b', ls = '--')
        plt.plot(crk8t, crk8N2, label = 'N2 (RK8)', color = 'r', ls = '--')
        
        ax.legend(loc = 'best')
    plt.title('Lotka-Volterra Competition Model')
    plt.xlabel('Time (years)')
    plt.ylabel('Population/Carrying Capacity')
    plt.show()

def q2_plotc():
    '''
    See q2_plota description.
    '''
    #looping through different values of c
    c = np.arange(0,5,1)
    for i in c:
        ct, cN1, cN2 = euler_solve(dNdt_comp, dT = 1, c=i)
        crk8t, crk8N1, crk8N2 = solve_rk8(dNdt_comp, dt = 1, c=i)
        
        fig, ax = plt.subplots(1,1)
        
        plt.plot(ct, cN1, label = 'N1 (Euler)', color = 'b')
        plt.plot(ct, cN2, label = 'N2 (Euler)', color = 'r')
        plt.plot(crk8t, crk8N1, label = 'N1 (RK8)', color = 'b', ls = '--')
        plt.plot(crk8t, crk8N2, label = 'N2 (RK8)', color = 'r', ls = '--')
        
        ax.legend(loc = 'best')
    plt.title('Lotka-Volterra Competition Model')
    plt.xlabel('Time (years)')
    plt.ylabel('Population/Carrying Capacity')
    plt.show()
    
def q2_plotd():
    '''
    See q2_plota description.
    '''
    #looping through different values of d
    d = np.arange(0,5,1)
    for i in d:
        ct, cN1, cN2 = euler_solve(dNdt_comp, dT = 1, d=i)
        crk8t, crk8N1, crk8N2 = solve_rk8(dNdt_comp, dt = 1, d=i)
        
        fig, ax = plt.subplots(1,1)
        
        plt.plot(ct, cN1, label = 'N1 (Euler)', color = 'b')
        plt.plot(ct, cN2, label = 'N2 (Euler)', color = 'r')
        plt.plot(crk8t, crk8N1, label = 'N1 (RK8)', color = 'b', ls = '--')
        plt.plot(crk8t, crk8N2, label = 'N2 (RK8)', color = 'r', ls = '--')
        
        ax.legend(loc = 'best')
    plt.title('Lotka-Volterra Competition Model')
    plt.xlabel('Time (years)')
    plt.ylabel('Population/Carrying Capacity')
    plt.show()
    
def q3_plotphase():
    '''
    This function creates one of the plots for Question 3-- the phase diagram
    for the predator-prey model. The time, N1, and N2 variables are unpacked
    for both the Euler Solver method and the RK8 method. Then, the N1
    population is graphed against the N2 population for both models.
    '''

    t, N1, N2 = euler_solve(predpray, dT = 0.05)
    rk8t, rk8N1, rk8N2 = solve_rk8(predpray)
    
    fig, ax = plt.subplots(1,1)
    
    plt.plot(N1, N2, label = 'Euler Phase Diagram', color = 'r')
    plt.plot(rk8N1, rk8N2, label = 'RK8 Phase Diagram', color = 'b', ls = '--')
    
    ax.legend(loc = 'best')
    plt.title('Phase Diagram')
    plt.xlabel('N1 (prey)')
    plt.ylabel('N2 (predator)')

def q3_plotinit():
    '''
    This plot follows the same logic as q2_plot1, where the N1_init values are
    being looped from 0 to 1. Rather than manually replotting for different
    N2_init values, the N2_init value is held constant at an equilibrium value.
    
    The reverse is true for q3_plotinit2(), where instead the N1_init value is
    held constant and N2_init is being looped.
    '''
    #Create ranges of values
    N1_init = np.arange(0,1,0.1)
    for i in N1_init:
        crk8t, crk8N1, crk8N2 = solve_rk8(predpray, N1_init = i, N2_init = 0.6, dt = 1)
        
        fig, ax = plt.subplots(1,1)
        
        plt.plot(crk8t, crk8N1, label = 'N1 (RK8)', color = 'b', ls = '--')
        plt.plot(crk8t, crk8N2, label = 'N2 (RK8)', color = 'r', ls = '--')
    
        ax.legend(loc = 'best')
        plt.title('Lotka-Volterra Competition Model')
        plt.xlabel('Time (years)')
        plt.ylabel('Population/Carrying Capacity')
    plt.show()

def q3_plotinit2():
    '''
    See q3_plotinit() docstring.
    '''
    #create ranges of values, vary N2_init
    N2_init = np.arange(0,1,0.1)
    for i in N2_init:
        crk8t, crk8N1, crk8N2 = solve_rk8(predpray, N1_init = 0.3, N2_init = i, dt = 1)
        
        fig, ax = plt.subplots(1,1)
        
        #plt.plot(ct, cN1, label = 'N1 (Euler)', color = 'b')
        #plt.plot(ct, cN2, label = 'N2 (Euler)', color = 'r')
        plt.plot(crk8t, crk8N1, label = 'N1 (RK8)', color = 'b', ls = '--')
        plt.plot(crk8t, crk8N2, label = 'N2 (RK8)', color = 'r', ls = '--')
    
        ax.legend(loc = 'best')
        plt.title('Lotka-Volterra Competition Model')
        plt.xlabel('Time (years)')
        plt.ylabel('Population/Carrying Capacity')
    plt.show()

def q3_plotcoeffa():
    '''
    This function focuses on the predator-prey equations from Question 3.
    This time, rather than looking at initial conditions, the coefficient
    values are being individually tested. Thus, the "q3_plotcoeffa", 
    "q3_plotcoeffb", "q3_plotcoeffc", and "q3_plotcoeffd" all follow the 
    same format, where the last letter indicates the coefficient that is being 
    tested.
    
    These functions follow the same logic as the q2_plota() function, but the 
    Euler solver is omitted to prevent a scalar overflow.
    '''
    #looping through different values of a
    a = np.arange(0,5,1)
    
    for i in a:
        rk8t, rk8N1, rk8N2 = solve_rk8(predpray, dt = 1, a=i)
        
        fig, ax = plt.subplots(1,1)
        
        
        plt.plot(rk8t, rk8N1, label = 'N1 (RK8)', color = 'b', ls = '--')
        plt.plot(rk8t, rk8N2, label = 'N2 (RK8)', color = 'r', ls = '--')
        
        ax.legend(loc = 'best')
        plt.title('Lotka-Volterra Competition Model')
        plt.xlabel('Time (years)')
        plt.ylabel('Population/Carrying Capacity')
    plt.show()
    
def q3_plotcoeffb():
    '''
    See q3_plotcoeffa().
    '''
    #looping through different values of b
    b = np.arange(0,5,1)
    
    for i in b:
        rk8t, rk8N1, rk8N2 = solve_rk8(predpray, dt = 1, b=i)
        
        fig, ax = plt.subplots(1,1)
        
        
        plt.plot(rk8t, rk8N1, label = 'N1 (RK8)', color = 'b', ls = '--')
        plt.plot(rk8t, rk8N2, label = 'N2 (RK8)', color = 'r', ls = '--')
        
        ax.legend(loc = 'best')
        plt.title('Lotka-Volterra Competition Model')
        plt.xlabel('Time (years)')
        plt.ylabel('Population/Carrying Capacity')
    plt.show()

def q3_plotcoeffc():
    '''
    See q3_plotcoeffa().
    '''
    #looping through different values of c
    c = np.arange(0,5,1)
    
    for i in c:
        rk8t, rk8N1, rk8N2 = solve_rk8(predpray, dt = 1, c=i)
        
        fig, ax = plt.subplots(1,1)
        
        
        plt.plot(rk8t, rk8N1, label = 'N1 (RK8)', color = 'b', ls = '--')
        plt.plot(rk8t, rk8N2, label = 'N2 (RK8)', color = 'r', ls = '--')
        
        ax.legend(loc = 'best')
        plt.title('Lotka-Volterra Competition Model')
        plt.xlabel('Time (years)')
        plt.ylabel('Population/Carrying Capacity')
    plt.show()
    
def q3_plotcoeffd():
    '''
    See q3_plotcoeffa().
    '''
    #looping through different values of d
    d = np.arange(0,5,1)
    
    for i in d:
        rk8t, rk8N1, rk8N2 = solve_rk8(predpray, dt = 1, d=i)
        
        fig, ax = plt.subplots(1,1)
        
        
        plt.plot(rk8t, rk8N1, label = 'N1 (RK8)', color = 'b', ls = '--')
        plt.plot(rk8t, rk8N2, label = 'N2 (RK8)', color = 'r', ls = '--')
        
        ax.legend(loc = 'best')
        plt.title('Lotka-Volterra Competition Model')
        plt.xlabel('Time (years)')
        plt.ylabel('Population/Carrying Capacity')
    plt.show()

