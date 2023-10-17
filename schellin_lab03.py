# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 09:34:44 2023

@author: annas
"""
import numpy as np
import matplotlib.pyplot as plt

sigma = 5.67E-8

def energybal(e=0.3, eg = 1.0, So = 1350, a = 0.33, nlayers = 4, debug = False):
    '''
    Given the N-layer atmospheric model that we derived in class, and its 
    subsequent matrix, this function creates a matrix (A) to model flux at each
    atmospheric layer in terms of emissivities. The function populates the
    matrix using the physical intuition/logic that was derived in class-- this
    is where the "for" loops are implemented. 
    
    Using linear algebra (Ax = b) where A is a matrix of fluxes in terms of
    emissivities and b is a vector representing the sum of each component, we
    can solve for x (the vector that represents the flux at each layer). 
    
    The fluxes are then converted to different temperatures using the 
    Stefan-Boltzmann equation. The function returns an array "temps", where
    each element is the temperature of a different atmospheric layer.
     Parameters
    ----------
    e : float, default = 0.3
        The value of emissivity.
    eg : float, default = 1
        The value of ground emissivity.
    So : float, default = 1350
        The value of solar insolation.
    a : float, default = 0.33.
        The value of albedo.
    nlayers : float, default = 4
        The number of layers being run in the model.
    debug: Boolean, default = false
        This is a debugger function-- when it is set to "true", it prints out
        the values of the A matrix.
    

    Returns
    -------
    temps: float
        The different temperatures (To, T1, T2, etc) in a vector form.
    '''
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)
    S = 0.25*(1-a)*So
    for i in range(nlayers+1):
        for j in range(nlayers+1):
            p = np.abs(i - j) - 1 #exponent power
            if i == 0:
                A[i,j] = eg*(1-e)**p
                if j ==i:
                    #if both j and i are equal to 0
                    A[i,j] = -1
            else:
                A[i,j] = e*(1-e)**p
                if j == i:
                    #if j and i are equal and they're not 0
                    A[i,j] = -2
            
    b[0] = -S #changing first index of b to S
    #Invert matrix:
    Ainv = np.linalg.inv(A)
    
    #Get solution-- array of fluxes
    fluxes = np.matmul(Ainv, b)

    #Convert fluxes to temperatures via Stefan-Boltzmann eqn
    temps = (fluxes/(sigma*e))**(1/4)
    temps[0] = (fluxes[0]/(sigma*eg))**(1/4) #surf temp has different e val
    
    if debug:
        print(f'A[i={i}, j = {j}] = {A[i,j]}')
    #return array to caller
    return temps

def q3exp1():
    '''
    This function is a helper function to create the surface temperature vs emissivity 
    plot from Question 3. 
    
    The function assumes a fixed single layer atmosphere (nlayers = 1).
    The emissivity values are varied from 0.05 to 1 (incremented by 0.05) using
    an array (em_all).
    An empty array is created (surf_Ts) of zeros the size of the em_all array.
    
    There is a "for" loop-- for every index and emissivity value, the function
    solves for the temperatures at each layer using the above "energybal"
    function. Because we are only examining the surface temperature, we fill 
    in the "surf_Ts" array using the first index of the returned temperature 
    array.
    
    The resulting plot shows a range of emissivity values and their 
    corresponding surface temperature values. From this plot, we want to see
    what the model predicts the emissivity of Earth to be for a single-layer
    atmosphere assuming the average Earth surface temperature is 288 K.
    '''
    #running model for a range of emissivities
    em_all = np.arange(0.05,1,0.05)
    surf_Ts = np.zeros(em_all.size) #empty array
    
    for i,e in zip(range(em_all.size), em_all):
        temp = energybal(e=e, nlayers = 1)
        surf_Ts[i] = temp[0]
    fig, ax = plt.subplots(1,1)
    plt.plot(em_all, surf_Ts)
    plt.xlabel('Emissivity')
    plt.ylabel('Surface Temperature (K)')
    plt.show()

def q3exp2():
    '''
    This function is a helper function to create the altitude profile of a 
    modeled Earth system from Question 3. 
    
    This function uses a fixed emissivity value of 0.255. The number of layers
    are varied from 0 to 4 (using an array range from 1 to 6, because the function 
                            uses n+1 layers and starts from layer 1).
    
    Similarly to the previous plot, an empty array of zeros (temps) is created
    to hold the results for plotting purposes.
    
    A "for" loop is created for each index and value of the atmospheric
    layer array. For each of these, the function solves for the temperature
    using the "energybal" function and puts the results into the empty "temps"
    array. 
    
    The resulting plot shows a plot of altitude vs temperature, where the 
    altitude is in terms of the atmospheric layers-- i.e., an atmospheric 
    altitude of 1.0 is equivalent to the height of one atmospheric layer.
    '''
    #plot altitude vs temperature to produce altitude profile
    nlayer_vary = np.arange(1,6,1)
    temps = np.zeros(nlayer_vary.size)
    
    for i,n in zip(range(nlayer_vary.size), nlayer_vary):
        temps = energybal(e = 0.255, nlayers = n-1)
    
    
    fig, ax = plt.subplots(1,1)
    plt.plot(temps, nlayer_vary-1)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Altitude (by layer)')
    plt.show()

def q4exp():
    '''
    This is a helper function to create a plot to answer Question 4-- how many 
    atmospheric layers (using the energybal model) would be expected on the
    planet Venus?
    
    See the 'q3exp2' docstring. This is essentially the same code, but using
    different So (insolation) and e (emissivity) values. The other difference
    is that only the first index of the temperature array is plotted because
    we are only looking at the surface temperature.
    '''
    #should just repeat q3exp2() but with new So and e values
    #and find altitude where Ts = 700
    nlayer_vary = np.arange(1,37,1)
    temps = np.zeros(nlayer_vary.size)
    surf_Ts = np.zeros(nlayer_vary.size)
    
    for i,n in zip(range(nlayer_vary.size), nlayer_vary):
        temps = energybal(e = 1.0, So = 2600, nlayers = n-1)
        surf_Ts[i] = temps[0]

    
    fig, ax = plt.subplots(1,1)
    plt.plot(surf_Ts, nlayer_vary-1)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Number of Layers')
    plt.axhline(y = 30, color = 'r', ls = '--')
    plt.axvline(x = 700, color = 'r', ls = '--')
    plt.show()

def nuclear_winter(e=0.5, eg=1, So = 1350, nlayers = 5, a = 0.33):
    '''
    This function is created to examine a nuclear winter scenario.
    
    See the 'energybal' docstring. The only difference between these two
    functions is that the nuclear_winter function assumes that the topmost
    layer of the atmosphere absorbs all incoming solar flux (none reaches ground).
    To account for this difference, rather than setting the first element of 
    the "b" array to be equal to -S (as it is in the energybal model), the -S
    value is instead set for the last index of the "b" array.
    
    Additionally, the emissivity (e) value is set to 0.5.
    '''
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)
    S = 0.25*(1-a)*So
    for i in range(nlayers+1):
        for j in range(nlayers+1):
            p = np.abs(i - j) - 1 #exponent power
            if i == 0:
                A[i,j] = eg*(1-e)**p
                if j ==i:
                    #if both j and i are equal to 0
                    A[i,j] = -1
            else:
                A[i,j] = e*(1-e)**p
                if j == i:
                    #if j and i are equal and they're not 0
                    A[i,j] = -2
            
    b[-1] = -S #changing first index of b to S
    #Invert matrix:
    Ainv = np.linalg.inv(A)
    
    #Get solution-- array of fluxes
    fluxes = np.matmul(Ainv, b)

    #Convert fluxes to temperatures via Stefan-Boltzmann eqn
    temps = (fluxes/(sigma*e))**(1/4)
    temps[0] = (fluxes[0]/(sigma*eg))**(1/4) #surf temp has different e val
    
    #return array to caller
    return temps

def q5_plot():
    '''
    This is a helper function to create a plot to answer the scenario outlined in
    Question 5. It plots an altitude profile to determine what the new Earth
    surface temperature would be in the nuclear winter scenario. 
    
    '''
    temps = nuclear_winter(nlayers = 5)

    fig, ax = plt.subplots(1,1)
    plt.plot(temps, np.arange(0,6))
    plt.xlabel('Temperature (K)')
    plt.ylabel('Altitude (by layer)')
    plt.show()
    #we might need to double-check this one, boys
