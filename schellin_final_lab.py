
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import random

from matplotlib.colors import ListedColormap

#plt.style.use('fivethirtyeight')

#Creating custom color map for the grid
colors = ['black', 'white', 'red']
atmos_cmap = ListedColormap(colors)

#Turning each color status into words
status = {1: 'Blank', 2: 'Aerosol', 3:'CCN'}

#if you don't want to generate plots, uncomment this line below:
#plt.ioff()

def aerosol_ccn(nx = 11, ny = 11, nstep = 12, p_spread = 1.0, p_start = 0.1, 
                p_ccn = 0.0, center_ccn = False):
    '''
    This is the main model used in the report. An array, 'atmos', is initialized
    with all 1s (status = blank tiles). Then, the chance that any of these tiles
    is actually an aerosol tile is evaluated for each grid cell. The same is done
    for CCN tiles to initialize the array with an initial amount of blank tiles, 
    aerosol tiles, and CCN tiles.
    
    After the 2D grid has been initialized, the function uses several "for" loops
    to determine what will happen to each grid cell for each time step at each
    x and y coordinate. First, if an aerosol tile is directly adjacent to a CCN
    tile, the aerosol tile turns into a CCN tile. Additionally, if a tile is an
    aerosol tile, a random number between 0 and 3 will be rolled to determine which
    direction it has a chance to move. The algorithm terminates when the number
    of steps (nsteps) has been exhausted.
    
    Parameters
    ----------
    nx, ny : int, defaults = 11
        Size of grid in X and Y directions.
    nstep : int, default = 12
        Maximum number of steps to take during simulation.
    p_spread : float, default = 1.0
        Probability that aerosol will move to the adjacent tile for each
        iteration.
    p_start : float, default = 0.1
        Probability of any tile being an aerosol at the start of the simulation.
    p_ccn : float, default = 0.0
        Probability of any tile being a CCN at the start of the simulation.
    center_ccn : kwarg, default = False
        If True, this variable sets a CCN particle in the middle of the grid.
    
    Returns
    --------
    A 2D grid with blank, aerosol, and CCN tiles. Also returns the number 
    of CCN particles at the end of the simulation.
    '''
    from numpy.random import rand
    
    #Initialize grid and call it 'atmos'(sphere). Grid has i,j,k dimensions,
    #time is represented with 'nstep'. Adding 1 to each tile so the default
    #for each tile is 'blank' status (non-aerosol).
    atmos = np.zeros((nstep, ny, nx), dtype = int) + 1
    
    #Set probability of any tile being an aerosol tile at t=0. If the random
    #number generated is less than p_start, the status of that tile becomes 
    #'aerosol' in the 'atmos' grid.
    is_aero = np.random.rand(ny,nx)
    is_aero = is_aero<p_start
    atmos[0, is_aero] = 2
    
    #Set probability of any tile being a CCN tile at t=0. If the random number
    #generated is less than p_ccn, the status of that tile becomes "CCN" in the
    #"atmos" grid.
    is_ccn = np.random.rand(ny, nx)
    is_ccn = is_ccn<p_ccn
    atmos[0, is_ccn] = 3
    
    if center_ccn:
        #for a single CCN in middle of grid
        atmos[0, ny//2, nx//2] = 3
    
    #for a single aerosol moving in middle of grid
    #atmos[0, ny//2, nx//2] = 2
    
    #Create initial plot for t=0.
    fig, ax = plt.subplots(1,1)
    ax.pcolor(atmos[0,:,:], cmap = atmos_cmap, vmin=1, vmax=3)
    
    #Time to run the simulation!
    for k in range(1, nstep):
        #Use previous grid cell values to create new grid.
        atmos[k, :, :] = atmos[k-1, :, :]
        
        #Aerosol tile can only move from one tile to next tile.
        #Determine if aerosol will move left, right, up, or down by rolling
        #the dice. If roll = 0, aerosol moves right; roll =1 for left; roll=2
        #for up; roll = 3 for down.
        for i in range(nx):
            for j in range(ny):
                if atmos[k-1, j, i] ==2:
                    #If a tile is an aerosol, move it. If the aerosol is next
                    #to a CCN tile, change its status to CCN. 
                    if i > 0:
                        if atmos[k-1, j, i-1] ==3:
                            atmos[k, j, i] = 3
                    if j > 0:
                        if atmos[k-1, j-1, i] ==3:
                            atmos[k, j, i] =3
                    if j < (ny - 2): 
                        if atmos[k-1, j+1, i] ==3:
                            atmos[k, j, i] = 3
                    if i < (nx - 2): 
                        if atmos[k-1, j, i+1] ==3:
                            atmos[k, j, i] =3
                    
                    
                if atmos[k, j, i] == atmos[k-1, j, i] and atmos[k,j,i] == 2 and np.random.rand()<p_spread:
                    #If tile is an aerosol, move it.
                    #Determine if aerosol will move left, right, up, or down 
                    #with a dice roll
                    move = random.randint(0, 3)
                    
                    if move == 0 and i<nx-1:
                        #move right
                        atmos[k,j,i+1]=2
                    if move == 1 and i>0:
                        #move left
                        atmos[k,j,i-1]=2
                    if move ==2 and j<ny-1:
                        #move down
                        atmos[k,j+1,i] =2
                    if move ==3 and j>0:
                        #move up
                        atmos[k,j-1,i] = 2
                    atmos[k,j,i] = 1
                
            if 2 not in atmos[k,:,:]:
                break
                
        fig, ax = plt.subplots(1,1)
        plt.xlabel('Micrometers')
        plt.ylabel('Micrometers')
        ax.pcolor(atmos[k,:,:], cmap = atmos_cmap, vmin = 1, vmax = 3)
        plt.show()
        
    #want to return the number of CCN particles at the end of the simulation
    #for quantitative analysis purposes
    ccn = (1*(atmos[k,:,:]==3)).sum()
    print(f'Number of CCN/cloud particles = {ccn}')
    

def question_2a():
    '''
    This is a helper function (proof of concept)
    for the first part of Question 2 of the lab report. 
    This function has one CCN tile at the center of the grid and 
    several aerosol particles moving around the grid. 
    '''
    aerosol_ccn(nx = 25, ny = 25, nstep = 50, center_ccn = True)
    
def question_2b():
    '''
    This is a helper function for the plot for Question 2. This function
    varies the number of aerosol particles at the beginning of the simulation. 
    The number of resulting cloud particles is plotted in the next function.
    '''
    for aero_vary in np.arange(0, 1, 0.05):
        aerosol_ccn(nx = 25, ny = 25, nstep = 50, p_start = aero_vary, center_ccn = True)
        
def question_2bplot():
    '''
    This is a helper function to create the plot for Question 2. This function
    plots the output from the question_2b() function. The x and y values are 
    manually put in.
    '''
    x = np.arange(0, 1, 0.05)
    y = np.array([1, 3, 8, 7, 16, 4, 14, 10, 9, 16, 15, 12, 16, 22, 19,
                  18, 15, 18, 18, 28])
    plt.plot(x, y)
    plt.xlabel('Aerosol Starting Tile Probability')
    plt.ylabel('Number of Cloud Particles')
    plt.show()
def question_3():
    '''
    This is a helper function (proof of concept) for Question 3 of the report.
    This uses the aerosol_ccn function to create a grid with multiple aerosol
    particles moving around with multiple CCN particles on the grid.
    '''
    aerosol_ccn(nx = 25, ny = 25, nstep = 50, p_start = 0.2, p_ccn = 0.05)
def question_3b():
    '''
    This is a helper function for Question 3 of the lab
    report. This function has several CCN tiles that are scattered around the
    grid and several aerosol particles moving around the grid.
    '''
    for ccn_vary in np.arange(0, 1, 0.05):
        aerosol_ccn(nx = 25, ny = 25, nstep = 50, p_start = 0.2, p_ccn = ccn_vary)

        
def question_3bplot():
    '''
    This is a helper function for the second part of Question 3 of the lab 
    report. This function plots the results of the p_ccn variation by plotting
    the number of CCN particles at the end of the simulation vs the p_ccn
    value used in the simulation. This function plots the output of the
    question_2b() function.
    '''
    x = np.arange(0,1, 0.05)
    y = np.array([0, 115, 141, 185, 204, 229, 252, 283, 312, 336, 367, 415, 
                  425, 428, 473, 500, 528, 573, 579, 601])
    plt.plot(x, y)
    plt.xlabel('CCN probability')
    plt.ylabel('Number of Cloud Particles')
    plt.show()
    
def question_4():
    '''
    This is a helper function for Question 4 of the lab report. Here, the 
    p_spread value is being varied. in the next function, the results of this
    variation will be plotted. 
    '''
    for spread_vary in np.arange(0, 1, 0.05):
        aerosol_ccn(nx = 25, ny = 25, nstep = 50, p_spread = spread_vary, 
                    p_start = 0.2, p_ccn = 0.05)
    
def question_4plot():
    '''
    This is a helper plotting function for Question 4 of the lab report. This
    function plots the results of the p_spread variation by plotting the 
    number of CCN particles at the end of the simulation against the p_spread
    value used in the simulation. This function plots the output of the 
    question_4() function by manually inputting that function output into 
    the 'y' array.
    '''
    x = np.arange(0, 1, 0.05)
    y = np.array([54, 77, 100, 102, 106, 86, 103, 91, 122, 93, 116, 103, 125, 
                  105, 108, 87, 109, 97, 128, 90])
    plt.plot(x, y)
    plt.xlabel('Aerosol Spreading Probability')
    plt.ylabel('Number of Cloud Particles')
    plt.show()

