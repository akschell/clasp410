# -*- coding: utf-8 -*-
#!/usr/bin/env/python3
"""
Created on Thu Sep  7 10:28:26 2023

@author: annas
This file contains scripts for completing Lab 01 for CLASP 410.

The function "spread" works as follows:
    
    An initial 3-dimensional array is created with i,j,k dimensions.
    "i" and "j" refer to the x and y coordinates, respectively, and the "k"
    dimension represents time.
    
    An array is initialized with all zeros + 2 so the initial status of every
    square is forested/a healthy individual.
    
    The chances of having any square be burning or initially bare are set
    separately.
    
    A color map is created so a barren status = tan, forest status = green, 
    and burning status = red (black = death).
    
    The initial forest density is calculated for quantitative data analysis
    by summing up the number of squares at t=0 that have the "forest" status.
    
    The counter "jump" starts at 0, and is increased by 1 every time the
    fire/disease spreads to a new square.
    
    Three "for" loops are created. One loops through time, the other through
    x coordinates, and the last through y coordinates. The code checks
    for burning squares one column over all available y-values (one x-value).
    If the square is on fire and it is not all the way to the left edge of the
    simulation, the square to the left has a chance to burn. After it is
    decided whether that square burns, the square that was on fire becomes
    barren (regardless of outcome). The same logic is repeated for the right,
    up, and down directions. After looping through all y-values for an 
    x-coordinate, the logic repeats for a new x-coordinate until all coordinates
    have been exhausted. At this point, a new time interval begins, and the
    code keeps looping through these time iterations until either 1) the number
    of steps (nsteps) has been exhausted (unlikely, this is set really high) or
    until there are no more burning tiles.
    
    The number of bare tiles at the end of the simulation is totalled. If there
    was a p_fatal>0, the number of fatalities is also totalled at the end.
    
    Important values for analysis are printed out with the simulation, recorded
    in Excel, and plotted.
    

To reproduce the plots shown in the lab report, alter the code as follows:
    For Question 1:
        Run "spread" function with nstep = 10, ny = 3, nx = 3,
        p_burn = 1.0, p_start = 0, p_fatal = 0.0, p_bare = 0.0
        
        After creating initial array, add line 
        forest[0, nx//2, ny//2] to set middle tile on fire.
        
        This line is commented within the code; delete the "#" to run it.
        
        To create a plot that is wider than it is tall, run "spread" function
        with nstep = 10, ny = 3, nx = 6, p_burn = 1.0, p_start = 0, 
        p_fatal = 0.0, p_bare = 0.0.
        
        
    For Question 2a:
        Run "spread" function with nstep = 300, ny=nx = 100, p_bare = 0.2,
        p_start = 0.001, p_fatal = 0.0, and run each trial while varying 
        p_burn from [0,1].
        
        33 total trials were run with different p_burn values and relevant
        numbers (number of days, rate of spread, times any fire spread) are
        in the output of the code. Those numbers were graphed using Excel.
        
    For Question 2b:
        Run "spread" function with nstep = 300, ny = nx = 100, p_bare varied
        from [0,1], p_start = 0.001, p_burn = 0.4, p_fatal = 0.0.
        
        33 total trials were run with different p_bare values. As before,
        relevant values were recorded and plotted in Excel.
        
    For Question 3:
        For simplicity, the same code was used for all questions. 
        
        The status "1" indicates those who were either initially
        immune (vaccinated) or who had the disease and survived
        (gained immunity). "p_bare" reflects the initial immune
        population.
        
        The status "2" indicates a healthy person.
        
        The status "3" indicates someone who is sick."p_burn" reflects the 
        probability of disease transmissibility.
        
        The status "4" indicates someone who had the disease and died from it. 
        "p_fatal" indicates how likely an infected individual is to die 
        from disease.
        
        To determine how disease mortality rate (p_fatal) affects the spread
        of disease, several trials are run while varying p_fatal from 0 to 1
        (nx = ny = 100, nstep = 300, p_start = 0.001, p_burn = 0.4, p_bare = 0.2).
        Like previous trials, these numbers were recorded and plotted in Excel.
        
        To determine how early vaccination rate (p_bare) affects the spread of
        disease, several trials are run while varying p_bare from 0 to 1
        (nx = ny = 100, nstep = 300, p_start = 0.001, p_burn = 0.4, p_fatal = 0.1).
        Important values were calculated, recorded, and plotted in Excel. 
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def spread(nstep =300, ny=3, nx=3, p_burn=0.4, p_bare=0.2, p_start = 0.001, p_fatal = 0.0):
    #creating initial array
    #adding 2 so that each initial square is forest
    forest = np.zeros((nstep, ny, nx), dtype = int)+2
    
    #For Question 1), uncomment this line below
    #forest[0,nx//2, ny//2] = 3
    
    #setting prob of burning squares at t=0
    isburn = np.random.rand(ny,nx)
    isburn = isburn<p_start
    forest[0,isburn] = 3
    
    #setting probability of bare squares at t=0
    isbare = np.random.rand(ny,nx)
    isbare = isbare<p_bare
    forest[0,isbare]=1
    #counting number of bare squares for quantitative discussion
    nbare_init = (1*(forest[0,:,:]==1)).sum()
    
    #creating initial plot for t=0
    forest_cmap = ListedColormap(['tan','darkgreen','crimson'])
    fig,ax= plt.subplots(1,1)
    ax.pcolor(forest[0,:,:], cmap=forest_cmap, vmin=1,vmax=3)
    plt.title('Forest Status iStep = 0')
    
    #calculating initial forest density for discussion
    nforest_init = (1*(forest[0,:,:]==2)).sum()
    init_forestp = (nforest_init/(nx*ny))*100
    
    
    #counting number of times fire spreads
    jump = 0
    for k in range(1,nstep):
        forest[k,:,:] = forest[k-1,:,:]
        for i in range(nx):
            for j in range(ny):
                if forest[k-1,j,i] == 3:
                    #if burning in the previous frame
                    #here is how to update the next frame
                    if i>0:
                        #burn left
                        if forest[k-1,j,i-1]==2 and np.random.rand()<p_burn:
                            forest[k,j,i-1] =3
                            #set new status of left square to burn
                    if i<nx-1:
                        #burn right
                        if forest[k-1,j,i+1] ==2 and np.random.rand()<p_burn:
                            forest[k, j, i+1] =3
                    if j>0:
                        #burn up
                        if forest[k-1,j-1, i] ==2 and np.random.rand()<p_burn:
                            forest[k, j-1, i] =3
                    if j<ny-1:
                        #burn down
                        if forest[k-1,j+1, i] ==2 and np.random.rand()<p_burn:
                            forest[k, j+1, i] =3
            
                    #updating number of times fire spreads
                    jump +=1
                    nfire_init = (1*(forest[0,:,:]==3)).sum()
                    #subtracting the initial number of fires
                    jumps = jump - nfire_init
                    forest[k, j, i] = 1 #make the burning cell barren
                    
                    #calc fatality prob
                    if np.random.rand()<p_fatal:
                        forest[k,j,i] = 4



        #make color map where index 1=barren, 2= forest, 3=fire
        forest_cmap = ListedColormap(['tan','darkgreen','crimson', 'black'])
        
        fig,ax= plt.subplots(1,1)
        
        ax.pcolor(forest[k,:,:], cmap=forest_cmap, vmin=1,vmax=4)
        
        #To prevent plots from generating, uncomment line below:
        #plt.close('all')
        
        plt.xlabel('X (km)')
        plt.ylabel('Y (km)')
        plt.title(f'Forest Status iStep = {k}')
        if 3 not in forest[k,:,:]:
            break #if stuff is no longer burning, stop the code
    
    #counting the number of bare squares
    nbare_end = (1*(forest[k,:,:]==1)).sum()
    spread = (nbare_end - nbare_init)/(k)
    
    #counting individuals who died
    rip = (1*(forest[k,:,:]==4)).sum()
    
    
    print(f'Number of initial bare squares = {nbare_init}')
    print(f'Number of ending bare squares = {nbare_end}')
    print(f'Number of days = {k}')
    print(f'Rate of spread = {spread} km^2 land made barren per day')
    print(f'Initial forest density = {init_forestp}%')
    print(f'Times any fire spread to next region = {jumps}')
    print(f'Number of deaths = {rip}')
    

#spread(nstep =300, ny=3, nx=3, p_burn=0.4, p_bare=0.2, p_start = 0.001, p_fatal = 0.0)
spread(ny = 100, nx = 100, p_start = 0.001, p_fatal = 0.1)
