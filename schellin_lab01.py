# -*- coding: utf-8 -*-
#!/usr/bin/env/python3
"""
Created on Thu Sep  7 10:28:26 2023

@author: annas
This file contains scripts for completing Lab 01 for CLASP 410.

To reproduce the plots shown in the lab report, alter the code as follows:
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def spread(nstep =300, ny=3, nx=3, p_burn=0.5, p_bare=0.2, p_start = 0.1, p_fatal = 0.0):
    #creating initial array
    #adding 2 so that each initial square is forest
    forest = np.zeros((nstep, ny, nx), dtype = int)+2
    
    
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
        plt.xlabel('X (km)')
        plt.ylabel('Y (km)')
        plt.title(f'Forest Status iStep = {k}')
        if 3 not in forest[k,:,:]:
            break #if stuff is no longer burning, stop the code
    
    #counting the number of bare squares
    nbare_end = (1*(forest[k,:,:]==1)).sum()
    spread = (nbare_end - nbare_init)/(k)
    
    print(f'Number of initial bare squares = {nbare_init}')
    print(f'Number of ending bare squares = {nbare_end}')
    print(f'Number of days = {k}')
    print(f'Rate of spread = {spread} km^2 land made barren per day')
    print(f'Initial forest density = {init_forestp}%')
    print(f'Times any fire spread to next region = {jumps}')
    #matplotlib.use('Agg')

#def spread(nstep =10, ny=3, nx=3, p_burn=1, p_bare=0.2, p_start = 0.4):
spread(ny = 10, nx = 10)

