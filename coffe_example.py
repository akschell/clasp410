# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 09:36:55 2023

@author: annas
"""
#!/usr/bin/ env python3

'''
coffee problem in class
'''
#parameters: T(env) = 25, To= 90 or 85, Tf = 60, tstop = 600, dt=1
import numpy as np
import matplotlib.pyplot as plt

# Create a time array
tf, tstep = 600,1
time = np.arange(0,tf, tstep)

def solve_temp(time, k=1./300, T_env = 25, To = 90):
    '''
    This function takes an array of times and returns an array of temperatures
    corresponding to each time.
    
    Parameters
    ==========
    time: Numpy array of times
        Array of time inputs for which you want corresponding temps
        
    
    other parameters
    ================
    
    Returns
    =======
    '''
    temp = T_env + (To - T_env) * np.exp(-k*time)
    
    return temp

def time_to_temp(T_targ, k = 1/300., T_env = 20, To = 90):
    '''
    Given an initial temperature To, an ambient temp T_env, and a cooling rate 
    k, return the time required to reach a target temperature T_targ
    '''
    
    return (-1/k)*np.log((T_targ - T_env)/(To - T_env))

#Solve our coffee question
T_cream = solve_temp(time, To = 85)
T_nocrm = solve_temp(time, To = 90)
T_smart = solve_temp(time, To = 65)

#Get time to drinkable temp
t_cream = time_to_temp(60, To=85) #add cream right away
t_nocrm = time_to_temp(60, To = 90) #add cream once at 60
t_smart = time_to_temp(60, To = 65) #add cream at 65

#create figure and axes objects
fig, ax = plt.subplots(1,1)

#plot lines and label
ax.plot(time, T_nocrm, label = 'No cream til cool')
ax.plot(time, T_cream, label = 'Cream right away')
ax.plot(time, T_smart, label = 'Cream at 65')

ax.axvline(t_nocrm, color = 'r', ls ='--', label = 'No Cream: T = 60')
ax.axvline(t_cream, color = 'b', ls = '--', label = "Cream: T = 60")
ax.axvline(t_smart, color = 'g', ls = '--', label = 'Cream at 65: T = 65')

ax.legend(loc = 'best')

fig.tight_layout()


    
    
    
    