# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 09:37:38 2023

@author: annas
"""
import matplotlib.pyplot as plt
import numpy as np

#put reference solution here to test against
t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4,
10.7, 8.5, 3.1, -6.0, -12.0, -16.9])
t_kanger_warm05 = t_kanger + 0.5
t_kanger_warm1 = t_kanger + 1
t_kanger_warm3 = t_kanger + 3

def sample_init(x):
    '''
    Simple initial boundary condition function.
    '''
    
    return 4*x - 4*x**2

def heat_solve(dt = 0.02, dx = 0.2, c2 = 1, xmax = 1.0, tmax = 0.2, 
               init = sample_init):
    '''
    This function is a numerical solver for the heat equation. The function
    takes in several inputs (as well as the initial conditions) to find a 
    solution to the heat equation. The function returns the x values (in terms
    of length or depth), the time values, and the temperature at each point.
    
    This first numerical solver was created specifically for the wire in ice
    scenario, where each end of the wire is fixed at 0 degrees Celsius.
    
    Parameters
    ===========
    dt, dx : default = 0.02, 0.2
        Time and space step
    c2 : float, default = 1.0
        Thermal diffusivity
    xmax, tmax: float, default = 1.0, 0.2
        Set max values for space and time grids
    init : scalar or function
        Set initial condition. If a function, should take position as an 
        input and return temperature using same units as t, temp.
    
    
    Returns
    ========
    x : numpy vector
        Array of position locations/x-grid
    t : numpy vector
        Array of time points/y-grid
    temp: numpy 2D grid
        Temperature as a function of time and space.

    '''
    #check stability criterion
    if (dt> (dx**2/(2*c2))):
        print('Out stability criterion is not met.')
        print('dt = ', dt, 'dx = ', dx, 'c2 =', c2)
        return
    
    # Set constants
    r = c2* dt/dx**2
    
    #Create space and time grids
    x = np.arange(0, xmax + dx, dx)
    t = np.arange(0, tmax+dt, dt)
    #save number of points
    M,N = x.size, t.size
    
    #Create temperature solution array:
    temp = np.zeros([M,N])
    
    #Set initial and boundary conditions.
    temp[0,:] = 0
    temp[-1,:] = 0
    #temp[:, 0] = 4*x - 4*x**2
    
    #Set initial condition
    if callable(int):
        temp[:,0] = init(x)
    else:
        temp[:, 0] = init
    
    #Solve!
    for j in range(0, N-1):
        temp[1:-1, j+1] = (1-2*r)*temp[1:-1,j] + r*(temp[2:, j] + temp[:-2, j])
    

    return x, t, temp

def temp_kanger(t):
    '''
    For an array of times in days, this function returns a time series of 
    temperatures for Kangerlussuaq, Greenland.

    '''
    t_amp = (t_kanger - t_kanger.mean()).max()
    return t_amp*np.sin(np.pi/180 * t - np.pi/2) + t_kanger.mean()
    #returns initial conditions for when x is zero

def heat_plot(dt = 0.02):
    '''
    This function (using a default float "dt" value of 0.02) plots the 
    results of the heat_solve equation (the wire with ends fixed at 0 Celsius).
    The output of this function is two different plots. The first is a heat 
    map showing the temperature gradient as a function of space (on each point
    of the wire) and time. The second plot is a line plot showing the 
    temperature changing with "depth" (length of the wire) spatially.
    '''
    #Get soln using solver
    x, time, heat = heat_solve()
    
    #Create a figure/axes object
    fig, axes = plt.subplots(1,1)
    
    #Create a color map and add a color bar
    map = axes.pcolor(time, x, heat, cmap ='seismic', vmin=-1, vmax=1)
    plt.colorbar(map, ax = axes, label = 'Temperature ($C$)')
    axes.invert_yaxis()
    plt.xlabel('Time (s)')
    plt.ylabel('Length (m)')

    #set indexing for final year of results
    loc = int(-365/dt)
    
    #extract the min values over the final year
    winter = heat[:, loc:].min(axis =1)
    
    #Create a temp profile plot
    fig, ax2 = plt.subplots(1,1, figsize=(10,8))
    ax2.plot(winter, x, label = 'Winter')
    ax2.invert_yaxis()
    plt.xlabel('Temperature (*C)')
    plt.ylabel('Length (m)')

def kanger_heat_solve(dt = 10, dx = 1, c2 = 0.0216, xmax = 100, tmax = 100*365, 
                      init = temp_kanger):
    '''
    This function is a numerical solver for the heat equation. The function
    takes in several inputs (as well as the initial conditions) to find a 
    solution to the heat equation. The function returns the x values (in terms
    of length or depth), the time values, and the temperature at each point.
    
    This second numerical solver was created using the given standard
    temperatures in Kangerlussaq, Greenland. The initial conditions used 
    are defined in the array "t_kanger" and used in the function "temp_kanger".
    
    Parameters
    ===========
    dt, dx : default = 0.02, 0.2
        Time and space step
    c2 : float, default = 1.0
        Thermal diffusivity
    xmax, tmax: float, default = 1.0, 0.2
        Set max values for space and time grids
    init : scalar or function
        Set initial condition. If a function, should take position as an 
        input and return temperature using same units as t, temp.
    
    
    Returns
    ========
    x : numpy vector
        Array of position locations/x-grid
    t : numpy vector
        Array of time points/y-grid
    temp: numpy 2D grid
        Temperature as a function of time and space.

    '''
    #check stability criterion
    if (dt> (dx**2/(2*c2))):
        print('Out stability criterion is not met.')
        print('dt = ', dt, 'dx = ', dx, 'c2 =', c2)
        return
    
    # Set constants
    r = c2* dt/dx**2
    
    #Create space and time grids
    x = np.arange(0, xmax + dx, dx)
    t = np.arange(0, tmax+dt, dt)
    #save number of points
    M,N = x.size, t.size
    
    #Create temperature solution array:
    temp = np.zeros([M,N])
    
    #Set initial and boundary conditions.
    temp[:,0] = 0
    temp[-1,:] = 5
    #temp[:, 0] = 4*x - 4*x**2
    
    #Set initial condition
    if callable(int):
        temp[0,:] = init(t)
    else:
        temp[0, :] = init
    
    #Solve!
    for j in range(0, N-1):
        temp[1:-1, j+1] = (1-2*r)*temp[1:-1,j] + r*(temp[2:, j] + temp[:-2, j])
    

    return x, t, temp
def kanger_plot(dt = 10):
    '''
    This function (using a default float "dt" value of 10) plots the 
    results of the kanger_heat_solve equation. 
    
    The output of this function is two different plots. The first is a heat 
    map showing the temperature gradient as a function of space (each depth)
    and time. Seasonal variation may be seen at the top of the plot-- the
    surface. The bottom of the plot being red is representative of geothermal
    warming from below the surface.
    
    The second plot is a line plot showing the temperature changing
    with depth (into the ground) in both the winter and summer seasons.
    '''
    #Get soln using solver
    x, time, heat = kanger_heat_solve()
    
    #Create a figure/axes object
    fig, axes = plt.subplots(1,1)
    
    #Create a color map and add a color bar
    map = axes.pcolor(time/365, x, heat, cmap ='seismic', vmin=-25, vmax=25)
    plt.colorbar(map, ax = axes, label = 'Temperature ($C$)')
    axes.invert_yaxis()
    plt.xlabel('Time (years)')
    plt.ylabel('Depth (m)')
    
    #set indexing for final year of results
    loc = int(-365/dt)
    
    #extract the min values over the final year
    winter = heat[:, loc:].min(axis =1)
    summer = heat[:, loc:].max(axis =1)
    
    #Create a temp profile plot
    fig, ax2 = plt.subplots(1,1, figsize=(10,8))
    ax2.plot(winter, x, label = 'Winter')
    ax2.plot(summer, x, label = 'Summer')
    plt.xlim(-8,2)
    plt.ylim(0,70)
    ax2.invert_yaxis()
    plt.xlabel('Temperature (*C)')
    plt.ylabel('Depth (m)')
    plt.legend()
def temp_kanger_warm05(t):
    '''
    For an array of times in days, this function returns a time series of 
    temperatures for Kangerlussuaq, Greenland under a 0.5 degree Celsius
    warming scenario.

    '''
    t_amp = (t_kanger_warm05 - t_kanger_warm05.mean()).max()
    return t_amp*np.sin(np.pi/180 * t - np.pi/2) + t_kanger_warm05.mean()

def kanger_plot_warm05(dt = 10):
    '''
    This function (using a default float "dt" value of 10) plots the 
    results of the kanger_heat_solve equation. This uses the scenario where
    Kangerlussaq, Greenland has been warmed by 0.5 degrees Celsius.
    
    The output of this function is two different plots. The first is a heat 
    map showing the temperature gradient as a function of space (each depth)
    and time. Seasonal variation may be seen at the top of the plot-- the
    surface. The bottom of the plot being red is representative of geothermal
    warming from below the surface.
    
    The second plot is a line plot showing the temperature changing
    with depth (into the ground) in both the winter and summer seasons.
    '''
    #Get soln using solver
    x, time, heat = kanger_heat_solve(init = temp_kanger_warm05)
    
    #Create a figure/axes object
    fig, axes = plt.subplots(1,1)
    
    #Create a color map and add a color bar
    map = axes.pcolor(time/365, x, heat, cmap ='seismic', vmin=-25, vmax=25)
    plt.colorbar(map, ax = axes, label = 'Temperature ($C$)')
    axes.invert_yaxis()
    
    #set indexing for final year of results
    loc = int(-365/dt)
    
    #extract the min values over the final year
    winter = heat[:, loc:].min(axis =1)
    summer = heat[:, loc:].max(axis =1)
    
    #Create a temp profile plot
    fig, ax2 = plt.subplots(1,1, figsize=(10,8))
    ax2.plot(winter, x, label = 'Winter')
    ax2.plot(summer, x, label = 'Summer')
    plt.xlim(-8,2)
    plt.ylim(0,70)
    ax2.invert_yaxis()

def temp_kanger_warm1(t):
    '''
    For an array of times in days, this function returns a time series of 
    temperatures for Kangerlussuaq, Greenland under a 1 degree Celsius
    warming scenario.

    '''
    t_amp = (t_kanger_warm1 - t_kanger_warm1.mean()).max()
    return t_amp*np.sin(np.pi/180 * t - np.pi/2) + t_kanger_warm1.mean()
def kanger_plot_warm1(dt =10):
     '''
    This function (using a default float "dt" value of 10) plots the 
    results of the kanger_heat_solve equation. This uses the scenario where
    Kangerlussaq, Greenland has been warmed by 1 degrees Celsius.
    
    The output of this function is two different plots. The first is a heat 
    map showing the temperature gradient as a function of space (each depth)
    and time. Seasonal variation may be seen at the top of the plot-- the
    surface. The bottom of the plot being red is representative of geothermal
    warming from below the surface.
    
    The second plot is a line plot showing the temperature changing
    with depth (into the ground) in both the winter and summer seasons.
    '''
    #Get soln using solver
    x, time, heat = kanger_heat_solve(init = temp_kanger_warm1)
    
    #Create a figure/axes object
    fig, axes = plt.subplots(1,1)
    
    #Create a color map and add a color bar
    map = axes.pcolor(time/365, x, heat, cmap ='seismic', vmin=-25, vmax=25)
    plt.colorbar(map, ax = axes, label = 'Temperature ($C$)')
    axes.invert_yaxis()
    
    #set indexing for final year of results
    loc = int(-365/dt)
    
    #extract the min values over the final year
    winter = heat[:, loc:].min(axis =1)
    summer = heat[:, loc:].max(axis =1)
    
    #Create a temp profile plot
    fig, ax2 = plt.subplots(1,1, figsize=(10,8))
    ax2.plot(winter, x, label = 'Winter')
    ax2.plot(summer, x, label = 'Summer')
    plt.xlim(-8,2)
    plt.ylim(0,70)
    ax2.invert_yaxis()

def temp_kanger_warm3(t):
    '''
    For an array of times in days, this function returns a time series of 
    temperatures for Kangerlussuaq, Greenland under a 3.0 degree Celsius
    warming scenario.

    '''
    t_amp = (t_kanger_warm3 - t_kanger_warm3.mean()).max()
    return t_amp*np.sin(np.pi/180 * t - np.pi/2) + t_kanger_warm3.mean()
def kanger_plot_warm3(dt =10):
     '''
    This function (using a default float "dt" value of 10) plots the 
    results of the kanger_heat_solve equation. This uses the scenario where
    Kangerlussaq, Greenland has been warmed by 0.5 degrees Celsius.
    
    The output of this function is two different plots. The first is a heat 
    map showing the temperature gradient as a function of space (each depth)
    and time. Seasonal variation may be seen at the top of the plot-- the
    surface. The bottom of the plot being red is representative of geothermal
    warming from below the surface.
    
    The second plot is a line plot showing the temperature changing
    with depth (into the ground) in both the winter and summer seasons.
    '''
    #Get soln using solver
    x, time, heat = kanger_heat_solve(init = temp_kanger_warm3)
    
    #Create a figure/axes object
    fig, axes = plt.subplots(1,1)
    
    #Create a color map and add a color bar
    map = axes.pcolor(time/365, x, heat, cmap ='seismic', vmin=-25, vmax=25)
    plt.colorbar(map, ax = axes, label = 'Temperature ($C$)')
    axes.invert_yaxis()
    
    #set indexing for final year of results
    loc = int(-365/dt)
    
    #extract the min values over the final year
    winter = heat[:, loc:].min(axis =1)
    summer = heat[:, loc:].max(axis =1)
    
    #Create a temp profile plot
    fig, ax2 = plt.subplots(1,1, figsize=(10,8))
    ax2.plot(winter, x, label = 'Winter')
    ax2.plot(summer, x, label = 'Summer')
    plt.xlim(-8,2)
    plt.ylim(0,70)
    ax2.invert_yaxis()
    
    
    