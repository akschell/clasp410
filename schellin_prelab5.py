# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt

radearth = 6357000 #Earth radius in meters
lamb = 100. #Thermal diffusivity of atmosphere
sigma = 5.67e-8 #S-B constant
C = 4.2e6 
rho = 1020 #density
dz = 50 #height/depth

def temp_warm(lats_in):
    '''
    Create a temperature profile for modern day "warm" earth.
    Fits temperature array to any given latitude.
    
    Parameters
    ----------
    lats_in : Numpy array
        Array of latitudes in degrees where temperature is required.
        0 corresponds to the south pole, 180 to the north.
        
    Returns
    --------
    temp : Numpy array
        Temperature in Celsius
    '''
    
    #Set initial temperature curve
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25, 
                      23, 19, 14, 9, 1, -11, -19, -47])
    
    #Get base grid:
    npoints = T_warm.size
    dlat = 180 / npoints #Latitude spacing
    lats = np.linspace(dlat/2., 180-dlat/2., npoints) #Lat cell centers
    
    #Fit a parabola to the above values
    coeffs = np.polyfit(lats, T_warm, 2)
    
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0]*lats_in**2
    return temp

def insolation(S0, lats):
    '''
    

    Parameters
    ----------
    S0 : float
        Solar constant (1370 for typical Earth conditions.)
    lats : Numpy array
        Latitudes to output insolation. Following the grid standards set in
        the diffusion program, polar angle is defined from the south pole.
        In other words, 0 is the south pole, 180 to the north.

    Returns
    -------
    insolation : numpy array
        Insolation returned over the input latitudes.

    '''
    
    max_tilt = 23.5 #tilt of earth in degrees
    
    #create an array to hold insolation
    insolation = np.zeros(lats.size)
    
    #Daily rotation of earth reduces solar constant by distributing the sun
    #energy along a zonal band
    dlong = 0.01
    angle = np.cos(np.pi/180.*np.arange(0,360,dlong))
    angle[angle < 0] = 0
    total_solar = S0*angle.sum()
    S0_avg = total_solar / (360/dlong)
    
    tilt = [max_tilt*np.cos(2.0*np.pi*day/365) for day in range(365)]
    
    #apply to each latitude zone
    for i, lat in enumerate(lats):
        #get solar zenith; do not let it go past 180. Convert to lat
        zen = lat - 90. + tilt
        zen[zen>90] = 90
        insolation[i] = S0_avg*np.sum(np.cos(np.pi/180.*zen)) / 365.
        
    insolation = S0_avg*insolation/365
    return insolation
    
def gen_grid(npoints = 18):
    '''
    A convenience function for creating the grid. Creates a uniform latitudinal
    grid from pole-to-pole.
    
    Parameters
    ----------
    npoints : float, default = 18
        The number of points on the grid.
    
    Returns
    --------
    dlat: float
        Latitude spacing in degrees.
    lats: numpy array
        Latitudes in degrees
    edge: Numpy array
        Latitude bin edges in degrees.
    '''
    
    dlat = 180 / npoints#latitude spacing
    lats = np.linspace(dlat/2., 180-dlat/2., npoints) #lat cell centers
    edge = np.linspace(0, 180, npoints+1) #Lat cell edges
    
    return dlat, lats, edge

def snowearth(npoints = 18, dt = 1, tstop = 10000, lamb = 100., S0 = 1370, 
              emiss = 1.0, albedo = 0.3, dosphere = True, dorad = True, 
              dyn_albedo = False, t_init = None, dogamma = False, gamma = 0):
    '''
    The Snowball Earth model used in the report. Uses gen_grid() to create a
    grid of npoints. Then, initial temperature conditions are either taken
    from an input (t_init) or calculated from the temp_warm() function. 
    
    A tridiagonal matrix A is created with -2s on the diagonal and 1s on either
    side of the diagonal (the top row of the matrix has separate BCs). Another
    matrix B is created to account for the spherical correction term. These
    matrices undergo several linear algebraic manipulations. 
    
    Different corrections (dorad and dosphere) are applied if the kwarg is
    set to True in the input. The model applies other kwargs if True such as 
    gamma correction and dynamic albedo. 
    
    The model returns an array of latitudes and their corresponding temperatures.

    Parameters
    ----------
    npoints : integer, default = 18
        Number of latitude grid points.
    dt : float, default = 1
        Time step.
    tstop : float, default = 10000
        Length of time used.
    lamb: float, default = 100
        Set diffusion coefficient in units.
    S0 : float, default = 1370
        Solar constant.
    emiss: float, default = 1.0
        Emissivity of the Earth.
    albedo: float, default = 0.3
        Albedo of the Earth (ground value).
    dosphere: kwarg, default = True
        If True, spherical correction term used in model.
    dorad: kward, default = True
        If True, radiative forcing term used in model.
    dyn_albedo: kwarg, default = False
        If True, dynamic albedo used in model where the albedo value is 0.6 for
        temperatures below -10 degrees Celsius (snow/ice) and 0.3 for all other
        latitudes.
    t_init: kwarg, default = None
        If not None, input is the temperature initial conditions.
    dogamma: kwarg, default = False
        If True, adds a "solar multiplier" gamma term to insolation to account
        for solar forcing on Snowball Earth.
    gamma: float, default = 0
        Solar multiplier gamma term.


    Returns
    -------
    lats: numpy array
        An array of latitudes in degrees representing solution grid.
    Temp: numpy array
        An array of temperatures on our grid at the end of the simulation
        in Celsius.

    '''
    #Create grid
    dlat, lats, edges = gen_grid(npoints)
    
    #Set (delta)y
    dy = np.pi * radearth * dlat/180
    
    #Set timestep from years to seconds
    dt_sec = dt*365*24*3600
    
    #Set initial condition
    Temp = np.zeros(npoints)
    
    if t_init is None:
        Temp = temp_warm(lats)
    else:
        Temp = np.ones((npoints,))*t_init
        
    #Create tri-diag 'A' matrix
    A = np.zeros((npoints, npoints))
    A[np.arange(npoints), np.arange(npoints)] = -2
    A[np.arange(npoints-1), np.arange(npoints-1)+1] = 1
    A[np.arange(npoints-1)+1, np.arange(npoints-1)] = 1
    
    #Apply zero-flux BCs
    A[0,0] = A[-1,-1] = -2
    A[0,1] = A[-1,-2] = 2
    A *= (1/dy)**2
    
    #Get matrix for advancing solution
    L = np.eye(npoints) - dt_sec*lamb*A
    Linv = np.linalg.inv(L)
    
    #Create matrix 'B' to assist with adding spherical-correction factor
    #Corner values for 1st order accurate Neumann boundary conditions
    B = np.zeros((npoints, npoints))
    B[np.arange(npoints-1), np.arange(npoints-1)+1] = 1
    B[np.arange(npoints-1)+1, np.arange(npoints-1)] = -1
    B[0,:] = B[-1,:] = 0
    
    #Set the surface area of the "side" of each lat ring at bin center
    Axz = np.pi * ((radearth+50.0)**2 - radearth**2)*np.sin(np.pi/180.*lats)
    #Find dAxz/dlat-- this never changes
    dAxz = np.matmul(B, Axz) / (Axz * 4 * dy**2)
    
    #Get total number of steps
    nsteps = int(tstop/dt)
    
    #Set insolation
    if dogamma:
        insol = gamma*insolation(S0, lats)
    else:
        insol = insolation(S0, lats)
    
    albedo_ice = 0.6
    albedo_gnd = 0.3
    albedo = np.ones((npoints,))
    if dyn_albedo:
        # Update albedo based on conditions:
        loc_ice = Temp <= -10
        albedo[loc_ice] = albedo_ice
        albedo[~loc_ice] = albedo_gnd
    else:
        albedo *= 0.3
    
    #Solve!
    for i in range(nsteps):
        #Add spherical correction:
        if dosphere:
            spherecorr = lamb*dt_sec*np.matmul(B, Temp)*dAxz
        else:
            spherecorr = 0
        #Add source terms:
        Temp += spherecorr
        
        #Add insolation term:
        if dorad:
            radiative = (1/(rho*C*dz))*((1-albedo)*insol - emiss*sigma*(Temp+273)**4)
        else:
            radiative = 0
      
        Temp += dt_sec*radiative
        
        #Advance solution.
        Temp = np.matmul(Linv, Temp)
        
    return lats, Temp

#Lab Report Questions

def question_1():
    '''
    This is a helper function for Question 1 on the lab report. It prints
    temperature curves for present-day Earth (Initial conditions), a curve
    for the Snowball Earth model using only basic diffusion, a curve for the 
    model using diffusion with a spherical correction, and a curve from the 
    model using diffusion with spherical and radiative forcing corrections. 
    '''
    lats, Temp = snowearth()
    warm = temp_warm(lats)
    
    nsnrlats, nsnrTemp = snowearth(dosphere = False, dorad = False)
    nrlats, nrTemp = snowearth(dorad = False)
    
    plt.plot(lats, warm, label = 'Initial Conditions')
    plt.plot(nrlats, nrTemp, label = 'Diff + SphCorr')
    plt.plot(lats, Temp, label = 'Diff + SphCorr + Rad')
    plt.plot(nsnrlats, nsnrTemp, label = 'Basic Diff')
    
    plt.legend(loc = 'best')
    plt.xlabel('Latitude')
    plt.ylabel('Temperature (Celsius)')

def question_2plot():
    '''
    This is a helper function for Question 2 of the lab report. This function
    independently varies the lambda and emissivity values in the Snowball Earth
    model to determine how each parameter separately impacts the resulting
    temperature curve. 
    '''
    #tune model to get green line in lab statement to match blue line
    vlats, vTemp = snowearth(lamb = 0.)
    vlats1, vTemp1 = snowearth(lamb = 50.)
    vlats2, vTemp2 = snowearth(lamb = 150.)
    vlats3, vTemp3 = snowearth(lamb = 0., emiss = 0.7)
    vlats4, vTemp4 = snowearth(lamb = 50., emiss = 0.7)
    vlats5, vTemp5 = snowearth(lamb = 100., emiss = 0.7)
    vlats6, vTemp6 = snowearth(lamb = 150., emiss = 0.7)
    lats, Temp = snowearth()
    warm = temp_warm(lats)
    
    plt.plot(lats, warm, label = 'Initial Conditions')
    plt.plot(vlats, vTemp, label = 'lamb = 0, emiss = 1.0')
    plt.plot(vlats1, vTemp1, label = 'lamb=50, emiss = 1.0')
    plt.plot(lats, Temp, label = 'lamb = 100, emiss = 1.0')
    plt.plot(vlats2, vTemp2, label = 'lamb = 150, emiss = 1.0')
    plt.plot(vlats3, vTemp3, label = 'lamb = 0, emiss = 0.7')
    plt.plot(vlats4, vTemp4, label = 'lamb = 50, emiss = 0.7')
    plt.plot(vlats5, vTemp5, label = 'lamb = 100, emiss = 0.7')
    plt.plot(vlats6, vTemp6, label = 'lamb = 150, emiss = 0.7')
    
    plt.legend(loc = 'best')
    plt.xlabel('Latitude')
    plt.ylabel('Temperature (Celsius)')

def question_2_play():
    '''
    This is another helper function for Question 2 of the lab report. In this
    helper function, I am playing around with the lambda and emissivity values
    until I get a curve that closely resembles the initial conditions 
    (present-day Earth) curve. 
    '''
    lats, Temp = snowearth()
    warm = temp_warm(lats)
    
    vlats, vTemp = snowearth(lamb = 45., emiss = 0.72)
    plt.plot(lats, warm, label = 'Initial Conditions')
    plt.plot(vlats, vTemp, label = 'lamb = 45, emiss = 0.72')

    plt.legend(loc = 'best')
    plt.xlabel('Latitude')
    plt.ylabel('Temperature (Celsius)')
    plt.show()

    
def question_3():
    '''
    This is a helper function for Question 3 of the lab report. In this function,
    the initial conditions are being varied and plotted-- one for a hot Earth
    condition (all temps at 60 Celsius), one for a cold Earth condition 
    (all latitudes have temp of -60 Celsius), and one for a flash-frozen Earth
    (albedo is suddenly changed increased as Earth is frozen). 
    '''
    lats, Temp = snowearth(albedo = 0.6, lamb = 45, emiss = 0.72)
    warm = temp_warm(lats)
    
    wlats, wTemp = snowearth(lamb = 45, emiss = 0.72, dyn_albedo = True, t_init = 60)
    clats, cTemp = snowearth(lamb = 45, emiss = 0.72, dyn_albedo = True, t_init = -60)
    
    plt.plot(wlats, wTemp, label = 'Hot solution', color = 'red')
    plt.plot(clats, cTemp, label = 'Cold solution', color = 'blue')
    plt.plot(lats, warm, label = 'Flash Freeze')
    plt.xlabel('Latitude')
    plt.ylabel('Temperature (Celsius)')
    plt.legend(loc = 'best')
    plt.show()
    
def question_4():
    '''
    This is a helper function for Question 4 of the lab report. In this function,
    the solar multiplier term is being varied dynamically. The first run of this
    function does the normal model under "cold Earth" conditions. The "for" 
    loop in this function means that for each subsequent run, the gamma value
    increases by 0.05 and the initial conditions used in the model are the 
    same as the temperatures outputted by the last iteration. The function
    does this for both increasing and decreasing gamma, and then plots the
    average global temperature with its corresponding gamma value. 
    '''
    lats, Temp = snowearth(lamb = 45, emiss = 0.72, dyn_albedo = True,
                            t_init = -60, dogamma = True, gamma = 0.40)
    x = []
    y = []
    x2 = []
    y2 = []
    for gamma in np.arange(0.4, 1.45, 0.05):
        lats, Temp = snowearth(lamb = 45, emiss = 0.72, dyn_albedo = True,
                                t_init = Temp, dogamma = True, gamma = gamma)
        y.append(np.average(Temp))
        x.append(gamma)

    for gamma in np.arange(1.4, 0.35, -0.05):
        lats, Temp = snowearth(lamb = 45, emiss = 0.72, dyn_albedo = True,
                                t_init = Temp, dogamma = True, gamma = gamma)
        y2.append(np.average(Temp))
        x2.append(gamma)
    
    
    plt.plot(x, y, label = 'Increasing Gamma')
    plt.plot(x2, y2, label = 'Decreasing Gamma')
    plt.legend(loc = 'best')
    plt.xlabel('Gamma')
    plt.ylabel('Temperature (Celsius)')
    plt.legend(loc = 'best')
    plt.show()

