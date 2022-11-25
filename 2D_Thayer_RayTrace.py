import numpy as np

# calculates the slant total delay and geometric bending effect over a given number of height levels
def get_RayTrace2D_Thayer_global(e_outgoing, stat_height, h_lev_all, h_diff, nr_h_lev_all, start_lev, n):

    # conversion factor for radians to degrees
    rad2deg = 180/np.pi
    # conversion factor for degrees to radians
    deg2rad = np.pi/180
    # constant for iteration accuracy of outgoing elevation angle
    accuracy_elev = 1e-7
    # variable for storing the geocentric heights of height levels (h_lev with added radius of the Earth at station position)
    R = [] # in [m]
    # variable for the slant total refractive index at the intersection point in the original height level (no mean value between two levels)
    n_int = n
    # variable for the coefficient A
    A = [0]
    # variable for storing the a priori bending effect
    ap_bend = 0
    # variable for storing the first theta (location dependend elevation angle) = "theoretical" outgoing elevation angle + a priori bending effect
    theta_start = 0 # in [rad]
    # variable for counting the number of iteration loops when calculating the outgoing elevation angle
    loop_elev = 0
    # variable for storing the mean coordinate accuracy (mean difference in the intersection point position result from the current and the last iteration step)
    epsilon_layer = 0
    # variable for counting the number of iteration loops when calculating the intersection point
    loop_layer = 0
    # variable for storing all height differences between two consecutive levels
    dh = [] # in [m]
    # loop to add all of the height differences between layers
    for i in range(nr_h_lev_all - 1):
        dh.append(h_diff)
    # variable for storing the slant distance for each ray path section between two consecutive levels used for slant delay calculation
    s = [] # in [m]
    ## determine output of number of height levels starting at station level up to highest supported height level
    nr_h_lev = nr_h_lev_all #- start_lev + 2 # note: +2 as station height level and "start_lev" are also needed
    # variable for storing the radius of the Earth at the station position
    R_e = 6371423 # in [m]
    ## Note: Adding the Earth radius calculated at the station's latitude to all height levels is sufficient, although it is an approximation as the radius changes as the ray path alters its latitude
    # Therefore a strict solution would need an iterative step to do ray-tracing, re-determine the radius and do ray-tracing and so on until a certain accuracy would be reached.
    # variable for storing first radius value at station position
    R.append(R_e + h_lev_all[0]) # in [m]
    # variable for elevation (location dependend)
    theta = [] # in [rad]
    # variable for elevation (fixed reference to station position)
    e = [] # in [rad]
    # variable for the geocentric angle          
    anggeo = [0] # in [rad]
    # variable for the interpolated (at intersection point) mean total refractive indices          
    n_total = []
    # calculate a priori bending effect, see Hobiger et al. 2008, Fast and accurate ray-tracing algorithms for real-time space geodetic applications using numerical weather models, equation (32) on page 9
    ap_bend = 0.02 * np.exp(-stat_height / 6000) / np.tan(e_outgoing) * deg2rad # in [rad], conversion is necessary
    # set initial theta (= location dependend) elevation angle and add it to the list of angles
    # first theta = "theoretical" outgoing elevation angle + a priori bending effect, see scriptum Atmospheric Effects in Geodesy 2012, equation (2.61) on page 36
    theta_start = e_outgoing + ap_bend # in [rad]
    theta.append(theta_start)
    # define first value for "diff_e" (difference between "theoretical" outgoing elevation angle and ray-traced outgoing elevation angle) used for iteration decision. Note: make sure the initial value is higher than the value of "accuracy_elev"
    diff_e = 100 * accuracy_elev ## in [rad]
    # initialize variable for counting the loop number for calculating the outgoing elevation angle
    loop_elev = 0
        
    # loop over all remaining height levels
    for i in range(1,nr_h_lev-1):
        # add the new radius at the current height level to the list of radii
        R.append(R[0]+h_lev_all[i])  
        # calculate intersection points using modified "Thayer" Approximation
        # calculate elevation angle of ray path at intersection point with current height level
        theta.append(np.arccos( R[i-1] * n_int[i-1] * np.cos(theta[i-1]) / ( R[i] * n_int[i] ) )) # in [rad]
        # calculate geocentric angle to intersection point in current height level
        anggeo.append(anggeo[i-1] + ( theta[i] - theta[i-1] ) / ( 1 + A[i-1] )) # in [rad]
        # calculate the new value for A in the current layer
        A.append(( np.log(n_int[i]) - np.log(n_int[i-1]) ) / ( np.log(R[i]) - np.log(R[i-1]) ))
        # difference between "theoretical" outgoing elevation angle and ray-traced outgoing elevation angle
        # note: The ray-traced outgoing elevation angle is the elevation angle value of the uppermost ray-traced level, which is using equation (2.72) from scriptum Atmospheric Effects in Geodesy 2012 on page 37: theta(uppermost level) - anggeo(uppermost level).
        diff_e = e_outgoing - (theta[-1] - anggeo[-1]) # in [rad]
        # determine new starting elevation angle at the station based on the difference between "theoretical" and ray-traced outgoing elevation angle
        theta_start = theta_start + diff_e # in [rad]
        # calculate elevation angle (fixed reference at first ray point = station position), see scriptum Atmospheric Effects in Geodesy 2012, equation (2.72) on page 37
        e.append(theta[i] - anggeo[i]) # in [rad]
    
    # get starting elevation angle at station
    e_stat = theta[0] # in [rad]
    # define outgoing elevation angle (location independent). Note: this is the elevation angle value of the uppermost ray-traced level
    e_outgoing_rt = e[-1] # in [rad]
    # loop over all remaining height levels
    for i in range(1,nr_h_lev - 1):
        # determine mean total slant refractive indices between two consecutive levels for slant delay calculation using the slant ray path distances
        n_total.append(( n_int[i] + n_int[i-1] ) / 2)
        # note: as earth radius "R_e" is constant, it is sufficient to directly use the "h_lev" values
        dh.append( h_lev_all[i] - h_lev_all[i-1])
        # calculation of s for the space between two levels
        s.append(R[i-1] * n_int[i-1] * np.cos(theta[i-1]) / ( 1 + A[i] ) * ( np.tan(theta[i]) - np.tan(theta[i-1]) ))
    
    # calculate slant delays
    # determine if the elevation at the station is smaller than a specific value, e.g. 89.99°
    # --> calculated slant path (formula: Thayer 1967, equation (17) can be used to calculate the slant delay
    if (e_stat * rad2deg) < 89.99:
        # calculate slant total delay 
        # note: size of refractive index vectors and dh is equal (nr_h_lev - 1)
        ds_total = np.dot(np.array(n_total)-1, s) # in [m]
    # otherwise set the slant delay equal to the zenith delay value
    else:
        # set slant total delay
        ds_total = 0 # in [m]
    
    # Calculate geometric bending effect, see scriptum Atmospheric Effects in Geodesy 2012, equation (2.75) on page 38
    dgeo = 0
    for i in range(1,nr_h_lev - 2):
        dgeo += (s[i] - (np.cos(e[i] - e_outgoing_rt) * s[i])) # in [m]
    
    # calculate slant total delay + geometric bending effect
    ds_total_geom = ds_total + dgeo # in [m]
    
    # print the delays to console for easy review
    print("Slant total delay (m):",ds_total)
    print("Geometric bending effect (m):",dgeo)
    print("Combined delay (m):",ds_total_geom)
    return ds_total_geom
        
# calculates the refractive indices for a predefined number of 1 km height levels above sea level
def n_gen(stat_height, h_lev_all, h_diff, nr_h_lev_all): 

    n = [] # list of refractive indices to be generated
    k1 = 77.60 # in [K/mb], refractivity constant from Thayer (1974)
    k2 = 64.9 # in [K/mb], refractivity constant from Thayer (1974)
    k3 = 3.776*10**5 # in [K**2/mb], refractivity constant from Thayer (1974)
    pv = [] # in [mb], list of water vapor partial pressures
    pd = [] # in [mb], list of dry air partial pressures
    T = [] # in [K], list of temperatures to be generated
    Tc = [] # in [C], list of temperatures to be generated
    
    for i in range(1,nr_h_lev_all): # loop over all heights above the station
        h_lev_all.append(i*h_diff) # add 1 km to the list of heights
        #height level cutoffs below are determined from the International Standard Atmosphere
        if i*h_diff <= 11000: # 0 - ground to tropopause
            # calculate temperature at the current height level
            T.append(288.15 - 0.0065 * i*h_diff) # in [K]
            Tc.append(20 - 0.0065 * i*h_diff) # in [C]
            # calculate partial pressures of water vapor and dry air at the current height level
            if (288.15 - 0.0065 * i*h_diff) >= 273.15:
                pv_i = 6.11*np.exp(17.27*(20 - 0.0065 * i*h_diff)/(237.3+(20 - 0.0065 * i*h_diff))) # in [mb]
            else:
                pv_i = 6.11*np.exp(21.875*(20 - 0.0065 * i*h_diff)/(265.5+(20 - 0.0065 * i*h_diff))) # in [mb]
            pv.append(pv_i)
            pd.append(1013.25 * ( (288.15 + (i*h_diff - 0) * 0.0065) / 288.15)**(-9.81*0.0289644/(8.3144598*0.0065)) - pv_i) # in [mb]
        elif i*h_diff <= 20000: # 1 - tropopause to stratosphere1
            T.append(216.65)
            Tc.append(-56.5)
            pd.append(226.321*np.exp( (-9.81*0.0289644*(i*h_diff - 11e3)/(8.3144598*216.65))))
            pv.append(0) # water vapor pressure is set to zero above the tropopause to simplify calculations due to most water vapor existing below it
        elif i*h_diff <= 32000: # 2 - stratosphere1 to stratosphere2
            T.append(216.65 + 0.001*i*h_diff)
            Tc.append(-56.5 + 0.001*i*h_diff)
            pd.append(54.749 * ( (216.65 + (i*h_diff - 20e3) * -0.001) / 216.65)**(-9.81*0.0289644/(8.3144598*-0.001)) - pv_i)
            pv.append(0)
        elif i*h_diff <= 47000: # 3 - stratosphere2 to stratopause
            T.append(228.65 + 0.0028*i*h_diff)
            Tc.append(-44.5 + 0.0028*i*h_diff)
            pd.append(8.6802 * ( (228.65 + (i*h_diff - 32e3) * -0.0028) / 228.65)**(-9.81*0.0289644/(8.3144598*-0.0028)) - pv_i)
            pv.append(0)
        elif i*h_diff <= 51000: # 4 - stratopause to mesophere1
            T.append(270.65)
            Tc.append(-2.5)
            pd.append(1.1091*np.exp( (-9.81*0.0289644*(i*h_diff - 47e3)/(8.3144598*270.65))))
            pv.append(0)
        elif i*h_diff <= 71000: # 5 - mesosphere1 to mesosphere2
            T.append(270.65 - 0.0028*i*h_diff)
            Tc.append(-2.5 - 0.0028*i*h_diff)
            pd.append(0.66939 * ( (270.65 + (i*h_diff - 51e3) * 0.0028) / 270.65)**(-9.81*0.0289644/(8.3144598*0.0028)) - pv_i)
            pv.append(0)
            
    for i in range(nr_h_lev_all - 1):
        # inverse compressibility factor for dry air from Thayer (1974)
        Zd_inv = 1+pd[i]*((57.90*10**(-8))*(1+0.52/T[i])-(9.4611*10**(-4))*Tc[i]/T[i]**2) 
        # inverse compressibility factor for water vapor from Thayer (1974)
        Zv_inv = 1+1650*(pv[i]/T[i]**3)*(1-0.01317*Tc[i]+1.75*10**(-4)*Tc[i]**2+1.44*10**(-6)*Tc[i]**3) 
        # refractivity and refractive index from the above variables
        N = k1*pd[i]/T[i]*Zd_inv+k2*pv[i]/T[i]*Zv_inv+k3*pv[i]/T[i]**2*Zv_inv
        n.append(N*10**(-6)+1)
        
    return n
        
def main():
    
    # variable for storing outgoing (vacuum) elevation angle. Note: this value is arbitrary and can range from >pi/18 to <pi/2
    e_outgoing = np.pi/4 # in [rad] 
    # variable for storing (ellipsoidal) height of station in [m]
    stat_height = 0 # in [m]
    # variable for storing (ellipsoidal) height levels in which the intersection points with the ray path should be estimated; in [m]
    h_lev_all = [stat_height] # in [m]
    # variable for storing the height difference between levels for constant differences
    h_diff = 1e3 # in [m]
    # variable for storing total number of available height levels
    nr_h_lev_all = 72
    # variable for storing index of first height level in "h_lev_all" above station height (needed for ray-tracing start above station)
    start_lev = 1
    # variable for storing list of refractive indices from the n_gen function
    n = n_gen(stat_height, h_lev_all, h_diff, nr_h_lev_all)
    # run the ray-tracing function
    get_RayTrace2D_Thayer_global(e_outgoing, stat_height, h_lev_all, h_diff, nr_h_lev_all, start_lev, n)

if __name__ == "__main__":
    main()
