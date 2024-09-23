# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 21:51:36 2022

@author: jwang
"""

# DPS formulation
# Input: sea-level rise (water level), year_offset, .

import math

####################################
## get dike heightening for time t
## based on calculating state varaibles, \
## BH and FH, and then heightening
## This needs to be run for each reach in each year
####################################

def calc_heightening(t, prev_h, water_level, n, year_offset, x, r, w):
    
    # set the previous height as h_0; otherwise, set it as the previous timestep
    prev_sl = water_level[-1+year_offset]
#     prev_h = h_0
#     if t > 0:
#         prev_h = prev_h
    
    ####################################
    ## fit the simple linear regression based on
    ## past n-year water level data. The coefficients
    ## are used as state variables to describe states## 
    ####################################
    
    sx = 0
    sy = 0
    sxx = 0
    sxy = 0
    ssr = 0
    for i in range(n):
        ind = year_offset - n + i
        sx += i
        sy += water_level[ind]
        sxx += i * i
        sxy += i * water_level[ind]
    slope = (sxy - (sx * sy)/n) / (sxx - (sx * sx)/n)
    intercept = (sy/n) - slope * (sx/n)
    for i in range(n):
        obs = water_level[year_offset-n+i]
        yfit = intercept + slope*i
        ssr += (yfit-obs)*(yfit-obs)
    mean_slr_rate = slope
    srss = math.sqrt(ssr)
    #obs = np.zeros(n)
    #for i in range(n):
    #    obs[i] = water_level[t+year_offset-n+i]
    #mean_slr_rate, srss = state_variables(obs,n)

    ####################################
    ## calculate the freeboard height and 
    ## buffer height for time t with given DPS decision variables## 
    ####################################
    FH_t = 0
    BH_t = 0
    FH_t += x[0] + r[0] * mean_slr_rate + w[0] * mean_slr_rate * mean_slr_rate
    FH_t += x[1] + r[1] * srss + w[1] * srss * srss
    BH_t += x[2] + r[2] * mean_slr_rate + w[2] * mean_slr_rate * mean_slr_rate
    BH_t += x[3] + r[3] * srss + w[3] * srss * srss
        
    if FH_t < 0:
        FH_t = 0
        
    if BH_t < 0:
        BH_t = 0
        
    ####################################
    ## calculate dike heightenings## 
    ####################################
    test_height = prev_h - BH_t
    u_t = 0
    if prev_sl > test_height:
        safe_height = prev_sl - test_height
        u_t = safe_height + FH_t

    return BH_t, FH_t, u_t