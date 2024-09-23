import os
import math
import sys
import json

import numpy as np
from scipy.optimize import brentq as root
import pandas as pd
import time
import csv

import storms as stm
import jpmos as jpm
import cost_model_09_27 as c_mdl_dps
import metrics_full_interpolation_dps as mtc_dps
import rain as rain
import copy
import time_series_storms as tss
import calc_heightening as ch
#import larose_function_variations_09_14 as lfv


def larose_future_intertemporal(sea_lvl, year_offset, intensity, frequency_mod, crest_height_upgrade, needed_arguments, calc_cost, oldFragFlag,\
                                 planning_horizon, discount_rate, prev_h, year):
    
    
    reach_object, dmg_data, nms_data, bfe_data, nsc_data, storm_params, \
    locParams, surgeSurfaceParams, waveSurfaceParams, wavePeriodSurfaceParams, \
    sigmaSurfaceParams, radii, historic_theta_bar, historic_theta_var, \
    historic_x_freq, lon, polderObject, MCIterates, base_frequency, \
    unit_price, rainfall_modifier, base_crest_heights, partic_rate, pop_mult, \
    ns_cost_mult, acq_cost_mult, \
    str_cost_mult, NS_std, acq_threshold, res_fp, rain_params, historic_c_p_a0, \
    historic_c_p_a1, reach_df  = copy.deepcopy(needed_arguments)
    
    
    
    #adjust crest heights
   # print("started call")
    #we want to include values rounded up and down to 1 decimal point, and
    #when we have something with exactly one decimal point we need the surrounding values
    #regardless. And there are some odd numerical precision issues forcing us to go
    #slightly wide
    
    active_dmg_data = dmg_data[(dmg_data.partic <= partic_rate + 0.11) &\
                               (dmg_data.partic >= partic_rate - 0.11) &\
                               (dmg_data.pop_mult <= pop_mult + 0.11)&\
                               (dmg_data.pop_mult >= pop_mult - 0.11)&\
                               (dmg_data.ns_std <= NS_std + 0.11)&\
                               (dmg_data.ns_std >= NS_std - 0.11)]

    
    
    frequency = base_frequency * (1 + frequency_mod)

    
    upgraded_reach = copy.deepcopy(reach_object)
    current_h = prev_h + crest_height_upgrade
    #upgraded_reach['height'] = reach_object['reachHeight'] + crest_height_upgrade
    upgraded_reach['reachHeight'] = current_h
    #print(current_h)
    #for overtopping and fragility calculations we must treat sections
    #of reaches with flood walls as separate from sections without floodwalls

    
    #####
    #storms
    
    swe_dist_by_storm = pd.DataFrame(columns = ['depths','depth_probs','storm_id']) #initialize empty dataframe
    storm_ids = storm_params.index
    
    prob_by_storm = jpm.get_storm_probs(intensity, radii, historic_theta_bar, historic_theta_var,\
                                         historic_x_freq, storm_params, lon, historic_c_p_a0, historic_c_p_a1)
    
    # storms ids that will occur
    num_storms, storms_occur = tss.time_series_storms(frequency, storm_ids, prob_by_storm)
    #print(storms_occur)

    # loop in the storms occuring
    for storm_id in storms_occur:
        this_storm = storm_params.loc[storm_id,:]
        surgeObject = stm.construct_surge_objects(this_storm,\
                                                   locParams.loc[storm_id,:],\
                                                   surgeSurfaceParams, waveSurfaceParams,\
                                                   wavePeriodSurfaceParams,sigmaSurfaceParams.loc[storm_id,:],\
                                                   sea_lvl, dps_flag = False) #make sure it references id and not row
        #construct_surge_objects is partially written but needs some functions that others have been working on,
        #please see the storm file - NG
        #also I'm making some assumptions about column names in storm_params, loc_params, need to adjust in the storm file
        #when those are settled - NG

        rainfall = rain.predict_rainfall_generic(rain_params,this_storm['c_p'],this_storm['delta_p'],\
                                                 this_storm['radius'],this_storm['lon'],this_storm['angle'])*(1+rainfall_modifier)
        
        flood_elevs = stm.calcFloodElevs(upgraded_reach,surgeObject,rainfall,polderObject,storm_id,MCIterates,oldFragFlag = oldFragFlag) #add column with storm id [wait, no, we have storm_id already from the loop - NGs]
        
        #swe_dist_by_storm = swe_dist_by_storm.concat(flood_elevs, ignore_index = True)
        swe_dist_by_storm = pd.concat([swe_dist_by_storm, flood_elevs], ignore_index = True)
    #print(swe_dist_by_storm)
    #initalize the weighted damage
    wtd_dmg = 0
    
    #set up the minimal swe
    min_swe = -13
    

    for ind in range(len(swe_dist_by_storm)):
        swe = max(swe_dist_by_storm['depths'][ind], min_swe)
        wtd_dmg += mtc_dps.dmg_calc_interpolation(active_dmg_data,partic_rate,pop_mult,swe,NS_std) * (swe_dist_by_storm['depth_probs'][ind])

    #print(wtd_dmg)
    return (wtd_dmg, current_h)


#sea_lvl_scenarios = [0.46, 0.63, 0.83]
#intensity_scenarios = [0.1, 0.125, 0.15]
#frequency_scenarios = [0, -0.14, -0.28]
def larose_future_dps_simplified_realistic(sea_lvl, water_level, n, year_offset, intensity, frequency_mod, x, r, w, needed_arguments, oldFragFlag,\
                   planning_horizon, discount_rate, prev_h, year, n_reach, upgrades):
   
    
    reach_object, dmg_data, nms_data, bfe_data, nsc_data, storm_params, \
    locParams, surgeSurfaceParams, waveSurfaceParams, wavePeriodSurfaceParams, \
    sigmaSurfaceParams, radii, historic_theta_bar, historic_theta_var, \
    historic_x_freq, lon, polderObject, MCIterates, base_frequency, \
    unit_price, rainfall_modifier, base_crest_heights, partic_rate, pop_mult, \
    ns_cost_mult, acq_cost_mult, \
    str_cost_mult, NS_std, acq_threshold, res_fp, rain_params, historic_c_p_a0, \
    historic_c_p_a1, reach_df  = copy.deepcopy(needed_arguments)
    
    
    
    #adjust crest heights
   # print("started call")
    #we want to include values rounded up and down to 1 decimal point, and
    #when we have something with exactly one decimal point we need the surrounding values
    #regardless. And there are some odd numerical precision issues forcing us to go
    #slightly wide
    
    active_dmg_data = dmg_data[(dmg_data.partic <= partic_rate + 0.11) &\
                               (dmg_data.partic >= partic_rate - 0.11) &\
                               (dmg_data.pop_mult <= pop_mult + 0.11)&\
                               (dmg_data.pop_mult >= pop_mult - 0.11)&\
                               (dmg_data.ns_std <= NS_std + 0.11)&\
                               (dmg_data.ns_std >= NS_std - 0.11)]

    
    
    frequency = base_frequency * (1 + frequency_mod)

    
    #upgraded_reach = copy.deepcopy(reach_object)
    #upgraded_reach['height'] = reach_object['reachHeight'] + crest_height_upgrade
       
    #for overtopping and fragility calculations we must treat sections
    #of reaches with flood walls as separate from sections without floodwalls

    
    #####
    #storms
    
    swe_dist_by_storm = pd.DataFrame(columns = ['depths','depth_probs','storm_id']) #initialize empty dataframe
    storm_ids = storm_params.index
    
    prob_by_storm = jpm.get_storm_probs(intensity, radii, historic_theta_bar, historic_theta_var,\
                                         historic_x_freq, storm_params, lon, historic_c_p_a0, historic_c_p_a1)
    
    # storms ids that will occur
    num_storms, storms_occur = tss.time_series_storms(frequency, storm_ids, prob_by_storm)
    #print(storms_occur)
    
    if num_storms != 0:
        surgeHeight = np.zeros((num_storms, n_reach))

    # loop in the storms occuring
        i = 0
        for storm_id in storms_occur:
            this_storm = storm_params.loc[storm_id,:]
            surgeHeight[i, :] = stm.pred_surge_wave_height(this_storm,\
                                                     locParams.loc[storm_id,:],\
                                                     surgeSurfaceParams,\
                                                     sea_lvl[year+n], dps_flag = True)
            i = i + 1
        # if multiple storms occur, return the highest surge heights
        surgeMax = (surgeHeight.max(axis = 0)) * 0.3048
        
    else: # if no storms occur, return the sea level rise
    
        surgeMax = np.full(shape = n_reach, fill_value = sea_lvl[year+n])
        
    upgraded_reach = copy.deepcopy(reach_object)
    current_h = upgrades + prev_h
    upgraded_reach['reachHeight'] = current_h
    
    
    #print(current_h)
    for storm_id in storms_occur:
        this_storm = storm_params.loc[storm_id,:]
        surgeObject = stm.construct_surge_objects(this_storm,\
                                                   locParams.loc[storm_id,:],\
                                                   surgeSurfaceParams, waveSurfaceParams,\
                                                   wavePeriodSurfaceParams,sigmaSurfaceParams.loc[storm_id,:],\
                                                   sea_lvl[year+n], dps_flag = False) #make sure it references id and not row
        #construct_surge_objects is partially written but needs some functions that others have been working on,
        #please see the storm file - NG
        #also I'm making some assumptions about column names in storm_params, loc_params, need to adjust in the storm file
        #when those are settled - NG

        rainfall = rain.predict_rainfall_generic(rain_params,this_storm['c_p'],this_storm['delta_p'],\
                                                 this_storm['radius'],this_storm['lon'],this_storm['angle'])*(1+rainfall_modifier)
        
        flood_elevs = stm.calcFloodElevs(upgraded_reach,surgeObject,rainfall,polderObject,storm_id,MCIterates,oldFragFlag = oldFragFlag) #add column with storm id [wait, no, we have storm_id already from the loop - NGs]
        
        # retired for pandas2
        #swe_dist_by_storm = swe_dist_by_storm.append(flood_elevs, ignore_index = True)
        swe_dist_by_storm = pd.concat([swe_dist_by_storm, flood_elevs], ignore_index = True)
    #print(swe_dist_by_storm)
    #initalize the weighted damage
    wtd_dmg = 0
    
    #set up the minimal swe
    min_swe = -13
    

    for ind in range(len(swe_dist_by_storm)):
        swe = max(swe_dist_by_storm['depths'][ind], min_swe)

        wtd_dmg += mtc_dps.dmg_calc_interpolation(active_dmg_data,partic_rate,pop_mult,swe,NS_std) * (swe_dist_by_storm['depth_probs'][ind])

    return (wtd_dmg, current_h, surgeMax)



