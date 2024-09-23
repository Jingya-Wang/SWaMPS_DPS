#from borg import *

directory = '/work/disasters/jw' #find actual file path later
dd = directory + '/dataforSnyder'

#from rhodium import *

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
import cost_model_09_27 as c_mdl
import metrics_full_interpolation_dps as mtc_dps
import rain as rain
import copy
from numba import jit
import larose_function_variations_09_27 as lfv
import calc_heightening as ch
import discount as dc
import time_series_storms as tss



#np.random.seed(19)

dd_2 = '/work/disasters/jw/swamps_DPS/'
#slr_rcp26 = pd.read_csv(dd_2 + 'RCP26_slr.csv', header = None)
#slr_rcp45 = pd.read_csv(dd_2 + 'RCP45_slr.csv', header = None)
#slr_rcp85 = pd.read_csv(dd_2 + 'RCP85_slr.csv', header = None)

#slr = pd.concat([slr_rcp26, slr_rcp45, slr_rcp85])

slr = pd.read_csv(dd_2 + '3_slr_sows_01_18.csv', header = None)
sea_level_rise = slr.to_numpy()
sea_level_rise = sea_level_rise[2,:]

#this could be slr or time series of storms
n_sow = 10000

#Read in needed data from config file, include dd as argument
dmg_data, nms_data, bfe_data, nsc_data = mtc_dps.get_analytica_data(dd)

rain_params = rain.get_rainfall_surface(dd)

storm_params = stm.get_storm_params(dd)
storm_params = storm_params.sort_values('storm_id')
storm_params = storm_params.set_index('storm_id')
storm_params['storm_id'] = storm_params.index

locParams = stm.get_loc_params(dd)
locParams = locParams.sort_values(['storm_id','reach_id'])
locParams = locParams.set_index('storm_id')
locParams['storm_id'] = locParams.index

locParams = stm.get_loc_params(dd)
locParams = locParams.sort_values(['storm_id','reach_id'])
locParams = locParams.set_index(['storm_id','reach_id'])


surgeSurfaceParams = stm.get_surge_surface_params(dd)
surgeSurfaceParams = surgeSurfaceParams.sort_values(['reach_id'])
surgeSurfaceParams = surgeSurfaceParams.set_index('reach_id')
surgeSurfaceParams['reach_id'] = surgeSurfaceParams.index

waveSurfaceParams = stm.get_wave_surface_params(dd)
waveSurfaceParams = waveSurfaceParams.sort_values(['reach_id'])
waveSurfaceParams = waveSurfaceParams.set_index('reach_id')
waveSurfaceParams['reach_id'] = waveSurfaceParams.index


wavePeriodSurfaceParams = stm.get_wave_period_surface_params(dd)
wavePeriodSurfaceParams = wavePeriodSurfaceParams.sort_values(['reach_id'])
wavePeriodSurfaceParams = wavePeriodSurfaceParams.set_index('reach_id')
wavePeriodSurfaceParams['reach_id'] = wavePeriodSurfaceParams.index

sigmaSurfaceParams = stm.sigma_coefs_config(dd)
storms_and_tracks = storm_params.loc[:,['track','storm_id']]
sigmaSurfaceParams = storms_and_tracks.merge(sigmaSurfaceParams, on = 'track')
sigmaSurfaceParams = sigmaSurfaceParams.sort_values(['storm_id','reach_id'])
sigmaSurfaceParams = sigmaSurfaceParams.set_index(['storm_id','reach_id'])





radii, historic_theta_bar,historic_theta_var, historic_x_freq, lon = jpm.get_storm_stats(dd)
polderObject = stm.polder(dd)
 #should be tested in model validation when compared to CLARA
#TODO make argument. will try
historic_c_p_a0, historic_c_p_a1 = jpm.get_jpmos_coefs(dd)
reach_objects = stm.construct_reach_objects(dd,file = "/reach_config_file_ipet_dummy_2m.csv")
#base_frequency = storm_param_desc.loc['index','column'] #fix
base_frequency = 0.22

unit_price = c_mdl.get_unit_price(dd)
reach_df = stm.get_reach_df(dd,file = "/reach_config_file_ipet_dummy_2m.csv").sort_values('reachID')
n_reach = len(reach_df)

#rainfall = rain.rainfall_prediction(storm_params, rain_params)
#rainfall = 0
base_crest_heights = reach_objects['reachHeight'].to_numpy() #something in the reach objects
print("parameters set")


partic_rate = 0
#sea_lvl = 1.03
rainfall = 0
intensity = 0
base_frequency = 0.22
pop_mult = 1
frequency_mod = 0
ns_cost_mult = 1
acq_cost_mult = 1
str_cost_mult = 1
NS_std = 1
acq_threshold = 1
res_fp = False
MCIterates = 25


# parameters for regression
n = 30
year_offset = 0
n_vars = 12
n_objs = 2
n_rbf = 3

# planning horizon
planning_horizon = 80
year_to_update = 60
# frequency of updates
time_step = 30
# num_steps must be equal to an integer
num_steps = int(year_to_update / time_step)
ped_lag = 3
cons_lag = 5

arguments = [reach_objects, dmg_data, nms_data, bfe_data, nsc_data, storm_params,
             locParams, surgeSurfaceParams, waveSurfaceParams, wavePeriodSurfaceParams,
             sigmaSurfaceParams, radii, historic_theta_bar, historic_theta_var,
             historic_x_freq, lon, polderObject, MCIterates, base_frequency,
             unit_price, rainfall, base_crest_heights, partic_rate, pop_mult,
             ns_cost_mult, acq_cost_mult, str_cost_mult, NS_std, acq_threshold,
             res_fp, rain_params, historic_c_p_a0, historic_c_p_a1, reach_df]


#sea_lvl_scenarios = [0.46, 0.63, 0.83]
#intensity_scenarios = [0.1, 0.125, 0.15]
#frequency_scenarios = [0, -0.14, -0.28]

###### modify into 80-year scale
#sea_lvl_scenarios = [0.46, 0.63, 0.83]
#intensity_scenarios = [0.16, 0.2, 0.24]
#frequency_scenarios = [0, -0.224, -0.448]

intensity_change = 0.2
frequency_change = -0.224

discount_rate = 0.03


def extract_peak_surge(sea_lvl, n, year_offset, intensity, frequency_mod, needed_arguments,\
                   planning_horizon, discount_rate, prev_h, year, n_reach):
   
    
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
                                                     sea_lvl[year+n])
            i = i + 1
        water_level = (surgeHeight.max(axis = 0)) * 0.3048
    
    else:
        water_level = np.full(shape = n_reach, fill_value = sea_lvl[year+n])
    
    return water_level


def over_time_surge(sea_level, n, year_offset, intensity_change, frequency_change,
                     planning_horizon, discount_rate, rs, needed_arguments):
    '''water level is previous n-year data: surge heights; needs to be adjusted every year'''

    np.random.seed(rs)
    water_levels = np.zeros((planning_horizon, n_reach))
    
    for year in range(planning_horizon):
        if year == 0:
            prev_h = base_crest_heights
#        sea_lvl = sea_level_rise[]
        intensity = (intensity_change / planning_horizon) * year
        frequency_mod = (frequency_change / planning_horizon) * year
        
        water_levels[year,:] = extract_peak_surge(sea_level, n, year_offset, intensity, frequency_mod, needed_arguments, \
                   planning_horizon, discount_rate, prev_h, year, n_reach)
        
    return water_levels.flatten()
    
    
water_levels = np.zeros((n_sow, planning_horizon*n_reach))
for i in range(n_sow):
    print(i)
    water_levels[i,:] = over_time_surge(sea_level_rise, n, year_offset, intensity_change, frequency_change, planning_horizon = 80, discount_rate = 0.03, rs = 19 + i * 10, needed_arguments = arguments)


np.savetxt("water_levels_medium_04-17-2024.csv", water_levels, delimiter=",",  fmt ='% s')

        
