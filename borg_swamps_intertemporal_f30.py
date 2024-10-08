from borg import *

directory = '/work/disasters/jw' #find actual file path later
dd = directory + '/data'
dd_2 = dd + '/swamps_DPS/'

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
import larose_function_variations_swamps_DPS as lfv
import calc_heightening as ch
import discount as dc
import time_series_storms as tss

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


slr = pd.read_csv(dd_2 + '3_slr_sows_01_18.csv', header = None)
sea_level_rises = slr.to_numpy()
# comment this out because we need to see which SLR scenario is needed
#sea_level_rise = sea_level_rise[2,:]

#this could be slr or time series of storms
n_sow = 35 # 35 scenarios after scenario reduction

# weights of each scnarios
weights = pd.read_csv(dd_2 + 'weights_all-20240510.csv', header = None)
weights = weights.to_numpy()

#IDs of the full ensemble
scenarios_ids = pd.read_csv(dd_2 + 'closest_all_20240510.csv', header = None)
scenarios_ids = scenarios_ids.to_numpy()

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
sea_lvl = 1.03
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


#intensity_scenarios = [0.1, 0.125, 0.15]
#frequency_scenarios = [0, -0.14, -0.28]

###### modify into 80-year scale
intensity_scenarios = [0.16, 0.2, 0.24]
frequency_scenarios = [0, -0.224, -0.448]
# we need to uncomment above and comment below, because we need to know which scenario we need later
#intensity_change = 0.125
#frequency_change = -0.14

discount_rate = 0.03
#planning_horizon = 50



####################################
## problem formulation, for each SOW
####################################
def larose_over_time_intertemporal(sea_level, crest_upgrades, year_offset, intensity_change, frequency_change,
                     planning_horizon, discount_rate, rs, needed_arguments):
    
#    # This is to adjust for the integer values: 0-50 being fed in by rhodium
#    upgrade = copy.deepcopy(crest_height_upgrade)
#    upgrade[:] = [x / 10 for x in upgrade]
    #print(rs)
    np.random.seed(rs)
    years = [time_step * x for x in list(range(0,num_steps+1))]
    
    i = 0
    j = 0
    
    dmg = np.zeros(planning_horizon)
    cost = np.zeros(planning_horizon)

    #crest_height_upgrade, needed_arguments, planning_horizon, discount_rate
    for year in range(planning_horizon):
        if year == 0:
            prev_h = base_crest_heights

#        sea_lvl = sea_level_rise[]
        intensity = (intensity_change / planning_horizon) * year
        frequency_mod = (frequency_change / planning_horizon) * year
        
        ## ns_cost + ped_cost
        if year == years[i]:
            crest_height_upgrade_to_be = crest_upgrades[i,:]
            ns_cost, ped_cost, cons_mtg_cost, cons_cost, om_cost = mtc_dps.get_cost(nsc_data, NS_std, pop_mult, partic_rate, ns_cost_mult, acq_cost_mult, crest_height_upgrade_to_be, prev_h, year, unit_price, reach_df, str_cost_mult, planning_horizon, discount_rate)
            cost[year] = ns_cost + ped_cost/3
            crest_height_upgrade = np.zeros(n_reach)
            dmg[year], current_h = lfv.larose_future_intertemporal(sea_level[year+n], year_offset, intensity, frequency_mod, crest_height_upgrade, needed_arguments, 1, False,\
                   planning_horizon, discount_rate, prev_h, year)
            i = i + 1
            j = j + 1
        ## only ped_cost
        elif year < (years[j-1] + ped_lag):
            cost[year] = ped_cost/3
            crest_height_upgrade = np.zeros(n_reach)
            dmg[year], current_h = lfv.larose_future_intertemporal(sea_level[year+n], year_offset, intensity, frequency_mod, crest_height_upgrade, needed_arguments, 1, False,\
                   planning_horizon, discount_rate, prev_h, year)
        ## cons_mtg_cost + cons_cost
        elif year >= (years[j-1] + ped_lag) and year < (years[j-1] + ped_lag + cons_lag):
            cost[year] = (cons_mtg_cost + cons_cost)/cons_lag
            crest_height_upgrade = np.zeros(n_reach)
            dmg[year], current_h = lfv.larose_future_intertemporal(sea_level[year+n], year_offset, intensity, frequency_mod, crest_height_upgrade, needed_arguments, 1, False,\
                   planning_horizon, discount_rate, prev_h, year)
        # om cost
        elif year == (years[j-1] + ped_lag + cons_lag):
            cost[year] = om_cost
            crest_height_upgrade = crest_height_upgrade_to_be
            dmg[year], current_h = lfv.larose_future_intertemporal(sea_level[year+n], year_offset, intensity, frequency_mod, crest_height_upgrade, needed_arguments, 1, False,\
                   planning_horizon, discount_rate, prev_h, year)
        # om cost, no need to update the upgrades
        else:
            cost[year] = om_cost
            crest_height_upgrade = np.zeros(n_reach)
            dmg[year], current_h = lfv.larose_future_intertemporal(sea_level[year+n], year_offset, intensity, frequency_mod, crest_height_upgrade, needed_arguments, 1, False,\
                   planning_horizon, discount_rate, prev_h, year)
        prev_h = current_h
        if i == (num_steps + 1):
            i = num_steps
    #print(dmg[year], cost[year])

    cost_pv, dmg_pv = dc.discounted(cost, dmg, planning_horizon, discount_rate)
              
    
    
    return (dmg_pv, cost_pv)



####################################
## problem formulation, get ready to be optimized
####################################
def swamps_intertemporal(*crest_upgrades):
    ####################################
    ## wrap up
    ## generate all SOWs
    ####################################
    np.random.seed(22)
    total_cost = np.zeros(n_sow)
    total_dmg = np.zeros(n_sow)
    
    crest_upgrades = np.reshape(crest_upgrades, (num_steps+1, n_reach))

    objs = [0.0] * n_objs
    #constrs = [0.0] * nconstrs
    
    ###############
    ## if use kernel rbf, need to normalize
    ###############
    ##Normalize weights to sum to 1
    #total = sum(orig_w)
    #if total != 0.0:
    #    for i in range(len(orig_w)):
    #        w[i] = orig_w[i] / total
    #else:
    #    for i in range(len(w)):
    #        w[i] = 1 / n_rbf

    for i in range(n_sow):
        #rs = 9 + i * 10
        scenarios_id = scenarios_ids[i]
        if scenarios_id < 10000: # low scenario
            rs = 9 + scenarios_id * 10
            sea_level_rise = sea_level_rises[1,:]
            intensity_change = intensity_scenarios[0]
            frequency_change = frequency_scenarios[0]
        elif 10000 <= scenarios_id < 20000: # medium scenario
            rs = 19 + (scenarios_id - 10000) * 10
            sea_level_rise = sea_level_rises[2,:]
            intensity_change = intensity_scenarios[1]
            frequency_change = frequency_scenarios[1]
        else:
            rs = 29 + (scenarios_id - 20000) * 10 # high scenario
            sea_level_rise = sea_level_rises[3,:]
            intensity_change = intensity_scenarios[2]
            frequency_change = frequency_scenarios[2]
        total_dmg[i], total_cost[i] = larose_over_time_intertemporal(sea_level_rise, crest_upgrades, year_offset, intensity_change, frequency_change,
                     planning_horizon = 80, discount_rate = 0.03, rs = rs, needed_arguments = arguments)

    # assign objectives weights
    
    #sum_cost = sum(total_cost)
    #sum_dmg = sum(total_dmg)
    #weighted_cost = sum_cost/n_sow
    #weighted_dmg = sum_dmg/n_sow

    weighted_cost = np.dot(total_cost, weights)
    weighted_dmg = np.dot(total_dmg, weights)

    objs[0] = weighted_cost
    objs[1] = weighted_dmg
    

    return objs
    

import timeit
start= timeit.default_timer()

### run with borg
nvars = 12 * (num_steps + 1)
nobjs = 2
Configuration.startMPI()

borg = Borg(nvars, nobjs, 0, swamps_intertemporal)
borg.setBounds(*[[0,2]]*nvars)
borg.setEpsilons(1000,100)

num_func_evals = 10000
result = borg.solveMPI(maxEvaluations = num_func_evals)
Configuration.stopMPI()

#result = borg.solve({"maxEvaluations":10000})

objectives_total = np.empty(shape=[0,nobjs])
strategies_total = np.empty(shape=[0,nvars])

if result:
    for solution in result:
        objectives = solution.getObjectives()
        objectives = np.column_stack(objectives)
        objectives_total = np.append(objectives_total,objectives,axis=0)
        strategies = solution.getVariables()
        strategies = np.column_stack(strategies)
        strategies_total = np.append(strategies_total,strategies,axis=0)
#        i = i+1

#print(i)
if result:
        np.savetxt("objectives_intertemporal_f" + str(time_step) + "_rs12.csv", objectives_total, delimiter=",",  fmt ='% s')
        np.savetxt("strategies_intertemporal_f" + str(time_step) + "_rs12.csv", strategies_total, delimiter=",",  fmt ='% s')

print(objectives_total)
print(strategies_total)


stop = timeit.default_timer()

print('Time: ', stop - start)

















