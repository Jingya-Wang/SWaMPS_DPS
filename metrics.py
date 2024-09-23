# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 14:08:52 2019

@author: richardz
"""

import pandas as pd
import numpy as np
import math
import cost_model as c_mdl

#Set baseline parameters as needed

def get_analytica_data(dd):
    
    ##########################################################################################
    #reads in data as dataframes from json files
    ##########################################################################################

    dmg_data = pd.read_json(dd + '/drp_data_2019_06_08.json', 'df')
    nms_data = pd.read_json(dd + '/nms_data_2019_06_08.json', 'df')
    bfe_data = pd.read_json(dd + '/bfe_data_2019_06_08.json', 'df')
    nsc_data = pd.read_json(dd + '/nsc_data_2019_06_08.json', 'df')
    
    return dmg_data, nms_data, bfe_data, nsc_data

def get_cost(nsc_data, NS_std, pop_mult, partic, ns_cost_mult, acq_cost_mult, crest_heights, unit_price, reach_objects, str_cost_mult):
    
    ##########################################################################################
    #calls functions for ns costs and structural costs, summing them
    ##########################################################################################
    
    cost = ns_cost_calc(nsc_data, NS_std, pop_mult, partic, ns_cost_mult, acq_cost_mult) + \
                        c_mdl.calc_str_cost(crest_heights, reach_objects, unit_price, str_cost_mult)
    
    return cost

def ns_cost_calc(nsc_data, ns_std, pop_mult, partic, ns_cost_mult, acq_cost_mult):
    
    ##########################################################################################
    #calculates the nonstructural costs, including the multipliers
    ##########################################################################################
    
    #creates a single row dataframe with the specific metrics being examined
    row_selection = pd.DataFrame(np.nan, index = range(0,1), columns = ['partic','ns_std','pop_mult'])
    row_selection.loc[0,'partic'] = round(partic,1)
    row_selection.loc[0,'ns_std'] = round(ns_std,1)
    row_selection.loc[0,'pop_mult'] = round(pop_mult,1)
    
    #retrieves cost data by merging single row with nsc dataframe
    spec_cost = pd.merge(nsc_data,row_selection, on = ['partic','ns_std','pop_mult'])
    
    #extracts cost types and multiplies by coefficients
    rf_val = ns_cost_mult * spec_cost.loc[0,'res_fp']
    elev_val = ns_cost_mult * spec_cost.loc[0,'elev']
    acq_val = acq_cost_mult * spec_cost.loc[0,'acq']
    
    #sums cost types into one overall cost
    ns_cost = rf_val + elev_val + acq_val
    
    return ns_cost

def get_nms(nms_data,ns_std,partic):
    
    ##########################################################################################
    #retrieves the number of mitigated structures
    ##########################################################################################
    
    #creates a single row dataframe with the specific metrics being examined
    row_selection = pd.DataFrame(np.nan, index = range(0,1), columns = ['partic','ns_std'])
    row_selection.loc[0,'partic'] = round(partic,1)
    row_selection.loc[0,'ns_std'] = round(ns_std,1)
    
    #retrieves nms data by merging single row with nms dataframe
    nms_row = pd.merge(nms_data,row_selection, on = ['partic','ns_std'])
    
    #isolates number of mitigated structures
    nms = nms_row.loc[0,'nms']
    
    return nms

def get_bfe(bfe_data,ns_std,partic):    
    
    ##########################################################################################
    #retrieves the number of structures above the bfe in the same manner as the nms
    ##########################################################################################
    
    row_selection = pd.DataFrame(np.nan, index = range(0,1), columns = ['partic','ns_std'])
    row_selection.loc[0,'partic'] = round(partic,1)
    row_selection.loc[0,'ns_std'] = round(ns_std,1)
    
    bfe_row = pd.merge(bfe_data,row_selection, on = ['partic','ns_std'])
    
    bfe = bfe_row.loc[0,'str_above_bfe']
    
    return bfe

def get_swe_100(SWE_cdf):
    
    #isolates the SWE for the 100 year return period from the dataframe coming from the jpmos section
    swe_100 = SWE_cdf.loc[list(SWE_cdf['return period (year)']).index(100),"SWE"] 
    
    return swe_100

def get_ead(dmg_data,PR,PM,SWE_cdf,NS,storm_freq):
    
    #executes the ead_calc function
    ead = ead_calc(dmg_calc_cdf(dmg_data,PR,PM,SWE_cdf,NS),storm_freq)
    
    return ead

def get_drp_100(dmg_data,PR,PM,SWE_cdf,NS):
    
    #executes the function to isolate the 100 year return period damage
    drp_100 = drp_100_calc(dmg_calc_cdf(dmg_data,PR,PM,SWE_cdf,NS))
    
    return drp_100

def dmg_calc_cdf(dmg_data,PR,PM,SWE_cdf_df,NS):
    
    ##########################################################################################
    #creates a list of damages corresponding to each of the 22 return periods
    ##########################################################################################
    
    #converts the swe column of the dataframe into a list
    SWE_cdf = list(SWE_cdf_df['SWE'])
    
    #rounds the given metrics to the nearest tenth (maybe ought to get interpolated later? 
    #I think it was brought up a while ago, but don't remember exactly
    PR_r =  round(PR,1)
    PM_r =  round(PM,1)
    
    #ns standard gets interpolated in later functions
    NS_r =  NS
    
    #tfdi is the list of damages by rp
    total_flood_dmg_int = [(get_flood_dmg_for_swe(dmg_data,PR_r,PM_r,SWE_cdf[x],NS_r)) for x in range(0,22)] 
    
    #print('tfdi: ')
    print(total_flood_dmg_int)
    
    return total_flood_dmg_int

def get_flood_dmg_for_swe(dmg_data,partic,pop_mult,SWE,ns_std):
    
    ##########################################################################################
    #function to pull stuff out of data file and into list sorted by return period, with dmg summed across all asset classes
    ##########################################################################################
    
    #creates list of swes in increments of .1
    SWE_index = [round((i/10),1) for i in range(-130,150,1)]
    
    #print(SWE_index)
    
    #gets damage, interpolating between two adjacent SWE increments
    rp_flood_dmg = dmg_calc_interpolation(dmg_data,partic,pop_mult,SWE,ns_std,SWE_index)

    return rp_flood_dmg


def dmg_calc_interpolation(dmg_data,partic,pop_mult,SWE,ns_std,SWE_index): 
    
    ##########################################################################################
    #gets damage, interpolating between two adjacent SWE increments
    ##########################################################################################
    
    #gets increments of .1 on either side of the real SWE in question
    lower_bound = math.floor(SWE*10)/10
    upper_bound = lower_bound + 0.1
    

    
    #extract damage value, allowing for odd rounding behavior in dmg_data.swe
    lower_dmg_df = dmg_data[(dmg_data.partic == round(partic,1)) & \
                                (dmg_data.ns_std == round(ns_std,1)) & \
                                (dmg_data.pop_mult == round(pop_mult,1)) & \
                                (dmg_data.swe >lower_bound - 0.01) &\
                                (dmg_data.swe < lower_bound + 0.01)]
    

    lower_dmg = lower_dmg_df.iloc[0]['drp']
    

    #extract damage value, allowing for odd rounding behavior in dmg_data.swe
    upper_dmg_df = dmg_data[(dmg_data.partic == round(partic,1)) & \
                                (dmg_data.ns_std == round(ns_std,1)) & \
                                (dmg_data.pop_mult == round(pop_mult,1)) & \
                                (dmg_data.swe >upper_bound - 0.01) &\
                                (dmg_data.swe < upper_bound + 0.01)]
    
    upper_dmg = upper_dmg_df.iloc[0]['drp']
    
    #calls the function to interpolate the two damage values
    interpolated_dmg = linear_interpolate(SWE,lower_bound,upper_bound,lower_dmg,upper_dmg)
    
    return interpolated_dmg


def linear_interpolate(value,lower_bound,upper_bound,lower_val,upper_val):

    #linear interpolation
    result = lower_val + ((value - lower_bound)*((upper_val - lower_val)/(upper_bound - lower_bound)))
    
    return result


def drp_100_calc(total_flood_dmg_int):
    
    #reads in the list of damages for all return periods, isolates the 100 year dmg, and returns it
    
    return_periods = [5,8,10,13,15,20,25,33,42,50,75,100,125,150,200,250,300,350,400,500,1000,2000]
    drp_100 = total_flood_dmg_int[return_periods.index(100)]
    
    return drp_100


def ead_calc(total_flood_dmg_int, storm_freq):
    
    #reads in the list of damages from all return periods, and mimics the analytica model to calculate EAD
    #each line/loop block, starting with f_annual, should correspond to a function block in Analytica
    
    #eliminating the abundance of for loops here would probably go a long way toward speeding things up

    return_periods = [5,8,10,13,15,20,25,33,42,50,75,100,125,150,200,250,300,350,400,500,1000,2000]
    interval_prob = [None] * len(return_periods)
    ead = 0
    num_storms = [0,1,2,3,4,5,6,7,8,9,10]
    rp_indices = [x for x in range(0,22)]
    p_severe_storm = [None] * len(return_periods)
    p_severe_storm[:] = [([None] * len(num_storms)) for x in range(0,len(p_severe_storm))]
    interval_storm_weight = [None] * len(return_periods)

    f_annual = [(1.0-(1.0/return_periods[x])) for x in rp_indices]

    f_event = [(f_annual[x] ** (1/storm_freq)) for x in rp_indices]

    prob_storms = [(math.exp(-storm_freq) * storm_freq ** num_storms[x] / math.factorial(num_storms[x])) for x in num_storms]
    
    for i in range(0,len(f_event)):
        if i == 0:
            interval_prob[i] = 1 - ((f_event[i] + f_event[i+1]) / 2)
        elif i == (len(f_event)-1):
            interval_prob[i] = 1 - ((f_event[i-1] + f_event[i]) / 2)
        else:
            interval_prob[i] = 1 - ((f_event[i-1] + f_event[i+1]) / 2) 
    next

    f_interval_prob = np.cumsum(interval_prob)

    for i in range((len(f_interval_prob)-1),-1, -1): 
        for j in range(0, len(num_storms)):
            if num_storms[j] == 0:
                p_severe_storm[i][j] = 0
            else:
                p_severe_storm[i][j] = f_interval_prob[i] ** num_storms[j]
            
            if i > 0 and i < 21:
                p_severe_storm[i+1][j] = p_severe_storm[i+1][j] - p_severe_storm[i][j]
        next
    next
    
    for i in range(0,len(rp_indices)):
        for j in range(0,len(prob_storms)):
            if j == 0:

                interval_storm_weight[i] = p_severe_storm[i][j] * prob_storms[j]
            else:
                interval_storm_weight[i] += p_severe_storm[i][j] * prob_storms[j]
            
        next 
    next
    
    for i in range(0,len(return_periods)):
        ead += total_flood_dmg_int[i] * interval_storm_weight[i]
    next

    return ead

#########################################################################################################################

##########################################################################################
    #dummy stuff included for testing to make script run independently from any others
##########################################################################################
"""
dmg_data, nms_data, bfe_data, nsc_data = get_analytica_data('D:/Projects/Larose/Damage_Model')

return_periods = [5,8,10,13,15,20,25,33,42,50,75,100,125,150,200,250,300,350,400,500,1000,2000]

dummy_SWE_cdf = pd.DataFrame(np.nan, index = range(0,22), columns = ['return period (year)', 'SWE'])
dummy_SWE_cdf['SWE'] = [i for i in range(-10,12,1)]
dummy_SWE_cdf['return period (year)'] = return_periods
print(dummy_SWE_cdf)

storm_freq = 0.22
partic = 1
ns_std = 13
pop_mult = 1.2
ns_cost_mult = 1.1
acq_cost_mult = 1.2

bfe = get_bfe(bfe_data, ns_std, partic)
nms = get_nms(nms_data, ns_std, partic)
nsc = ns_cost_calc(nsc_data, ns_std, pop_mult, partic, ns_cost_mult, acq_cost_mult)
swe_100 = get_swe_100(dummy_SWE_cdf)

tfdi = dmg_calc_cdf(dmg_data, partic, pop_mult, dummy_SWE_cdf, ns_std)
drp_100 = drp_100_calc(tfdi)
ead = ead_calc(tfdi, storm_freq)

print('BFE: ' + str(bfe))
print('NMS: ' + str(nms))
print('NSC: ' + str(nsc))
print('SWE_100: ' + str(swe_100))
print('DRP_100: ' + str(drp_100))
print('EAD:     ' + str(ead))
"""