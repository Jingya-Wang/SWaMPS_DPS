# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 22:32:37 2022

@author: jwang
"""

import numpy as np
import pandas as pd

#overall_frequency = 0.253731
#planning_horizon = 50
#storm_ids = storm_params.index[0:10]
#
## dummy data for probs for storms
#prob_by_storm = [1/10] * 10


def time_series_storms(overall_frequency, storm_ids, prob_by_storm):
    # number of storms occursing, assuming Poisson distribution
    num_storms = np.random.poisson(overall_frequency)

    # generate random numbers
    rand_num = np.random.uniform(0, 1, num_storms)
    
    # cumulative sum of probs of storms
    cum_probs_zero = pd.Series([0])
    cum_probs_following = np.cumsum(prob_by_storm['prob_by_storm'])
    cum_probs = cum_probs_zero._append(cum_probs_following, ignore_index = True)

    # find the corresponding storm_ids based their probabilities
    storms_occur = [None] * num_storms
    
    for i in range(num_storms):
        for j in range(1,len(storm_ids)+1):
            if rand_num[i] >= cum_probs[j-1] and rand_num[i] < cum_probs[j]:
                storms_occur[i] = storm_ids[j-1]

    return num_storms, storms_occur
        
        
