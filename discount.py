# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 21:52:10 2022

@author: jwang
"""

####################################
## calculate discounted loss and investment
## over time
####################################
import math

def discounted(cost, dmg, n_years, discount_rate):
    
    # discounted investment
    total_cost = 0
    total_dmg = 0
    for i in range(n_years):
        total_cost += cost[i] * math.exp(-1 * discount_rate * i)
        total_dmg += dmg[i] * math.exp(-1 * discount_rate * i)
        
    return total_cost, total_dmg