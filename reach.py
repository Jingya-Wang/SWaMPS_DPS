# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 21:29:36 2019

@author: ngeldner
"""

import numpy as np
import math
import pandas as pd





        


def reach(frame):
    fragCurveWall = {'charLength': float(frame.charLengthWall),\
                     'pMax': float(frame.pMaxWall),\
                     'k': float(frame.kWall),\
                     'x_c': float(frame.x_cWall)}
    
    fragCurveNoWall = {'charLength': float(frame.charLengthNoWall),\
                       'pMax': float(frame.pMaxNoWall),\
                       'k': float(frame.kNoWall),\
                       'x_c': float(frame.x_cNoWall)}
            
    my_reach = {'reachID': frame.reachID, 'reachName':frame.reachName, \
                'reachLength': float(frame.totalReachLengthInMeters),\
                'reachLengthWall': float(frame.tWallLength)*0.3048,\
                'crownWidth': float(frame.crownWidth)*0.3048,\
                'reachHeight':float(frame.existingTopOfLevee)*0.3048,\
                'groundHeight': float(frame.existingGrade)*0.3048,\
                'rearSlope':float(frame.rearSlope),\
                'forwardSlope':float(frame.forwardSlope),\
                'fragCurveWall': fragCurveWall,\
                'fragCurveNoWall': fragCurveNoWall}
    
    my_reach['reachLengthNoWall'] = my_reach['reachLength'] - my_reach['reachLengthWall']
    return my_reach
    
    