"""
This module contains functions that handle conversions from cumulative to
incremental runoff values. It is based on code from the 
CreateInflowFileFromGriddedRunoff class from the RAPIDpy codebase written
by Alan Snow.
"""

import numpy as np

def apply_era_interim_t255_runoff_rule(runoff_data):
    # A) ERA Interim Low Res (T255) - data is cumulative
    # from time 3/6/9/12
    # (time zero not included, so assumed to be zero)
    ro_first_half = (
        np.concatenate([runoff_data[0:1,:], np.subtract(runoff_data[1:4,:],
                                                        runoff_data[0:3,:])]))

    # from time 15/18/21/24
    # (time restarts at time 12, assumed to be zero)
    ro_second_half = (
        np.concatenate([runoff_data[4:5,:], np.subtract(runoff_data[5:,:],
                                                runoff_data[4:7,:])]))

    ro = np.concatenate([ro_first_half, ro_second_half])

    return ro

def apply_era_interim_t1279_runoff_rule(runoff_data):
    # A) ERA Interim Low Res (T1279) - data is cumulative
    # from time 6/12
    # 0 1 2 3 4
    # (time zero not included, so assumed to be zero)
    ro_first_half = (
        np.concatenate([runoff_data[0:1, ], np.subtract(runoff_data[1:2,:],
                                                        runoff_data[0:1,:])]))

    # from time 15/18/21/24
    # (time restarts at time 12, assumed to be zero)
    ro_second_half = (
        np.concatenate([runoff_data[2:3, ], np.subtract(runoff_data[3:,:],
                                                        runoff_data[2:3,:])]))

    ro = np.concatenate([ro_first_half, ro_second_half])
        
    return ro

def cumulative_to_incremental(runoff_data):
    """
    Subtract `runoff_data[i-1]` from `runoff_data[i]` for i = 1,2,...,n to 
    obtain incremental values from a cumulative dataset. Assume that the runoff
    value prior to the first value in `runoff_data` is zero.
    """
    ro = np.append(runoff_data[0], np.diff(runoff_data))

    return ro
