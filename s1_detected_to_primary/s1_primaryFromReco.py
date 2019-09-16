
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
import sys

from pprint import pprint

from pax import units


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def getS1AreaCorrected():
    
    return s1_photons_test(10)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def getJSON():
    
    filename_json = '../pax_waveform_simulator/pax_output/Feb26/instructions_000000/pax_info.json'
    file_json     =  open(filename_json)
    data_json     = json.load(file_json)
    data_json     = data_json['configuration']
    
    #
    config_wfs = data_json['WaveformSimulator']

    #pprint(data_json, depth=2)
    
    return data_json


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def s1_photons_test(n_photons, x=0.0, y=0.0, z=0, t=0.0):
    
    config_wfs = getJSON()['WaveformSimulator']

    
    #--------------------------------------------------------------------------
    # Recombination time from NEST 2014
    # 3.5 seems fishy, they fit an exponential to data,
    # but in the code they use a non-exponential distribution...
    #--------------------------------------------------------------------------

    drift_field              = config_wfs['drift_field']
    efield                   = (drift_field / (units.V / units.cm))
    s1_ER_recombination_time = 3.5 / 0.18 * (1 / 20 + 0.41) * math.exp(-0.009 * efield)

    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    s1_ER_primary_singlet_fraction   = config_wfs['s1_ER_primary_singlet_fraction']
    s1_ER_secondary_singlet_fraction = config_wfs['s1_ER_secondary_singlet_fraction']
    s1_ER_recombination_fraction     = config_wfs['s1_ER_recombination_fraction']
    singlet_lifetime_liquid          = config_wfs['singlet_lifetime_liquid']
    triplet_lifetime_liquid          = config_wfs['triplet_lifetime_liquid']
    maximum_recombination_time       = config_wfs['maximum_recombination_time']
    
    
    #--------------------------------------------------------------------------
    # Primary excimer fraction from Nest Version 098
    # See G4S1Light.cc line 298
    #--------------------------------------------------------------------------

    density = config_wfs['liquid_density'] / (units.g / units.cm ** 3)
    excfrac = 0.4 - 0.11131 * density - 0.0026651 * density ** 2    # primary / secondary excimers
    excfrac = 1 / (1 + excfrac)                                     # primary / all excimers
    excfrac /= 1 - (1 - excfrac) * (1 - s1_ER_recombination_fraction)
        
    s1_ER_primary_excimer_fraction = excfrac
    
    
    
    #--------------------------------------------------------------------------
    # How many of these are primary excimers? Others arise through recombination.
    #--------------------------------------------------------------------------
    
    n_primaries = np.random.binomial(n=n_photons, p=s1_ER_primary_excimer_fraction)
    
    primary_timings = singlet_triplet_delays(
        np.zeros(n_primaries),  # No recombination delay for primary excimers
        t1=singlet_lifetime_liquid,
        t3=triplet_lifetime_liquid,
        singlet_ratio=s1_ER_primary_singlet_fraction
    )

    
    #--------------------------------------------------------------------------
    # Correct for the recombination time
    # For the non-exponential distribution: see Kubota 1979, solve eqn 2 for n/n0.
    # Alternatively, see Nest V098 source code G4S1Light.cc line 948
    #--------------------------------------------------------------------------
    
    secondary_timings = s1_ER_recombination_time * (-1 + 1 / np.random.uniform(0, 1, n_photons - n_primaries))
    secondary_timings = np.clip(secondary_timings, 0, maximum_recombination_time)
    
    
    #--------------------------------------------------------------------------
    # Handle singlet/ triplet decays as before
    #--------------------------------------------------------------------------
    
    secondary_timings += singlet_triplet_delays(
        secondary_timings,
        t1=singlet_lifetime_liquid,
        t3=triplet_lifetime_liquid,
        singlet_ratio=s1_ER_secondary_singlet_fraction
    )

        
    timings = np.concatenate((primary_timings, secondary_timings))

    return timings + t * np.ones(len(timings))
    
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    arr_return = np.zeros(1)
    
    return arr_return


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def singlet_triplet_delays(times, t1, t3, singlet_ratio):
    """
    Given a list of eximer formation times, returns excimer decay times.
        t1            - singlet state lifetime
        t3            - triplet state lifetime
        singlet_ratio - fraction of excimers that become singlets
                        (NOT the ratio of singlets/triplets!)
    """
    n_singlets = np.random.binomial(n=len(times), p=singlet_ratio)
    return times + np.concatenate([
        np.random.exponential(t1, n_singlets),
        np.random.exponential(t3, len(times) - n_singlets)
    ])
    
    
#--------------------------------------------------------------------------
# Need to calculate primary number of S1 photons from energy, e.g., reverse this process:
#
#   https://github.com/XENON1T/pax/blob/master/pax/simulation.py#L561
#
# def s1_photons(self, n_photons, recoil_type, x=0., y=0., z=0, t=0.):
#        """Returns a list of photon detection times at the PMT caused by an S1 emitting n_photons.
#        """
#        # Apply light yield / detection efficiency
#        log.debug("Creating an s1 from %s photons..." % n_photons)
#        ly = self.s1_light_yield_map.get_value(x, y, z) * self.config['s1_detection_efficiency']
#        n_photons = np.random.binomial(n=n_photons, p=ly)
#        log.debug("    %s photons are detected." % n_photons)
#        if n_photons == 0:
#            return np.array([])
#
#        if recoil_type.lower() == 'alpha':
#            # again neglible recombination time, same singlet/triplet ratio for primary & secondary excimers
#            # Hence, we don't care about primary & secondary excimers at all:
#            timings = self.singlet_triplet_delays(
#                np.zeros(n_photons),
#                t1=self.config['singlet_lifetime_liquid'],
#                t3=self.config['triplet_lifetime_liquid'],
#                singlet_ratio=self.config['s1_ER_alpha_singlet_fraction']
#            )
#
#        elif recoil_type.lower() == 'led':
#            # distribute photons uniformly within the LED pulse length
#            timings = np.random.uniform(0, self.config['led_pulse_length'],
#                                        size=n_photons)
#
#        elif self.config.get('s1_model_type') == 'simple':
#            # Simple S1 model enabled: use it for ER and NR.
#            timings = np.random.exponential(self.config['s1_decay_time'], size=n_photons)
#
#        elif recoil_type.lower() == 'er':
#            # How many of these are primary excimers? Others arise through recombination.
#            n_primaries = np.random.binomial(n=n_photons, p=self.config['s1_ER_primary_excimer_fraction'])
#
#            primary_timings = self.singlet_triplet_delays(
#                np.zeros(n_primaries),  # No recombination delay for primary excimers
#                t1=self.config['singlet_lifetime_liquid'],
#                t3=self.config['triplet_lifetime_liquid'],
#                singlet_ratio=self.config['s1_ER_primary_singlet_fraction']
#            )
#
#            # Correct for the recombination time
#            # For the non-exponential distribution: see Kubota 1979, solve eqn 2 for n/n0.
#            # Alternatively, see Nest V098 source code G4S1Light.cc line 948
#            secondary_timings = self.config['s1_ER_recombination_time']\
#                * (-1 + 1 / np.random.uniform(0, 1, n_photons - n_primaries))
#            secondary_timings = np.clip(secondary_timings, 0, self.config['maximum_recombination_time'])
#            # Handle singlet/ triplet decays as before
#            secondary_timings += self.singlet_triplet_delays(
#                secondary_timings,
#                t1=self.config['singlet_lifetime_liquid'],
#                t3=self.config['triplet_lifetime_liquid'],
#                singlet_ratio=self.config['s1_ER_secondary_singlet_fraction']
#            )
#
#            timings = np.concatenate((primary_timings, secondary_timings))
#
#        elif recoil_type.lower() == 'nr':
#            # Neglible recombination time, same singlet/triplet ratio for primary & secondary excimers
#            # Hence, we don't care about primary & secondary excimers at all:
#            timings = self.singlet_triplet_delays(
#                np.zeros(n_photons),
#                t1=self.config['singlet_lifetime_liquid'],
#                t3=self.config['triplet_lifetime_liquid'],
#                singlet_ratio=self.config['s1_NR_singlet_fraction']
#            )
#
#        else:
#            raise ValueError('Recoil type must be ER, NR, alpha or LED, not %s' % type)
#
#        return timings + t * np.ones(len(timings))
#--------------------------------------------------------------------------