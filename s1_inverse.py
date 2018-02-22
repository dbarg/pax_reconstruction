

####################################################################################################
####################################################################################################

import sys
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


####################################################################################################
####################################################################################################

def getS1AreaCorrected():
    
    return getS1AreaCorrectedER()


####################################################################################################
####################################################################################################

def getS1AreaCorrectedER():
    
    ################################################################################################
    # How many of these are primary excimers? Others arise through recombination.
    ################################################################################################
    
    #n_primaries = np.random.binomial(n=n_photons, p=self.config['s1_ER_primary_excimer_fraction'])
    #
    #primary_timings = self.singlet_triplet_delays(
    #    np.zeros(n_primaries),  # No recombination delay for primary excimers
    #    t1=self.config['singlet_lifetime_liquid'],
    #    t3=self.config['triplet_lifetime_liquid'],
    #    singlet_ratio=self.config['s1_ER_primary_singlet_fraction']
    #)

    
    ################################################################################################
    # Correct for the recombination time
    # For the non-exponential distribution: see Kubota 1979, solve eqn 2 for n/n0.
    # Alternatively, see Nest V098 source code G4S1Light.cc line 948
    ################################################################################################
    
    #secondary_timings = self.config['s1_ER_recombination_time']\
    #    * (-1 + 1 / np.random.uniform(0, 1, n_photons - n_primaries))
    #secondary_timings = np.clip(secondary_timings, 0, self.config['maximum_recombination_time'])
    
    
    ################################################################################################
    # Handle singlet/ triplet decays as before
    ################################################################################################
    
    #secondary_timings += self.singlet_triplet_delays(
    #    secondary_timings,
    #    t1=self.config['singlet_lifetime_liquid'],
    #    t3=self.config['triplet_lifetime_liquid'],
    #    singlet_ratio=self.config['s1_ER_secondary_singlet_fraction']
    #)

        
    timings = np.concatenate((primary_timings, secondary_timings))

    
    
    ################################################################################################
    ################################################################################################
    
    arr_return = np.zeros(1)
    
    return arr_return


####################################################################################################
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
####################################################################################################
    

