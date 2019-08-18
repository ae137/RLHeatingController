""" temperature_simulator.py: Utility functions for training a heating controller

Copyright (C) 2017 Andreas Eberlein

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
"""

import math
import random
from collections import namedtuple

# random.seed(0)

# Named tuple for storing data required for initializing the simulation of the evolution of
# the inside temperature
TempInsideInitProps = namedtuple('TempInsideInitProps', ['mean', 'spread'])

# Named tuple for storing data required for initializing the simulation of the evolution of
# the outside temperature
TempOutsideInitProps = namedtuple('TempOutsideInitProps', ['mean_min', 'spread_min',
                                                           'mean_max', 'spread_max'])


def temp_inside_init(Tin_init) -> float:
    """ Get initial value for inside temperature

    Parameters
    ----------
    Tin_init: TempInsideInitProps
        Named tuple with data for initializing inside temperature

    Returns
    -------
    float
        Initial value for inside temperature
    """
    return Tin_init.mean + random.randint(-Tin_init.spread, Tin_init.spread)


class EnvironmentSimulator(object):
    """ Simple class for simulating the evolution of the outside temperature over time """

    def __init__(self, init_props):
        """ Initialize simulator for outside temperature

        Parameters
        ----------
        init_props: TempOutsideInitProps
            Named tuple with data for initializing outside temperature
        """
        self.init_props = init_props
        self.reset()

    def getOutTemp(self, time_step) -> float:
        """ Return current outside temperature

        Parameters
        ----------
        time_step: float
            Time step of simulation (in multiples of quarter hours)

        Returns
        -------
        float
            Outside temperature at time time_step
        """
        return self.To_min + (self.To_max - self.To_min) \
            * (math.sin(math.pi * time_step / 96))**2

    def reset(self):
        """ Reinitialize simulator of outside temperature with new min and max values """
        self.To_min = self.init_props.mean_min + random.randint(-self.init_props.spread_min,
                                                                self.init_props.spread_min)
        self.To_max = self.init_props.mean_max + random.randint(-self.init_props.spread_max,
                                                                self.init_props.spread_max)


class TargetTemperature(object):
    """ Simple class for simulating the evolution of the target inside temperature over time """

    def __init__(self, T_target):
        """ Initialize time simulator for time dependence of target inside temperature

        Parameters
        ----------
        T_target: float
            Target inside temperature
        """
        self.T_target = T_target    # Can later be replaced by more complicated target

    def getTargetTemp(self, time_step) -> float:
        """ Return current target inside temperature

        Parameters
        ----------
        time_step: float
            Time step of simulation (in multiples of quarter hours)

        Returns
        -------
        float
            Target inside temperature at time time_step
        """
        return self.T_target


def main_func():
    """ Function for visualizing the simulated temperature dependence of the environment. """
    import matplotlib.pyplot as plt

    To_min = 5      # Mean of minimum outside temperature in °C
    To_max = 15     # Mean of maximum outside temperature in °C

    Tout_init_props = TempOutsideInitProps(mean_min=To_min, spread_min=3,
                                           mean_max=To_max, spread_max=3)
    EnvTempSim = EnvironmentSimulator(Tout_init_props)

    time_hours = []
    temp_data = []
    for time_step in range(96 * 4):
        if (time_step % 96 == 0):
            EnvTempSim.reset()
        time_hours.append(time_step / 4)
        temp_data.append(EnvTempSim.getOutTemp(time_step))

    plt.xlabel('Time (h)')
    plt.ylabel('Outside temperature (°C)')
    plt.plot(time_hours, temp_data)
    plt.show()


if __name__ == "__main__":
    main_func()
