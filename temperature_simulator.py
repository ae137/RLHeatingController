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
# the outside temperature
TempInitProps = namedtuple('TempOutsideInitProps', ['mean_min', 'spread_min',
                                                    'mean_max', 'spread_max'])


class TemperatureSimulatorBase(object):
    """ Simple base class for temperature simulators """

    def __init__(self, init_props):
        """ Initialize simulator

        Parameters
        ----------
        init_props: TempInitProps
            Named tuple with data for initializing temperature simulator
        """
        self.init_props = init_props
        self.reset()

    def reset(self):
        """ Reinitialize temperature simulator with new min and max values """
        self.T_min = self.init_props.mean_min + random.uniform(-self.init_props.spread_min,
                                                               self.init_props.spread_min)
        self.T_max = self.init_props.mean_max + random.uniform(-self.init_props.spread_max,
                                                               self.init_props.spread_max)


class EnvironmentTemperatureSimulator(TemperatureSimulatorBase):
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
        return self.T_min + (self.T_max - self.T_min) \
            * (math.sin(math.pi * time_step / 96))**2


class TargetTemperatureSimulator(TemperatureSimulatorBase):
    """ Simple class for simulating the evolution of the target inside temperature over time """
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
        # Time dependence of temperature without variation:
        # return self.T_max
        # Time dependence of target temperature corresponding to a weekend day:
        return self.T_max if (time_step % 96 > 27 and time_step % 96 < 87) else self.T_min
        # Time dependence of target temperature corresponding to a week day:
        # return self.T_max if ((time_step % 96 > 23 and time_step % 96 < 33) or
        #                       (time_step % 96 > 68 and time_step % 96 < 88)) else self.T_min


def main_func():
    """ Function for visualizing the simulated temperature dependence of the environment. """
    import matplotlib.pyplot as plt

    To_min = 5      # Mean of minimum outside temperature in °C
    To_max = 15     # Mean of maximum outside temperature in °C
    Tout_init_props = TempInitProps(mean_min=To_min, spread_min=3, mean_max=To_max, spread_max=3)
    EnvTempSim = EnvironmentTemperatureSimulator(Tout_init_props)

    Ti_min = 16     # Mean of minimum inside target temperature in °C
    Ti_max = 21     # Mean of maximum inside target temperature in °C
    Tin_init_props = TempInitProps(mean_min=Ti_min, spread_min=2, mean_max=Ti_max, spread_max=2)
    TargetTempSim = TargetTemperatureSimulator(Tin_init_props)

    time_hours = []
    temp_data = []
    target_temp_data = []
    for time_step in range(96 * 4):
        if (time_step % 96 == 0):
            EnvTempSim.reset()
            TargetTempSim.reset()
        time_hours.append(time_step / 4)
        temp_data.append(EnvTempSim.getOutTemp(time_step))
        target_temp_data.append(TargetTempSim.getTargetTemp(time_step))

    plt.xlabel('Time (h)')
    plt.ylabel('Outside temperature (°C) / Target temperature (°C)')
    plt.plot(time_hours, temp_data)
    plt.plot(time_hours, target_temp_data)
    plt.show()


if __name__ == "__main__":
    main_func()
