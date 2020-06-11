import math
import random
from collections import namedtuple

# random.seed(0)

# Named tuple for storing data required for initializing the simulation of the evolution of
# the outside temperature
TempInitProps = namedtuple('TempInitProps', ['mean_min', 'spread_min',
                                             'mean_max', 'spread_max'])
SwitchingTimes = namedtuple('SwitchingTimes', ['on_early', 'off_early',
                                               'on_late', 'off_late',
                                               'spread'])
DayLength = 96      # Length of a day in multiples of 1/4 h


def addNoiseToFloat(x: float, s: float) -> float:
    return x + random.uniform(-s, s)


def addNoiseToInt(x: int, s: int) -> int:
    return x + random.randint(-s, s)


class TemperatureSimulatorBase(object):
    """ Simple base class for temperature simulators """

    def __init__(self, init_props: TempInitProps):
        """ Initialize simulator

        Parameters
        ----------
        init_props: TempInitProps
            Named tuple with data for initializing temperature simulator
        """
        self.init_props = init_props
        # Default switching times in 1/4 h (one day = 96)
        self.t_switch_base = SwitchingTimes(on_early=23, off_early=33,
                                            on_late=68, off_late=88, spread=1)
        self.t_switch = self.t_switch_base
        self.profile_types = ["Weekday", "Holiday"]
        self.reset()

    def reset(self):
        """ Reinitialize temperature simulator with new min and max values """
        self.T_min = addNoiseToFloat(self.init_props.mean_min, self.init_props.spread_min)
        self.T_max = addNoiseToFloat(self.init_props.mean_max, self.init_props.spread_max)
        self.t_switch = SwitchingTimes(on_early=addNoiseToInt(self.t_switch_base.on_early,
                                                              self.t_switch_base.spread),
                                       off_early=addNoiseToInt(self.t_switch_base.off_early,
                                                               self.t_switch_base.spread),
                                       on_late=addNoiseToInt(self.t_switch_base.on_late,
                                                             2*self.t_switch_base.spread),
                                       off_late=addNoiseToInt(self.t_switch_base.off_late,
                                                              2*self.t_switch_base.spread),
                                       spread=self.t_switch_base.spread)
        self.day_type = "Weekday" if random.random() < 0.72 else "Holiday"


class EnvironmentTemperatureSimulator(TemperatureSimulatorBase):
    def getOutTemp(self, time_step: int) -> float:
        """ Return current outside temperature

        Parameters
        ----------
        time_step: int
            Time step of simulation (in multiples of quarter hours)

        Returns
        -------
        float
            Outside temperature at time time_step
        """
        return self.T_min + (self.T_max - self.T_min) \
            * (math.sin(math.pi * time_step / DayLength))**2


class TargetTemperatureSimulator(TemperatureSimulatorBase):
    """ Simple class for simulating the evolution of the target inside temperature over time """
    def getTargetTemp(self, time_step: int) -> float:
        """ Return current target inside temperature

        Parameters
        ----------
        time_step: int
            Time step of simulation (in multiples of quarter hours)

        Returns
        -------
        float
            Target inside temperature at time time_step
        """
        assert self.day_type in self.profile_types, "Encountered unknown temperature profile"
        if self.day_type == "Weekday":
            return self.T_max if ((time_step % DayLength > self.t_switch.on_early
                                   and time_step % DayLength < self.t_switch.off_early) or
                                  (time_step % DayLength > self.t_switch.on_late
                                   and time_step % DayLength < self.t_switch.off_late)) \
                                       else self.T_min
        else:
            return self.T_max if (time_step % DayLength > self.t_switch.on_early
                                  and time_step % DayLength < self.t_switch.off_late) \
                                      else self.T_min


def main_func():
    """ Function for visualizing the simulated temperature dependence of the environment. """
    import matplotlib.pyplot as plt

    To_min = 5      # Mean of minimum outside temperature in °C
    To_max = 15     # Mean of maximum outside temperature in °C
    Tout_init_props = TempInitProps(mean_min=To_min, spread_min=3, mean_max=To_max, spread_max=3)
    EnvTempSim = EnvironmentTemperatureSimulator(Tout_init_props)

    Ti_min = 16     # Mean of minimum inside target temperature in °C
    Ti_max = 21     # Mean of maximum inside target temperature in °C
    Tin_init_props = TempInitProps(mean_min=Ti_min, spread_min=1, mean_max=Ti_max, spread_max=1)
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
