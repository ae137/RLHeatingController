import temperature_simulator as temp_sim
import heater_state_machine as heater_state

To_min = 0          # Mean of minimum outside temperature in °C
To_max = 15         # Mean of maximum outside temperature in °C
T_target_low = 16   # Mean of minimum inside target temperature in °C
T_target_high = 21  # Mean of maximum inside target temperature in °C
heater_delay = 4    # Time delay of heater

Tin_init_props = temp_sim.TempInitProps(mean_min=T_target_low, spread_min=2,
                                        mean_max=T_target_high, spread_max=2)
Tout_init_props = temp_sim.TempInitProps(mean_min=To_min, spread_min=3,
                                         mean_max=To_max, spread_max=3)

env_config_dict = \
    {
        "h": 0.15,                  # Heating strength
        "l": 0.025,                 # Energy loss coefficient
        "len_hist": 3,              # History length for last temperatures
        "temp_diff_penalty": 1,
        "action_penalty": 0.25,
        'horizon': 400,
        "out_temp_sim": temp_sim.EnvironmentTemperatureSimulator(Tout_init_props),
        "tgt_temp_sim": temp_sim.TargetTemperatureSimulator(Tin_init_props),
        "heater": heater_state.HeaterState(heater_delay)
    }
