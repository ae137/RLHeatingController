import heating_control
import temperature_simulator as temp_sim

import matplotlib.pyplot as plt

if __name__ == "__main__":    
    t_init = 0
    To_min = 5
    To_max = 5
    T_target = 21

    config = {"h": 0.15,
            "l": 0.025,
            "temp_diff_penalty": 1,
            "out_temp_sim": temp_sim.EnvironmentSimulator(t_init, To_min, To_max),
            "target_temp": temp_sim.TargetTemperature(t_init, T_target)}
    
    env = heating_control.HeatingEnv(config)
    
    To_list = []
    Ti_list = []
    steps_list = []
    
    for i in range(8):
        Tnew, *_ = env.step(1)
        Ti_list.append(Tnew[0])
        To_list.append(Tnew[1])
        steps_list.append(i)
        
    plt.plot(steps_list, Ti_list, label='Ti')
    # plt.plot(steps_list, To_list, label='To')
    plt.legend()
    plt.show()
