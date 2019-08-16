import math

class EnvironmentSimulator(object):
    def __init__(self, start_time, To_min, To_max):
        self.start_time = start_time
        self.To_min = To_min
        self.To_max = To_max
        
    def getOutTemp(self, time_step):
        return self.To_min + (self.To_max - self.To_min) \
            * (math.sin(math.pi * (time_step - self.start_time) / 96))**2
    
    def getInitTime(self):
        return self.start_time
    
class TargetTemperature(object):
    def __init__(self, start_time, T_target):
        self.start_time = start_time
        self.T_target = T_target    # Can later be replaced by more complicated target

    def getTargetTemp(self, time):
        return self.T_target
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    EnvTempSim = EnvironmentSimulator(0, -5, 10)
    
    time_hours = []
    temp_data = []
    for time_step in range(96 * 2):
        time_hours.append(time_step / 4)
        temp_data.append(EnvTempSim.getOutTemp(time_step))
    
    plt.plot(time_hours, temp_data)
    plt.show()
