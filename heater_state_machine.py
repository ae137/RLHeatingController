
class HeaterState(object):
    def __init__(self, numStates=2):
        self.minState = 0
        assert numStates >= 2, "Heater state machine needs to have at least two states"
        self.maxState = numStates - 1   # Delay until fully switched on in 1/4 hours
        self.state = self.minState

    def on_event(self, input):
        assert (input == 0 or input == 1), "Called heater state machine with bad state transition"
        stateProposal = (self.state + 1) if input else (self.state - 1)
        self.state = max(min(self.maxState, stateProposal), self.minState)

        return self.state / self.maxState

    def reset(self):
        self.state = self.minState
