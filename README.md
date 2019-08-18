# Training a heating controller with reinforcement learning

## Motivation
In many houses (or apartments), heating systems are temperature controlled (for example by room thermostats) but not time controlled. This is inefficient and inconvenient, because it does not allow to lower the room temperature for saving energy at night or when nobody is in a room for long time, and at the same time provide comfortable temperatures when people use a room.

In this repository, we develop a heating controller with reinforcement learning, using [Ray / rllib](https://ray.readthedocs.io/en/latest/index.html) as reinforcement learning framework. We implement an OpenAI gym environment in which a controller can be trained virtually. Besides, we implement a simple simulator that allows to visualize the behaviour of the agent. Our goal is, later, to bring the trained controller to a mini computer and use it for controlling the heating system in an apartment.

## Usage
 1. Check out this repository
 2. Set up `anaconda` environment by running `conda env create -f environment.yml`.
 3. Run `python heating_controller_train.py` after configuring simulator (mostly setting target temperature and outside temperature).
 4. Configure path to training checkpoint in `heating_controller_simulate.py` (and adapt configuration from `heating_controller_train.py`) and run `python heating_controller_simulate.py` for visualizing the effect of the controller.

## Results
`TODO`



## License
GPLv3, see [LICENSE](https://github.com/ae137/SYKQuench/blob/master/LICENSE) for more information