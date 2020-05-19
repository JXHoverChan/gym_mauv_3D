# Project Title

This repo implements a 6-DOF simulation model for an AUV according to the stable baselines (OpenAI) interface for reinorcement learning control.

## Getting Started

To install all packages needed in your virtual environment, run:

```
pip install requirements.txt
```
 
### Training an agent:

All hyperparameters and setup can be tuned in the file [train.py](https://github.com/simentha/gym-auv/blob/master/train3d.py).

For training an agent, run:

```
python train.py --exp_id [x]
```

Where x is the experiment id number. 


## Running an agent in the environment

For running an agent in any scenario, use:

```
python run.py --exp_id [x] --scenario [scenario] --controller_scenario [controller_scenario] --controller [y]
```

Where x is the experiment id number, scenario is what scenario to run, controller_scenario is which scenario the controller was trained in and y is
which agent number to run. If no y is provided, the agent called "last_model.pkl" is chosen. Scenarios can be either of "beginner", "intermediate",
"proficient", "advanced", "expert", "test_path", "test_path_current" (Path following with disturbance), "horizontal", "vertical" or "deadend". 


