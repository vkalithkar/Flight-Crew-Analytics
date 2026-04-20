# Flight-Crew-Analytics

Names:
	Katy L
	Sierra L.
	Sheoli L.
	Vandana K.

# Flight Crew Physiological Data for Crew State Monitoring

This project explores physiological data collected fromm pilot/copilot pairs across multiple experimental conditions, including startle/surprise, channelized attention, and a full flight simulator. EEG signature, ECG, Respiration, and GSR (galvanic skin response) were all measured. This data is sourced from the NASA Open Data Portal.

### Experimental Conditions

1. **Flight Simulation (LOFT)**
    - Realistic flight simulator environment
    - Designed to mimic real-world cockpit operations

2. **Channelized Attention (CA)**
    - Tasks designed to induce intense focus on a single goal
    - Simulates "tunnel vision" under workload

3. **Diverted Attention (DA)**
    - Tasks requiring attention to be split accross multiple stimuli
    - Simulates multitasking/distraction

4. **Startle/Surprise (SS)**
    - Sudden, unexpected events introduced during a task
    - Designed to mimic what could happen during flight when something does go according to plan

## Research Questions

Can we train a model to predict pilot cognitive states?
    - Channelized Attention (CA)
    - Diverted Attention (DA)
    - Startle/Surprise (SS)

## Repository Structure
Flight-Crew-Analytics/
│
├── scripts/
├── config/
├── output/
├── data/
└── pipeline.slurm
