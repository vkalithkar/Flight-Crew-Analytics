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

**Can we train a model to predict pilot cognitive states?**
    
- Channelized Attention (CA)
- Diverted Attention (DA)
- Startle/Surprise (SS)


### Project Checklist 

1. Problem definition -- what are we trying to do with this data, and what does success look like here?

	- Classification project -- build model to predict which cognitive state pilot is in
	- States = CA, DA, SS, but also baseline
	- What are we looking at ? EEG signature, ECG, respiration, and GSR (Galvanic skin response)
	- Evaluation metrics (ideas) : F1, AUC?

2. Understanding the data / EDA -- getting to know the data

	- Beware : high-frequency 256 Hz, so 256 readings per second
	
	- Guiding questions for EDA : 

		1) What are the distributions of each sensor signal across cognitive states?
	        2) Is the data balanced across classes (including "baseline")
	        3) Are there obvious artifacts or noise bursts visible in the time series plots?
	        4) How do signals differ between plots?
	
3. Preprocessing/cleaning -- uniquely important because of how messy physiological data tends to be. Some tasks: 

	- Removing artifacts like signal spikes, dropots, motion artifacts, other unhelpful noise
	- Filtering (basically smoothing noise while trying to presere meaningful signal features)
	- Handle any missing values
	- Normalization/standardization -- especially important here,since physiological baselines differ between people

4. Feature engineering -- this step will require thorough domain research on each feature

5. Model selection/training (how set are we on XGBoost? Given the time limit, this could be the best idea) 

6. Evaluation -- evaluting our model on the LOFT test set (from the "flight simulation" portion of the dataset)

7. Interpretation, iteration of steps, etc
