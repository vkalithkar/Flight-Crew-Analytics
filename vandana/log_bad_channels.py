# this script might hopefully prove to be unnecessary if it works with PyPREP

# All the pilot patients, each has num_CA.csv, num_DA.csv, num_LOFT.csv, num_SS.csv
pilots = [5,6,9,10,11,12,13,14,17,18,19,21,22,23,24,25,26]
runs = ["CA", "DA", "SS","LOFT"]

# Bad channels 
bad_channels = {
    f"{p}_{run}": [] 
    for p in pilots 
    for run in runs
}

# Logger Function
def log_bad(pilot, run, channels):
    """
    Adds channels to a specific run for a pilot.
    Usage:
        log_bad(11, 'CA', 'EEG_F4')
        log_bad(11, 'all', ['EEG_F4', 'EEG_Cz'])
    """
    pilot_id = str(pilot)

    # Convert single string to list if necessary
    if isinstance(channels, str):
        channels = [channels]

    key = f"{pilot}_{run}"
    bad_channels[key] = list(set(bad_channels[key] + channels))


# The bad channels I found, for transparency
log_bad(5, 'CA', ['EEG_F4', 'EEG_F1'])

print(bad_channels)

