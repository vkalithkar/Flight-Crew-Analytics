import pandas as pd

def label_window(window_df, state_col, baseline_label=0):
    """
    Label a 30s window:
      - 0 if no event (only baseline)
      - event label if any event present
      - majority event label if multiple events (ignoring baseline time)
    
    window_df  : rows belonging to this window
    state_col  : column containing per-sample state labels
    baseline_label : the value that counts as "no event" (default 0)
    """
    states = window_df[state_col]
    
    # Remove baseline samples entirely before comparing events
    events = states[states != baseline_label]
    
    if events.empty:
        return baseline_label
    
    # Return whichever event took up the most time (most samples)
    return events.value_counts().idxmax()


def label_all_windows(df, time_col, state_col, subject_col, 
                      window_size=30, baseline_label=0):
    """
    Apply label_window across all 30s windows for all subjects.
    
    df           : raw time series dataframe
    time_col     : column with timestamps (seconds)
    state_col    : column with per-sample ground truth states
    subject_col  : column with subject IDs (windows never cross subjects)
    window_size  : in seconds (default 30)
    baseline_label : value representing no event
    """
    results = []

    for subject, subj_df in df.groupby(subject_col):
        subj_df = subj_df.sort_values(time_col).reset_index(drop=True)
        
        t_start = subj_df[time_col].iloc[0]
        t_end   = subj_df[time_col].iloc[-1]
        
        window_start = t_start
        while window_start < t_end:
            window_end = window_start + window_size
            
            mask = (subj_df[time_col] >= window_start) & \
                   (subj_df[time_col] <  window_end)
            window_df = subj_df[mask]
            
            if window_df.empty:
                window_start = window_end
                continue
            
            label = label_window(window_df, state_col, baseline_label)
            
            results.append({
                subject_col:   subject,
                "window_start": window_start,
                "window_end":   window_end,
                "label":        label,
                "n_samples":    len(window_df),
                "n_event_samples": (window_df[state_col] != baseline_label).sum(),
            })
            
            window_start = window_end

    return pd.DataFrame(results)