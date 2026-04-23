def assign_labels(window_df, state_col, states, purity_threshold=0.7):
    counts = window_df[state_col].value_counts()
    total = len(window_df)
    
    proportions = {s: counts.get(s, 0) / total for s in states}
    max_prop = max(proportions.values())
    majority = max(proportions, key=proportions.get)
    
    if max_prop >= purity_threshold:
        # Clean window — hard label, high sample weight
        return {
            "label": majority,
            "sample_weight": max_prop,
            **{f"p_{s}": proportions[s] for s in states}
        }
    else:
        # Mixed window — soft label, downweighted
        return {
            "label": majority,        # still need a hard label for XGBoost
            "sample_weight": max_prop, # low weight = model less confident here
            **{f"p_{s}": proportions[s] for s in states}
        }