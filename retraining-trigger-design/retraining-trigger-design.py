def retraining_policy(daily_stats, config):
    days_since_retrain = 0
    cooldown_counter = 0
    retrain_days = []

    for daily in daily_stats:
        # decrement cooldown if active
        if cooldown_counter > 0:
            cooldown_counter -= 1

        # increment staleness first (important!)
        days_since_retrain += 1

        drift_trigger = daily['drift_score'] > config['drift_threshold']
        perf_trigger = daily['performance'] < config['performance_threshold']
        stale_trigger = days_since_retrain == config['max_staleness']

        if (drift_trigger or perf_trigger or stale_trigger) and cooldown_counter == 0:
            if config['budget'] >= config['retrain_cost']:
                retrain_days.append(daily['day'])
                config['budget'] -= config['retrain_cost']
                days_since_retrain = 0
                cooldown_counter = config['cooldown']

    return retrain_days