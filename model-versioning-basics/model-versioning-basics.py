def promote_model(models):
    """
    Decide which model version to promote to production.
    """
    return sorted(
        models,
        key=lambda m: (
            -m["accuracy"],     # higher is better
            m["latency"],       # lower is better
            -int(m["timestamp"].replace("-", ""))  # later date is better
        )
    )[0]["name"]