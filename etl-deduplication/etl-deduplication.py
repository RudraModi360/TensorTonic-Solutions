from collections import defaultdict

def deduplicate(records, key_columns, strategy):
    """
    Deduplicate records by key columns using the given strategy.
    """

    groups = defaultdict(list)
    order = []  # to preserve first appearance order

    # Step 1: group records by composite key
    for record in records:
        key = tuple(record[col] for col in key_columns)

        if key not in groups:
            order.append(key)  # track first appearance

        groups[key].append(record)

    results = []

    # Step 2: apply strategy
    for key in order:
        vals = groups[key]

        if strategy == "first":
            results.append(vals[0])

        elif strategy == "last":
            results.append(vals[-1])

        elif strategy == "most_complete":
            # pick record with fewest None values
            best = vals[0]
            min_none = sum(1 for v in best.values() if v is None)

            for rec in vals[1:]:
                none_count = sum(1 for v in rec.values() if v is None)

                if none_count < min_none:
                    best = rec
                    min_none = none_count

            results.append(best)

        else:
            raise ValueError("Invalid strategy")

    return results