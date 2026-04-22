def convert_json(schema):
    """
    Convert schema list into dict: column -> rules
    """
    json_opp = {}
    for record in schema:
        json_opp[record["column"]] = record
    return json_opp


def fullfill_constraints(key, val, rules):
    """
    Validate a single value against rules.
    Returns list of errors (instead of raising).
    """
    errors = []

    # Null check
    if val is None:
        if not rules.get("nullable", False):
            errors.append(f"{key}: null")
        return errors  # skip further checks

    expected_type = rules["type"]
    actual_type = type(val).__name__

    # Type check
    type_ok = False
    if expected_type == "int":
        type_ok = (type(val) is int)
    elif expected_type == "float":
        type_ok = (type(val) is float or type(val) is int)
    elif expected_type == "str":
        type_ok = (type(val) is str)

    if not type_ok:
        errors.append(f"{key}: expected {expected_type}, got {actual_type}")
        return errors  # skip range

    # Range check
    if expected_type in ["int", "float"]:
        if "min" in rules and val < rules["min"]:
            errors.append(f"{key}: out of range")
        elif "max" in rules and val > rules["max"]:
            errors.append(f"{key}: out of range")

    return errors


def validate_records(records, schema):
    """
    Validate records against a schema definition.
    Returns list of tuples: (record_index, is_valid, errors)
    """
    json_schema = convert_json(schema)
    results = []

    for idx, record in enumerate(records):
        errors = []

        # iterate in schema order
        for col_def in schema:
            key = col_def["column"]

            # Missing column
            if key not in record:
                errors.append(f"{key}: missing")
                continue

            val = record[key]

            # Apply validation
            errs = fullfill_constraints(key, val, json_schema[key])
            errors.extend(errs)

        results.append((idx, len(errors) == 0, errors))

    return results