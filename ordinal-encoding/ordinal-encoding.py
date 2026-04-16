def ordinal_encoding(values, ordering):
    """
    Encode categorical values using the provided ordering.
    """
    mapping_dict={}

    for i,order in enumerate(ordering):
        mapping_dict[order]=i

    return [mapping_dict[val] for val in values]