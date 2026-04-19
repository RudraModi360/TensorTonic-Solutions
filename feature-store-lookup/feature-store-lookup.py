def feature_store_lookup(feature_store, requests, defaults):
    """
    Join offline user features with online request-time features.
    """
    results=[]
    for req in requests:
        if feature_store.get(req['user_id']) is not None:
            req['online_features']=req['online_features']|feature_store[req['user_id']]
            results.append(req['online_features'])
        else:                     
            req['online_features']=req['online_features']|defaults
            results.append(req['online_features'])
    return results