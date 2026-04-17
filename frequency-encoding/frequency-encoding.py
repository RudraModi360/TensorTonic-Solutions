from collections import Counter

def frequency_encoding(values):
    """
    Replace each value with its frequency proportion.
    """
    result={}
    final_response=[]
    counts=Counter(values)
    for key , val in counts.items():
        result[key]=val/len(values)

    for val in values:
        res=result[val]
        final_response.append(res)
    return final_response