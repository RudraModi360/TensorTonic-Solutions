def simple_moving_average(values, window_size):
    """
    Compute the simple moving average of the given values.
    """
    result=[]
    st=0
    end=st+window_size
    while end<=len(values):
        val=values[st:end]
        result.append(sum(val)/window_size)
        st+=1
        end=st+window_size
    return result