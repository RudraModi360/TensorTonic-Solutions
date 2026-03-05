import math

def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    dcg=0.0
    idcg=0.0
    if len(relevance_scores)>k:
        for i in range(1,k+1):
            dcg+=(math.pow(2,relevance_scores[i-1])-1)/(math.log2(i+1))
    else :
        for i in range(1,len(relevance_scores)+1):
            dcg+=(math.pow(2,relevance_scores[i-1])-1)/(math.log2(i+1))
    relevance_scores_sorted=sorted(relevance_scores,reverse=True)
    if len(relevance_scores)>k:
        for i in range(1,k+1):
            idcg+=(math.pow(2,relevance_scores_sorted[i-1])-1)/(math.log2(i+1))
    else :
        for i in range(1,len(relevance_scores)+1):
            idcg+=(math.pow(2,relevance_scores_sorted[i-1])-1)/(math.log2(i+1))
    if idcg ==0.0:
        return 0.0
    else :
        return dcg/idcg