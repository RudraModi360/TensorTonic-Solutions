import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    b,num_tokens,d_in=Q.shape
    q=np.dot(Q,W_q)
    k=np.dot(K,W_k)
    v=np.dot(V,W_v)

    d_out=q.shape[-1]
    head_dim=d_in//num_heads

    q=q.reshape(b, num_tokens, num_heads, head_dim)
    k=k.reshape(b, num_tokens, num_heads, head_dim)
    v=v.reshape(b, num_tokens, num_heads, head_dim)
    
    q=np.transpose(q, (0, 2, 1, 3))
    k=np.transpose(k, (0, 2, 1, 3))
    v=np.transpose(v, (0, 2, 1, 3))

    attn_scores=q@np.transpose(k,(0,1,3,2))
    attn_scores=attn_scores/np.sqrt(head_dim)

    attn_weights = softmax(attn_scores, axis=-1)

    context_vec=attn_weights@v

    context_vec = np.transpose(context_vec, (0, 2, 1, 3))
    context_vec = context_vec.reshape(b, num_tokens, d_out)

    output = context_vec @ W_o
    return output