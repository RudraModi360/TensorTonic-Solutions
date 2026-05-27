import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    return gamma*((x-np.mean(x,keepdims=True,axis=-1))/(np.sqrt(np.std(x,keepdims=True,axis=-1)**2+eps)))+beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    b,num_tokens,d_in=Q.shape
    d_out=Q.shape[-1]
    
    q=np.dot(Q,W_q)
    k=np.dot(K,W_k)
    v=np.dot(V,W_v)

    head_dim=d_out//num_heads

    q=q.reshape(b,num_tokens,num_heads,head_dim)
    k=k.reshape(b,num_tokens,num_heads,head_dim)
    v=v.reshape(b,num_tokens,num_heads,head_dim)

    q=np.transpose(q,(0,2,1,3))
    k=np.transpose(k,(0,2,1,3))
    v=np.transpose(v,(0,2,1,3))

    attn_scores=q@np.transpose(k,(0,1,3,2))
    attn_weights=attn_scores/(np.sqrt(head_dim))

    attn_weights=softmax(attn_weights)

    context_vec=attn_weights@v

    context_vec = np.transpose(context_vec, (0, 2, 1, 3))
    context_vec=context_vec.reshape(b,num_tokens,d_out)

    context_vec=context_vec@W_o

    return context_vec

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    return np.dot(np.maximum(0,np.dot(x,W1)+b1),W2)+b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    attn_out=multi_head_attention(Q=x,K=x,V=x,W_q=W_q,W_k=W_k,W_v=W_v,W_o=W_o,num_heads=num_heads)
    x = layer_norm(x + attn_out, gamma1, beta1)
    ffn_out = feed_forward(x, W1, b1, W2, b2)
    x = layer_norm(x + ffn_out, gamma2, beta2)
    return x