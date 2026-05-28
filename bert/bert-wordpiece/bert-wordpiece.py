from typing import List, Dict

class WordPieceTokenizer:
    """
    WordPiece tokenizer for BERT.
    """
    
    def __init__(self, vocab: Dict[str, int], unk_token: str = "[UNK]", max_word_len: int = 100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_word_len = max_word_len
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into WordPiece tokens.
        """
        tokens = []
        for word in text.lower().split():
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        return tokens
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word into subwords using
        greedy longest-match-first WordPiece algorithm.
        """
        
        if len(word) > self.max_word_len:
            return [self.unk_token]
        
        subwords = []
        start = 0
        
        while start < len(word):
            end = len(word)
            current_subword = None
            
            while start < end:
                piece = word[start:end]
                
                # Add ## for continuation pieces
                if start > 0:
                    piece = "##" + piece
                
                if piece in self.vocab:
                    current_subword = piece
                    break
                
                end -= 1
            
            # No valid subword found
            if current_subword is None:
                return [self.unk_token]
            
            subwords.append(current_subword)
            start = end
        
        return subwords