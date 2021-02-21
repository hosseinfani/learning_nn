lm = {
    'h': 100,#size of hidden space
    'w': 5, #context window size P(w+1 | 1, ..., w)
    'd': None,#size of input semantic vectors. For sparse vector methods, depends on #doc for sparse semantic vectors
    'v': None, ##size of vocabularly (unique tokens) and depends on vocab set
    'lr': 0.01,
    'b': 1000, #batch
    'e': 10, #epochs
    'g': 1 #gpu
}