def relu(z):
    return z * (z > 0)

def softmax(z):
    exps = np.exp(z - np.max(z))
    return exps/exps.sum(axis = 1, keepdims = True)
