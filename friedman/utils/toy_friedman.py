import numpy as np

def friedman(X, eps=0.1, random_state=0):
    """
    Generates the Friedman regression problem
    Inputs:
        - X : np.array(N, 5), input value
        - eps: np.float, amount of noise
    Outputs:
        - y: np.array(N), target value
    """
    if random_state is not None:
        np.random.seed(random_state)
    return 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4] + eps * np.random.randn(len(X))

def generate_x_friedman(n_domains = 10, N = 100, alpha = np.ones(9)/9, noise=0.5, target_shift_mean=0, target_shift_std=0, random_state=0):
    """
    Generates source input X according to normal distribution centered on 1 or -1 for every feature and target input as a
    combination of the inputs with a shift
    Inputs:
        - n_domains: number of domains to generate
        - N: number of samples in each domain
        - alpha: weight of each source to generate the target distribution
        - target_shift_mean: mean on each shift of the target X[:,f]
        - target_shift_std: std on each shift of the target X[:,f]
    """
    assert len(alpha)==n_domains-1
    if random_state is not None:
        np.random.seed(random_state)
    X = np.zeros((n_domains, N, 5))
    rest = N - np.sum([int(alpha[n]*N) for n in range(n_domains-1)])
    idx = np.random.choice(np.arange(n_domains-1), rest)
    counts = np.zeros(n_domains-1).astype('int')
    for i in idx:
        counts[i] = counts[i]+1
    j=0
    for n in range(n_domains-1):
        for f in range(5):
            mu = np.random.choice([-1,1])
            X[n, :, f] = mu + np.random.randn(N)*noise
            X[n_domains-1, j:j+int(alpha[n]*N),f] = mu+target_shift_mean+np.random.randn()*target_shift_std + np.random.randn(int(alpha[n]*N))*noise
            X[n_domains-1, j+int(alpha[n]*N):j+int(alpha[n]*N)+counts[n],f] = mu+target_shift_mean+np.random.randn()*target_shift_std + np.random.randn(counts[n])*noise
        j = j+int(alpha[n]*N)+counts[n]
    return X 

def generate_y_friedman(X, normalize = True, random_state=0, eps=0.1):
    """
    Generates regression values according to Friedman    
    Inputs:
        - n_domains: number of domains to generate
        - N: number of samples in each domain
        - alpha: weight of each source to generate the target distribution
        - target_shift_mean: mean on each shift of the target X[:,f]
        - target_shift_std: std on each shift of the target X[:,f]
    """
    if random_state is not None:
        np.random.seed(random_state)

    y = np.array([friedman(x, eps=eps) for x in X])
    if normalize:
        mu_y, std_y = np.mean(y), np.std(y)
        y = np.array([(yy-mu_y)/std_y for yy in y])
        return y
    else:
        return y
    
def generate_x_friedman_clusters(n_domains = 10, N = 100, alpha = np.ones(9)/9, clusters=None, noise=0.1, cluster_noise=0.1,
                                 target_shift_mean=0, target_shift_std=0, random_state=0):
    """
    Generates source input X according to normal distribution centered on 1 or -1 for every feature and target input as a
    combination of the inputs with a shift
    Inputs:
        - n_domains: number of domains to generate
        - N: number of samples in each domain
        - alpha: weight of each source to generate the target distribution
        - target_shift_mean: mean on each shift of the target X[:,f]
        - target_shift_std: std on each shift of the target X[:,f]
    """
    assert len(alpha)==n_domains-1
    if random_state is not None:
        np.random.seed(random_state)
    if clusters is None:
        clusters = np.arange(len(clusters))
    n_clusters = len(clusters)
    
    X = np.zeros((n_domains, N, 5))
    rest = N - np.sum([int(alpha[n]*N) for n in range(n_domains-1)])
    idx = np.random.choice(np.where(alpha!=0)[0], rest)
    counts = np.zeros(n_domains-1).astype('int')
    for i in idx:
        counts[i] = counts[i]+1
    j=0
    mu_clusters = np.array([[np.random.choice([-1,1]) for f in range(5)] for k in range(n_clusters)])
    for n in range(n_domains-1):
        for f in range(5):
            mu = mu_clusters[clusters[n], f]+np.random.randn()*cluster_noise
            X[n, :, f] = mu + np.random.randn(N)*noise
            X[n_domains-1, j:j+int(alpha[n]*N),f] = mu+target_shift_mean+np.random.randn()*target_shift_std + np.random.randn(int(alpha[n]*N))*noise
            X[n_domains-1, j+int(alpha[n]*N):j+int(alpha[n]*N)+counts[n],f] = mu+target_shift_mean+np.random.randn()*target_shift_std + np.random.randn(counts[n])*noise
        j = j+int(alpha[n]*N)+counts[n]
    return X 
