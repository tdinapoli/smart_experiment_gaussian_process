import numpy as np
import matplotlib.pyplot as plt

# ------------- Settings
np.random.seed(0)
sigma_noise = 0.1  # measurement noise
n_params = 2  # number of unknown parameters
n_initial_samples = 1000  # number of samples from prior

# ------------- The true system
P_true = np.array([2.0, -1.0])  # unknown to us!

def model(X, P):
    """The true model: simple linear + nonlinear"""
    return P[0] * X + P[1] * np.sin(X)

# ------------- Prior over P
P_samples = np.random.uniform(low=-5, high=5, size=(n_initial_samples, n_params))

# ------------- Initial observations (empty)
X_obs = []
Y_obs = []

# ------------- Utility functions
def simulate_Y(X, P_samples):
    """Simulate outcomes Y at X given parameter samples P"""
    return np.array([model(X, P) for P in P_samples])

def likelihood(Y_obs, Y_pred, sigma):
    """Gaussian likelihood"""
    return np.exp(-0.5 * ((Y_obs - Y_pred) / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)

def update_weights(P_samples, weights, X_new, Y_new):
    """Update particle weights given new observation"""
    Y_preds = simulate_Y(X_new, P_samples)
    new_weights = weights * likelihood(Y_new, Y_preds, sigma_noise)
    new_weights /= np.sum(new_weights)  # normalize
    return new_weights

def expected_information_gain(X_cand, P_samples, weights):
    """Estimate expected information gain for a candidate X"""
    # Simulate possible outcomes
    Y_preds = simulate_Y(X_cand, P_samples)
    
    # Discretize possible outcomes (for simplicity)
    Y_bins = np.linspace(np.min(Y_preds), np.max(Y_preds), 10)
    
    expected_KL = 0.0
    for y in Y_bins:
        # Update weights for hypothetical outcome y
        new_w = weights * likelihood(y, Y_preds, sigma_noise)
        if np.sum(new_w) == 0:
            continue
        new_w /= np.sum(new_w)
        
        # Compute KL divergence
        kl = np.sum(new_w * (np.log(new_w + 1e-12) - np.log(weights + 1e-12)))
        p_y = np.sum(weights * likelihood(y, Y_preds, sigma_noise))
        expected_KL += p_y * kl
    return expected_KL

# ------------- Main loop
weights = np.ones(n_initial_samples) / n_initial_samples

for iteration in range(10):
    # Candidates to choose from
    X_candidates = np.linspace(-5, 5, 20)
    
    # Evaluate expected information gain for each candidate
    gains = []
    for X_cand in X_candidates:
        gain = expected_information_gain(X_cand, P_samples, weights)
        gains.append(gain)
    
    # Pick best candidate
    X_next = X_candidates[np.argmax(gains)]
    
    # Perform experiment
    Y_next = model(X_next, P_true) + np.random.normal(0, sigma_noise)
    print(f"Iteration {iteration}: measure at X={X_next:.2f}, observed Y={Y_next:.2f}")
    
    # Update observations
    X_obs.append(X_next)
    Y_obs.append(Y_next)
    
    # Update belief over P
    weights = update_weights(P_samples, weights, X_next, Y_next)

# ------------- Plot posterior samples
plt.figure(figsize=(8,4))
plt.scatter(P_samples[:,0], P_samples[:,1], c=weights, cmap='viridis', s=5)
plt.colorbar(label="posterior weight")
plt.plot(P_true[0], P_true[1], 'rx', markersize=10, label="True P")
plt.xlabel("P0")
plt.ylabel("P1")
plt.legend()
plt.title("Posterior belief over parameters P after 10 measurements")
plt.show()
