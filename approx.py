import numpy as np

def approximate_credal_set(K, ext_P, epsilon=0.01, max_iter=1000):
    """
    Approximates an initial credal set P by a new credal set P' with L=3 extreme points.
    
    Parameters:
    K (int): Number of gambles to generate for the Monte-Carlo approximation.
    ext_P (list or np.ndarray): Extreme points of the original credal set P.
    epsilon (float): Small value for local improvement steps.
    max_iter (int): Maximum number of iterations for the greedy search.
    
    Returns:
    np.ndarray: The extreme points of the approximated credal set P'.
    float: The final NS_4 value.
    """
    n = 3
    ext_P = np.array(ext_P)
    
    # 1. Obtain the initial credal set P' by including the degenerate probabilities
    # Since L=3 and n=3, we use the 3 degenerate distributions where P(x_i) = 1.
    ext_P_prime = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    # 2. Randomly select K gambles from L_{0,1} (uniform distribution in [0,1]^3)
    K_gambles = np.random.uniform(0, 1, size=(K, n))

    def get_bounds(credal_set, f):
        """Computes lower and upper previsions (expectations) of a gamble f."""
        expected_values = np.dot(credal_set, f)
        return np.min(expected_values), np.max(expected_values)

    def compute_NS4(credal_set):
        """Computes the NS_4 measure for a given credal set over K gambles."""
        imprecision_sum = 0.0
        for f in K_gambles:
            lower, upper = get_bounds(credal_set, f)
            imprecision_sum += (upper - lower)
        return imprecision_sum / K

    def is_included(P_original, P_new):
        """
        Approximately tests if P_original is included in P_new by checking 
        bounds over the K gambles.
        """
        for p in P_original:
            for f in K_gambles:
                lower_new, upper_new = get_bounds(P_new, f)
                expected_p = np.dot(p, f)
                # Check if the expectation falls within the new credal set's bounds
                if expected_p < lower_new - 1e-9 or expected_p > upper_new + 1e-9:
                    return False
        return True

    current_ns4 = compute_NS4(ext_P_prime)
    changed = True
    iterations = 0

    # 3. Greedy algorithm with local improvements
    while changed and iterations < max_iter:
        changed = False
        iterations += 1

        for p_idx in range(len(ext_P_prime)):
            p = ext_P_prime[p_idx]
            
            for i in range(n):
                for j in range(n):
                    if i != j and p[i] > 0:
                        # Create a local improvement p_new by shifting mass from index i to j
                        p_new = p.copy()
                        transfer = min(p[i], epsilon)
                        p_new[i] -= transfer
                        p_new[j] += transfer

                        # Build the candidate credal set P''
                        ext_P_double_prime = ext_P_prime.copy()
                        ext_P_double_prime[p_idx] = p_new

                        # Condition: P must be included in the new credal set P''
                        if is_included(ext_P, ext_P_double_prime):
                            new_ns4 = compute_NS4(ext_P_double_prime)
                            
                            # Condition: NS_4(P'') must be strictly lower than NS_4(P')
                            if new_ns4 < current_ns4 - 1e-9:
                                ext_P_prime = ext_P_double_prime
                                current_ns4 = new_ns4
                                changed = True
                                break # Apply improvement and restart search
                if changed:
                    break
            if changed:
                break

    return ext_P_prime, current_ns4

#  Example usage:
K = 1000
# Example original credal set P defined by some extreme points
ext_P = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4], [0.7,0.3,0.0]]
P_prime, final_ns4 = approximate_credal_set(K, ext_P)
print("Approximated P':\n", P_prime)
print("Final NS_4:", final_ns4)