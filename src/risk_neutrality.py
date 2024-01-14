import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt

# Black-Scholes Vega Function
def black_scholes_vega(S, K, T, r, sigma, num_contracts=1):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    vega = S * si.norm.pdf(d1) * np.sqrt(T) * num_contracts
    return vega

# Implied Volatility Function
def implied_vol_skew(K, base_vol=0.2, skew_factor=0.005):
    return base_vol + skew_factor * (100 - K)

# Create Portfolio
np.random.seed(0)
num_options = 50
strikes = np.linspace(50, 150, num_options)
num_contracts = np.random.randint(-5, 10, num_options)
expiry = 1  # 1 year
spot = 100  # Current spot price
risk_free_rate = 0.01

portfolio = [{'K': K, 'T': expiry, 'r': risk_free_rate, 'sigma': implied_vol_skew(K), 'num_contracts': n} 
             for K, n in zip(strikes, num_contracts)]

# Function to Aggregate Vega into Buckets
def aggregate_vega(portfolio, spot):
    bucket_width = 7.5  # 5% of 150 (maximum strike)
    buckets = np.arange(50, 150 + bucket_width, bucket_width)  # Extend to cover the highest strike
    vega_buckets = np.zeros(len(buckets) - 1)

    for opt in portfolio:
        vega = black_scholes_vega(spot, opt['K'], opt['T'], opt['r'], opt['sigma'], opt['num_contracts'])
        index = np.digitize(opt['K'], buckets) - 1
        index = min(index, len(vega_buckets) - 1)  # Handle edge case
        vega_buckets[index] += vega

    return buckets, vega_buckets


# Aggregate Vega at Initial Spot
buckets, vega_buckets_initial = aggregate_vega(portfolio, spot)

# Bump Spot Price
bump = 20  # e.g., 5%
new_spot = spot * (1 + bump / 100)

# Aggregate Vega after Spot Bump
_, vega_buckets_bumped = aggregate_vega(portfolio, new_spot)

# Plotting
plt.figure(figsize=(12, 6))

# Initial Vega Profile
plt.subplot(1, 2, 1)
plt.bar(buckets[:-1], vega_buckets_initial, width=7.5, alpha=0.6, label='Initial Vega')
plt.xlabel('Strike Price Buckets')
plt.ylabel('Aggregated Vega')
plt.title('Initial Vega Profile')
plt.legend()

# Vega Profile after Spot Bump
plt.subplot(1, 2, 2)
plt.bar(buckets[:-1], vega_buckets_bumped, width=7.5, alpha=0.6, color='orange', label='Post-Bump Vega')
plt.xlabel('Strike Price Buckets')
plt.ylabel('Aggregated Vega')
plt.title('Vega Profile after Spot Bump')
plt.legend()

plt.tight_layout()
plt.show()

# Line Plots for Comparison
plt.figure(figsize=(12, 6))
plt.plot(buckets[:-1], vega_buckets_initial, marker='o', label='Initial Vega', linestyle='-', color='blue')
plt.plot(buckets[:-1], vega_buckets_bumped, marker='x', label='Post-Bump Vega', linestyle='-', color='orange')
plt.xlabel('Strike Price Buckets')
plt.ylabel('Aggregated Vega')
plt.title('Vega Profile Comparison: Initial vs. Post-Bump')
plt.legend()
plt.grid(True)
plt.show()
