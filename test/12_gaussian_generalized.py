import numpy as np
from scipy.stats import gennorm

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def main():
    beta = 1.3
    mean, var, skew, kurt = gennorm.stats(beta, moments='mvsk')
    print('Real params:\n\t- Mean: %f\n\t- Var: %f\n\t- Beta: %f' % (mean, var, beta))

    # Generate histogram
    r = gennorm.rvs(beta, size=1000)

    # Get params
    beta_est, mean_est, var_est = gennorm.fit(r)
    print('Fitted params:\n\t- Mean: %f\n\t- Var: %f\n\t- Beta: %f' % (mean, var, beta))

    # Generate pdf
    x = np.linspace(gennorm.ppf(0.01, beta_est), gennorm.ppf(0.99, beta_est), 100)
    y = gennorm.pdf(x, beta_est)
    
    fig, ax = plt.subplots(1, 1)
    ax.hist(r, bins=100, color='b', density=True, histtype='stepfilled', alpha=0.3)
    ax.legend(loc='best', frameon=False)

    ax.plot(x, y, 'r-', lw=5, alpha=0.6, label='gennorm pdf')
    plt.show()

if __name__ == '__main__':
    main()