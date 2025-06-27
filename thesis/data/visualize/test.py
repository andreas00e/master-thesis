import numpy as np 
from matplotlib import pyplot as plt 

def linear_anneal(max_kl_weight, step, x0):
    return min(max_kl_weight, step / x0)

def logistic_anneal(max_kl_weight, k, step, x0):
    return float(max_kl_weight / (1 + np.exp(-k * (step - x0))))

def main():   
    stop = 10_000
    x = np.arange(start=0, stop=stop)
    linear_y = np.array([linear_anneal(max_kl_weight=1, step=i, x0=5_000) for i in range(stop)])
    logistic_y = np.array([logistic_anneal(max_kl_weight=1, k=5e-3, step=i, x0=5_000) for i in range(stop)])


    fig = plt.figure(figsize=(10, 5))
    plt.plot(x, linear_y, color='c', ls='--', label='linear') 
    plt.plot(x, logistic_y, color='m', ls='--', label='logistic')
    plt.title('Monotonic KL Annealing Schedules')
    plt.xlabel('Time Steps')
    plt.ylabel('KL Penalty Term')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    
    fig.savefig(fname='thesis/data/visualize/figure.png')
    pass 

if __name__ == '__main__': 

    main() 
    
