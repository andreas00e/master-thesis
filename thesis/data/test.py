import numpy as np 
import pandas as pd 
import seaborn as sns 
from scipy.stats import norm 
from matplotlib import pyplot as plt 

rng = np.random.RandomState(1)
x1 = rng.normal(0, 1, size=500)
x2 = rng.normal(3, 1.5, size=500)
x3 = rng.normal(-1, 0.5, size=500)

x1_out = np.quantile(a=x1, q=[0, 0.25, 0.5, 0.75, 1])


x = np.concatenate((x1, x2, x3), axis=0)
group = np.repeat(np.array(['G1', 'G2', 'G3']), repeats=[500, 500, 500], axis=0)
df = {'x': x, 'group': group}

df = pd.DataFrame(data=df)


plt.figure(figsize=(8, 5))
sns.histplot(x=x, hue=group, multiple='dodge', kde=True)

# y = norm.pdf(x=x1_out, loc=np.mean(x1), scale=np.std(x1))
# plt.plot(y)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Side-by-Side histogram with Seaborn')
plt.grid()
plt.legend()

plt.savefig('seaborn.png')
plt.show()


