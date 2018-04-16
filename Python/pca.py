import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
#%matplotlib inline

data = pd.read_csv('features.csv')
X = data.values[:, 3:]

#print(X)
#exit()
X = scale(X)

pca = PCA(n_components=577)

#print(len(X[0]))

pca.fit(X)

#82 pca components explain 99% of the variance in our data when scaled
#the amount of variance that each PCA explains
var = pca.explained_variance_ratio_

#cumulative variance explains
var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=7)*100)
print(var)

#X1 = data.values[:, 0:85]
#X1[:, 3:] = pca.fit_transform(X)

#X1[:, 3:] = 

#X2 = data.values
#X2[:, 3:] = X

#print(X1[0])
df = pd.DataFrame(pca.fit_transform(X))
#df2 = pd.DataFrame(X2)
df.to_csv('new_pca_data.csv')
#df2.to_csv('new_scaled_data.csv')
plt.plot(var)
plt.ylabel('Cumulative Variance Explained')
plt.xlabel('Number of PCAs')
plt.show()


