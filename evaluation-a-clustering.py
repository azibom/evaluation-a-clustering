# import requirements
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
iris = load_iris()
# now we calculate the inertia for 15 diffrent "n_clusters" to find the best one 
inertia_list = []
for k in np.arange(1, 15):
 kmn = KMeans(n_clusters=k)
 kmn.fit(iris.data)
 inertia_list.append(kmn.inertia_)
# draw the chart
plt.plot(np.arange(1,15),inertia_list,'ro-')
plt.xlabel('number of clusters')
plt.ylabel('Inertia')
plt.show()
