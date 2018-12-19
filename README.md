# evaluation-a-clustering
evaluation a clustering

inertia :distance from each sample to centroids of its cluster or how spread out the clusters.
(Lower is better)

we evaluate our clustring with see the process of the inersia chart
now it is the time to have the code

```python
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
```

if you look at the chart you can see, it is descending chart and maybe you think it is good to set cluster as large number but in this case, you don't consider the complexity of our model, it is very import to make a simple model 
so you should choose the amount which is logical for our goal(in this case it is three :star::star::star:)

I hope this article will be useful to you.
