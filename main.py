import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import axes3d
from sklearn.cluster import KMeans

path = '/Users/JohnsonJohnson/Downloads/point-cloud.csv'

# a: scatter the dimension reduced data
raw = pd.read_csv(path)
tmp = np.array(raw)
data = TSNE(n_components=3).fit_transform(tmp)

# Create plot
fig1 = plt.figure()

ax1 = fig1.gca(projection='3d')
ax1.scatter(data[:,0], data[:,1], data[:,2],color='red')
plt.title('Raw data')


# b: Automatically detect the separate sections of the scan
# and visualize them in separate 3D graphs. There should be 5
# distinct sections in total.

data_clustered = KMeans(n_clusters=5, random_state=666).fit_predict(data)
fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
ax2.scatter(data[:,0], data[:,1], data[:,2], c = data_clustered)
plt.title('Clustered data')



#Plot face of each section

fig3, axs = plt.subplots(5, 1)
for i in range(5):
    indices = np.where(data_clustered==i)
    x = np.array(data[np.array(indices),0])
    y = np.array(data[np.array(indices),1])
    # z = np.array(data[np.array(indice),2])
    axs[i].plot(x, y,'ro')
    # axs[i,0].plot(x, y,'r')

plt.suptitle('Face of each cluster')

#Clean up the noise of each section
for i in range(5):
    indices = np.where(data_clustered == i)
    current_cluster = data[indices, :]
    current_cluster = np.squeeze(current_cluster)
    if (np.size(current_cluster) == 3):
        continue
    pca = PCA(n_components=2)

    pca.fit(np.array(current_cluster))
    refined_cluster= pca.transform(current_cluster)

plt.show()






