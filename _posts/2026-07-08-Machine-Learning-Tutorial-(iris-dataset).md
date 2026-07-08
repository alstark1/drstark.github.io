---
title: Machine Learning Tutorial (iris dataset)
date: 2026-07-08 16:00:00 -0400
categories: [ML,Scikit-learn]
tags: [python,classification]
math: true
---

# Introduction to Machine Learning
## Information About This Post
In this post, we will look at different Machine Learning (ML) techniques applied to the iris dataset using Scikit-learn.
I have made a Jupyter notebook for the reader to follow along with, and I have not set a specific seed for random number generation, so results may vary. 
I have allowed the user to select how much printout they desire with the prl variable.
The intention is for 0 to print no results as text, 1 to print information which I consider to be important to the post, with most of the checks left out, and 2 to print everything.
The notebook can be [downloaded]({{ '/assets/notebooks/iris.ipynb' | relative_url }}) or used within a [browser]({{ '/assets/webhost_nb/jupyterlite/lab/index.html?path=iris.ipynb' | relative_url }}){:target="_blank" rel="noopener noreferrer"}.
As a rough overview, the topics covered here include principal component analysis (PCA), K-nearest neighbors (KNN), K-means clustering (KMC), and support vector machines (SVM), as well as concepts related to analysis within these methods.
Since this is rather involved, feel free to use the links on the right side of the page to navigate to a specific section/method you are interested in; the notebook also contains headers with the same subsections.

## 1. Preparing Data

To start, we can load the data directly from Scikit-learn by running the first cell:
```python
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

prl = 2
iris = load_iris()
```
Feel free to set your desired print level here too. The next cell prints the data so you can get an idea of how it is structured:
```python
iris
```
The iris dataset contains 150 samples of 3 iris species (50 samples per species) and includes information quantifying sepal length, sepal width, petal length, and petal width.
Along with data that classifies which species it belongs to.
Once we understand the data structure, we assign variables to the information that will be useful to us:
```python
x = iris.data
y = iris.target
plt_label = iris.feature_names
if(prl > 1):
    print("Data:", x)
    print("Target:", y)
    print("Feature Names:", iris.feature_names)
    print("Target Names:", iris.target_names)
```
The next 4 cells each produce a 3-dimensional plot with different combinations of the 4 independent variables.
A general trend emerges across all of these plots: the setosa species is well separated from versicolor and virginica.
The iris dataset is a great example problem, as the methods explored here can be applied to well-separated classes (setosa and versicolor) and to classes which are somewhat mixed (versicolor and virginica), but not as messy as most real-world problems.
This visualization is a luxury we are afforded here, as there are 4 independent variables, and only 2 are necessary to retain most of the information, which will become clear in the principal component analysis (PCA).

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
  <img src="/assets/img/2026-07-08-website_post/irisdata_plot1_light.png" alt="Plot 1" class="light">
  <img src="/assets/img/2026-07-08-website_post/irisdata_plot1_dark.png" alt="Plot 1" class="dark">
  <img src="/assets/img/2026-07-08-website_post/irisdata_plot2_light.png" alt="Plot 2" class="light">
  <img src="/assets/img/2026-07-08-website_post/irisdata_plot2_dark.png" alt="Plot 2" class="dark">
  <img src="/assets/img/2026-07-08-website_post/irisdata_plot3_light.png" alt="Plot 3" class="light">
  <img src="/assets/img/2026-07-08-website_post/irisdata_plot3_dark.png" alt="Plot 3" class="dark">
  <img src="/assets/img/2026-07-08-website_post/irisdata_plot4_light.png" alt="Plot 4" class="light">
  <img src="/assets/img/2026-07-08-website_post/irisdata_plot4_dark.png" alt="Plot 4" class="dark">
</div>

While all of the methods here can be used for classification--the species of iris in this case--they are a mix of supervised and unsupervised learning.
Supervised learning requires both inputs and outputs for training.
In the iris example, it would require providing the sepal length, sepal width, petal length, petal width, and species (the output) as inputs.
Unsupervised learning requires only input data for training and will sort the data into classes (or predict numerical values via regression) using an algorithm.
It is useful to set aside data for after training so one can test the machine with data it has never "seen" before; Scikit-learn has exactly this functionality in the form of the train_test_split function:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,stratify=y)
scaler = StandardScaler()
x_train_scale = scaler.fit_transform(x_train)
x_test_scale = scaler.transform(x_test)
```
The test_size option can be an integer specifying the number of data points to include in the test set or a decimal indicating the ratio of the train and test sets; stratify will ensure that roughly equal ratios of setosa, versicolor, and virginica are in the train and test sets.
If it were truly a random split, the train set could contain almost exclusively 2 species, and the third would not be trained on.
The data is also scaled using the StandardScaler function so that all data is scaled to have a mean of 0 and a standard deviation of 1.
The data is scaled so that no single feature is favored over others due to how we choose to measure the inputs.
In the case of the iris dataset, you can find examples where people do not scale the data and the results are still satisfactory; however, it is best practice to always scale your data.
Also, it is important to fit only the training data when scaling; if one were to both fit (find the mean and standard deviation) and then transform the test data as well, transformations would be inconsistent between sets.

## 2. Principal Component Analysis (PCA)

The idea behind PCA is to determine how much variance (equation 1) each of the independent variables contributes in a different basis of components, which is composed of linear combinations of the original independent variables.
PCA allows one to combine and separate certain features into principal components, usually with the goal of reducing the dimensionality of independent variables.
$$
\begin{equation}
Var(X)=E[(X-\mu)^2]
\label{eq:2.1}
\end{equation}
$$
In other words, how correlated are the independent variables, and are some mapping to the same behavior?
An example of this would be looking at volatility in the stock market; one could analyze the volatility of NVDA, TSMC, AMD, etc., since these stocks are all tech stocks, their volatility is somewhat correlated.
PCA might compress these features into "volatility of tech stocks".
However, it is not always obvious what each principal component represents; we can gain more insight via a loading plot, which will be seen near the end of this section.
PCA is achieved via singular value decomposition (SVD), where the singular values $\sigma$ are obtained, and $Variance \propto \sigma^2$.
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.999999)
```
The PCA function used here can specify how many components are desired based on either the retention of variance (from 0.0 to 1.0) or an integer indicating the number of components.
In this case, I retained almost all of the variance, so we can produce a scree plot which gives us an idea of how many dimensions are sufficient to describe our independent variables.
```python
if(prl > 0):
    print("Variance ratio for each component:",pca.explained_variance_ratio_)
    print("Singular Values:",pca.singular_values_)
```

![PCA Scree Plot-light](/assets/img/2026-07-08-website_post/pca_scree_light.png){: .light}
![PCA Scree Plot-dark](/assets/img/2026-07-08-website_post/pca_scree_dark.png){: .dark}

A scree plot shows the variance associated with each principal component; the cell preceding to the plot prints the ratio of the variance from each component and the singular values used to obtain the variance.
It is important to think about how much variance in sufficient to retain for your purposes, it will be left to the reader to experiment with this on their own; some ranges are 95-99% for applications with require minimal loss of information (financial predictions), 85-95% for applications where some information can afford to be lost (scientific applications where data can be overfit).
With visualization in mind, I created a second PCA variable, pca2, to retain only the first 2 components.
The pca2 variable will be useful later when using the support vector machine (SVM), as it will allow for visualization on a 2-dimensional plot.
A loading plot was also produced to see how each of the original 4 variables corresponds to the principal components.

![PCA Loading Plot-light](/assets/img/2026-07-08-website_post/pca_loading_light.png){: .light }
![PCA Loading Plot-dark](/assets/img/2026-07-08-website_post/pca_loading_dark.png){: .dark }

The loading plot indicates that the petal length and width are highly correlated and largely contribute to the (first) principal component with the highest variance.
The sepal width is mostly related to the (second) principal component with the second highest variance; there is also a slight negative correlation relating to the first principal component.
Finally, sepal length seems to contribute roughly equally to both principal components in a positively.

## 3. K Nearest Neighbors (KNN)

The idea behind K-nearest neighbors is quite simple; if we visualize our data on a grid, as we did when preparing the iris data, we can see that distinct regions relating to each class exist.
If we introduce a new data point, we can select $K$ neighbors to predict its species.
Say the point is between the setosa and versicolor clusters and we choose $K=3$ (where K is the number of neighbors), the new point will be classified as whatever the majority of the nearest points are classified as.
If, among the 3 nearest points, 2 are setosa and 1 is versicolor, the new point will be classified as setosa.
Thinking further about this, what if we had $K=4$ and we get a tie?
There are a few solutions; we can increase the number of neighboring points considered by 1, which will break the tie, or we can randomly pick a category from the contributing points.
Alternatively, to reduce the likelihood of ties, we can weight the importance of the points by how close they are with some function.
Taking a look at the iris dataset as an example, first KNeighborsClassifier is loaded, initially we will take a look at $K=3$:
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
```
```python
knn.fit(x_train_scale,y_train)
```
I next predict a point with arbitrary variable values chosen:
```python
x_new = np.array([[1.5, 1.5, 1.2, 0.2]])
prediction_new = knn.predict(x_new)
if(prl > 0):
    print(prediction_new)
```

![KNN Visual Plot-light](/assets/img/2026-07-08-website_post/knn_visual_light.png){: .light }
![KNN Visual Plot-dark](/assets/img/2026-07-08-website_post/knn_visual_dark.png){: .dark }

The red point in the above figure was created in the preceding cell.
It is clear from the figure that the red point has many yellow (virginica) neighbors; as such, the model should predict virginica with a reasonable number of neighbors.
We can also consider an alternative case where the red point now lands between versicolor and virginica; the prediction is then more sensitive to the number of neighbors used.
It can be concluded then, that for points which lie deeper in the region of a given class, the more consistent the prediction will be, this is desirable behavior.
We can think about this with chairs as an example; a typical dining room chair will be deep within the chair class, and an apple will fall outside of it.
If we assume an object on the edge of the chair class, such as a bench, the prediction will be less certain and may change depending on how many neighbors are considered.
We can now proceed to classify our test set.

```python
predictions_test = knn.predict(x_test_scale)
names = iris.target_names
if(prl > 0):
    print("Predictions Text:",names[predictions_test])
    print("Predictions Numerical:",predictions_test)
    print("True Targets Numerical:",y_test)
```
The prediction and accuracy are printed for your convenience, after which an array of KNN models is created, each with a different number of neighbors from 1 to the number of data points in the training set (112).
```python
knum = len(x_train_scale)
knn_m = []
for i in range(knum):
    knn_m.append(KNeighborsClassifier(n_neighbors=i+1))
```
A simple analysis of the error (ratio of incorrect predictions to total predictions) shows generally low error for $K \lesssim 20$.

![KNN Error Plot-light](/assets/img/2026-07-08-website_post/knn_error_light.png){: .light }
![KNN Error Plot-dark](/assets/img/2026-07-08-website_post/knn_error_dark.png){: .dark }

## 4. K-Means Clustering (KMC)

K-means clustering is an unsupervised learning technique; the idea is to define a specified number of centroids to represent different clusters of data (we can set the number of centroids to 3 since we have 3 species) and minimize the distances between points in a cluster and their assigned centroid.
A cluster here is defined as the ser of points closest to a centroid.
The question is then how we determine where the centers go.
We need to know where a center is to calculate the distance from a point to the center, but we also need to know which points are associated with a given center to calculate the distance.
The answer is that we solve this self-consistently using some initial assumption of where the centers are located.
The procedure goes as follows:

1. Start by defining a guess for the location of the centroids; this can be random vectos with proper scaling, or you can choose randomly from your data points for initial guesses.

2. Calculate the distance from each point to each centroid and determine which point is associated with a given centroid.
We will assume Euclidean distance is chosen (in principle, any distance metric can be chosen), which would use:
$$
\begin{equation}
distance = \sqrt{\sum_{m=1}^{M} {(x_m^{(n)}-c_m^{(k)})^2}}
\label{eq:4.1}
\end{equation}
$$
The point would be associated with whichever center it is closest to; in other words, the distance function is minimized by choosing the $k^{th}$ center with the shortest distance.

3. Calculate the new position of the centroid by averaging over the positions of all of the points assigned to the centroid in the current position. Repeat steps 2 and 3 until self-consistency is attained.

This procedure can get stuck in local minima pretty easily; another way of looking at this is that the final clusterings are heavily dependent on the initial guess for the centroid positions.
One way to rectify this is to run the algorithm many times and see which clusterings it tends toward most; this is also generally a good idea with any ML method.
As was mentioned at the beginning, this code does not set a seed value, so you should be able to test this pretty easily by resetting the notebook's kernel and analyzing the differences in the plots you see.
With the iris example, we first import Kmeans from Scikit-learn and fit the model with n_clusters=3 and the n_init='auto', which controls how many times the algorithm is restarted with different seeds to keep the best quality run, when set to auto it depends on the method used.
In the default case, the method is set to k-means++, where the seeds (initial centroids) are spread out probabilistically, and n_init is set to 1.
```python
from sklearn.cluster import KMeans

scaler_kmc = StandardScaler()
kmeans = KMeans(n_clusters=3,n_init='auto')
x_scale = scaler_kmc.fit_transform(x)
cluster_labels = kmeans.fit_predict(x_scale)

centroids = kmeans.cluster_centers_
```
The resulting solutions for the centroids are plotted and seem to work quite well.
From a purely qualitative view, it seems to have recovered each of the clusters for the species.

![KMC 3Cluster Plot-light](/assets/img/2026-07-08-website_post/kmc_3cl_light.png){: .light }
![KMC 3Cluster Plot-dark](/assets/img/2026-07-08-website_post/kmc_3cl_dark.png){: .dark }

To see how well this worked, the next cell shows a comparison of accuracy (predictions correct over total predictions).
It is not trivial to obtain this for KMC, since, being unsupervised, the cluster labels can be permuted relative to the outputs we are already familiar with.
For example, the mapping could be 0:versicolor, 1:virginica, 2:setosa instead of how the initial clusters are defined as 0:setosa, 1:versicolor, 2:virginica. 
The following cell uses the Hungarian algorithm to reorder the species to match our initial data, so we can calculate the accuracy and visualize which points were incorrectly predicted.
The cluster quality is also calculated using the adjusted_rand_score function, which compares pairs of points and forms a confusion matrix (true positive, true negative, false positive, false negative) based on agreement.
It uses these classifications to calculate the quality of the clustering while correcting for random chance ([more information](https://link.springer.com/article/10.1007/s00357-022-09413-z){:target="_blank" rel="noopener noreferrer"}).
Scores range from -0.5 to 1.0, where 0.0 indicates the clustering is not better than random chance, a negative score indicates clustering is worse than random chance, and positive scores indicate better than random chance.
```python
if(prl > 0):
    print(cluster_labels)
    print(y)

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

y_true = np.array(y)
y_pred = np.array(cluster_labels)
n_classes = max(y_true.max(), y_pred.max()) + 1

cm = confusion_matrix(y_pred, y_true, labels=range(n_classes))
row_ind, col_ind = linear_sum_assignment(-cm)

mapping = dict(zip(row_ind, col_ind))
cluster_labels_matched = np.array([mapping[label] for label in y_pred])

if(prl > 1):
    print(mapping)
if(prl > 0):
    print(cluster_labels_matched)

num_correct_arr = y == cluster_labels_matched

if(prl > 1):
    print(num_correct_arr)
```

![KMC Wrong Pred Plot-light](/assets/img/2026-07-08-website_post/kmc_wrong_pred_light.png){: .light }
![KMC Wrong Pred Plot-dark](/assets/img/2026-07-08-website_post/kmc_wrong_pred_dark.png){: .dark }

Similar to the KNN method, an array with different numbers of clusters is constructed, and each of the resulting cluster discretizations is plotted.
The inertia is obtained to make another scree plot; instead of analyzing principal components we are instead looking for the number of clusters which reduce the inertia significantly.
Inertia is calculated as a function of distance from the centroid for each point; thus, if we create more centroids (increase K), say in the limit where we have 1 centroid for each point, the inertia will be reduced to 0.
$$
\begin{equation}
I=\sum_{j=1}^{K} {\sum_{i=1}^{N_{points}} {\lvert x_i-\mu_j\rvert^2}}
\label{eq:4.2}
\end{equation}
$$
Where K is the number of clusters and N_points is the number of points in a given cluster.
It is important to look for an elbow in the scree plot; this will inform your choice of cluster number and help avoid overfitting (the elbow is around 2 or 3 clusters here).

![KMC Scree Plot-light](/assets/img/2026-07-08-website_post/kmc_scree_light.png){: .light }
![KMC Scree Plot-dark](/assets/img/2026-07-08-website_post/kmc_scree_dark.png){: .dark }

The above figure is consistent with the fact that there are 3 clusters in the iris dataset.

## 5. Support Vector Machine (SVM)

The final method we will look at is the support vector machine (SVM), which is a supervised learning technique. 
This method works by predicting (in this case) a class based on which side of a hyperplane the vector formed by the new data lands.
It is easiest to think about this in 2 dimensions, if we only include the setosa and versicolor points and plot petal length vs. petal width (you can reference the 3-dimensional plots from the preparing data section and compress them along 1 axis).
We can imagine plotting a line between the separated sets of data points; if a new data point lies on one side or the other, it will be predicted to belong to the class associated with that side of the line (setosa or versicolor).
How will we choose this line though?
We define the line so that the margin from the line is maximized; in other words, if we have some margin (represented by the purple and yellow lines below), then our line is chosen to be equidistant from the yellow and purple lines.

![SVM Linear Plot-light](/assets/img/2026-07-08-website_post/svm_linear_light.png){: .light }
![SVM Linear Plot-dark](/assets/img/2026-07-08-website_post/svm_linear_dark.png){: .dark }

When we attempt to predict a new point, we can also have some confidence in the prediction; if the magnitude is greater than 1, it comfortably falls within a cluster; if, however, the magnitude is less than 1, we can have more skepticism.
The equations which govern the hyperplane are given by:
$$
\begin{equation}
\boldsymbol{\theta}^T\mathbf{x}+\theta_0 = 0
\label{eq:5.1}
\end{equation}
$$
Where we have to find the vector $\boldsymbol{\theta}$ and the scalar $\theta_0$ to predict the class.
$\boldsymbol{\theta}$ is a vector which is orthogonal to the hyperplane and thus defines its position along with the scalar $\theta_0$.
We can define the decision function (equation 4) as being +1 at the yellow line and -1 at the purple line.
We can then rewrite our decision function to define the margins:
$$
\begin{equation}
\boldsymbol{\theta}^T\mathbf{x}+\theta_0 = \pm 1
\label{eq:5.2}
\end{equation}
$$
If we subtract the negative expression from the positive, it results in:
$$
\begin{equation}
\boldsymbol{\theta}^T(\mathbf{x}^+-\mathbf{x}^-) = 2
\label{eq:5.3}
\end{equation}
$$
For a given data point n, since we know setosa is less than -1, equation 6 can be constrained to:
$$
\begin{equation}
\boldsymbol{\theta}^T\mathbf{x}^{(n)}+\theta_0 \leq -1
\label{eq:5.4}
\end{equation}
$$
Since we know versicolor is greater than 1, equation 6 can be constrained to:
$$
\begin{equation}
\boldsymbol{\theta}^T\mathbf{x}^{(n)}+\theta_0 \geq 1
\label{eq:5.5}
\end{equation}
$$
We can apply a variable $y^{(n)}$ to this equation, with $y^{(n)}$ taking values 1 or -1, allowing this inequality to be satisfied for a given $y$ output (in the iris case, the species).
$$
\begin{equation}
y^{(n)}(\boldsymbol{\theta}^T\mathbf{x}^{(n)}+\theta_0)-1 \geq 0
\label{eq:5.6}
\end{equation}
$$
From equation 6, it is obvious that the margin width is:
$$
\begin{equation}
margin\ width = \frac{2}{\lvert\boldsymbol{\theta}\rvert}
\label{eq:5.7}
\end{equation}
$$
We want to place the hyperplane equidistant from the two margins, as this reduces the likelihood of misclassification compared to other positions where the hyperplane could be placed.
If we were to place the hyperplane closer to the versicolor species, we would be more likely to classify points as setosa without sufficient justification.
Since $\boldsymbol{\theta}$ is in the denominator, minimizing $\boldsymbol{\theta}$ will maximize the margin.
Another way to write the minimization of $\boldsymbol{\theta}$ explicitly is:
$$
\begin{equation}
\min_{\boldsymbol{\theta},\theta_0}\frac{1}{2}\lvert\boldsymbol{\theta}\rvert^2
\label{eq:5.8}
\end{equation}
$$
Solving for equation 11 will give a satisfactory hyperplane.
Alternatively, to get more freedom in the dimensionality in which we can construct the hyperplane, we can obtain the dual of this minimization problem using Lagrange multipliers. 
$$
\begin{equation}
L=\frac{1}{2}\lvert\boldsymbol{\theta}\rvert^2-\sum_{n=1}^{N} {\alpha_n}(y^{(n)}(\boldsymbol{\theta}^T\mathbf{x}^{(n)}+\theta_0)-1)
\label{eq:5.9}
\end{equation}
$$
The $\alpha$s here are the Lagrange multipliers and are constrained by $\alpha\geq0$.
You will notice that the first part of the right-hand side of equation 12 is the margin width term, and the second part is the constraint of the margins being $\pm 1$ from the hyperplane. 
To solve for the critical points, we differentiate with respect to $\boldsymbol{\theta}$ and $\theta_0$ resulting in:
$$
\begin{equation}
\sum_{n=1}^{N} {\alpha_n y^{(n)}} = 0
\label{eq:5.10}
\end{equation}
$$
and
$$
\begin{equation}
\boldsymbol{\theta}-\sum_{n=1}^{N} {\alpha_n y^{(n)} \mathbf{x}^{(n)}} = 0 \rightarrow \boldsymbol{\theta} = \sum_{n=1}^{N} {\alpha_n y^{(n)} \mathbf{x}^{(n)}}
\label{eq:5.11}
\end{equation}
$$
To make the algebra explicit, I will first rewrite equation 12 as:
$$
\begin{equation}
L=\frac{1}{2}\lvert\boldsymbol{\theta}\rvert^2-\sum_{n=1}^{N} {\alpha_n y^{(n)} \boldsymbol{\theta}^T \mathbf{x}^{(n)} + \alpha_n y^{(n)} \theta_0 - \alpha_n}
\label{eq:5.12}
\end{equation}
$$
If we substitute equation 14 into equation 15:
$$
\begin{equation}
L=\frac{1}{2}\sum_{i=1}^{N} {\sum_{j=1}^{N} {\alpha_i\alpha_j y^{(i)}y^{(j)}\mathbf{x}^{(i)T}\mathbf{x}^{(j)}}} - \sum_{i=1}^{N} {\sum_{j=1}^{N} {\alpha_i\alpha_j y^{(i)}y^{(j)}\mathbf{x}^{(i)T}\mathbf{x}^{(j)}}} - \sum_{n=1}^{N} {\alpha_n y^{(n)} \theta_0 - \alpha_n}
\label{eq:5.13}
\end{equation}
$$
Then substitute equation 13 into equation 16 and simplify:
$$
\begin{equation}
L=\sum_{n=1}^{N} {\alpha_n} - \frac{1}{2}\sum_{i=1}^{N} {\sum_{j=1}^{N} {\alpha_i\alpha_j y^{(i)}y^{(j)}\mathbf{x}^{(i)T}\mathbf{x}^{(j)}}}
\label{eq:5.14}
\end{equation}
$$
Equation 17 is the dual form where we can maximize the set of $\alpha$s by considering pairs, which is equivalent to minimizing equation 11.
The $\alpha$s will mostly be 0, as only the points which define the margin will have $\alpha > 0$.
In the iris data, it looks like 2 setosa points and 1 versicolor point define the margin.
This is where the method gets its name; the points with $\alpha > 0$ are the support vectors that define the margin; even if all of the other data points were removed, the partitioning would not change.
A problem appears though, for clusters which overlap.
Let us consider the versicolor and virginica species, for which there is no hyperplane that satisfies the constraints we have outlined.
We can resolve these issues by using soft margins and a kernel.
The previous formulation uses what are called hard margins, where it is strictly defined by equations 7 and 8; instead, with soft margins, we use a function which allows us to define a hyperplane along clusters which are not linearly separable by penalizing points which are on the opposite side of the hyperplane compared to most of the other points in the cluster:
$$
\begin{equation}
J=\frac{1}{N}\sum_{n=1}^{N} {\max(1-y^{(n)}(\boldsymbol{\theta}^T\mathbf{x}^{(n)} + \theta_0),0)} + \lambda\lvert\boldsymbol{\theta}\rvert^2
\label{eq:5.15}
\end{equation}
$$
$\lambda$ balances maximizing the margin and penalizing points which cross the hyperplane into the opposite classification.
The max in the first term of the function allows points which lie far on the correct side of the margin to contribute nothing to the loss function.
On the correct side, $y^{(n)}(\boldsymbol{\theta}^T\mathbf{x}^{(n)} + \theta_0) \geq 1$, causing the max to always evaluate to 0.
Meanwhile, if $y^{(n)}(\boldsymbol{\theta}^T\mathbf{x}^{(n)} + \theta_0) < 1$, then the function grows linearly, correcting the margin towards the point in/near the other cluster.
An equivalent formulation of this for the primal is:
$$
\begin{equation}
J=\frac{1}{2}\lvert\boldsymbol{\theta}\rvert^2 + C\sum_{n} {\zeta_n}
\label{eq:5.16}
\end{equation}
$$  
Where the cost function is minimized over $\boldsymbol{\theta}$ and $\theta_0$, and $\zeta_n$ adds slack to the strict margin at $\pm 1$.
Scikit-learn uses equation 19, so we will compare how accuracy changes with $C$, $C$ and $\lambda$ are inversely proportional.
Another useful tool to further separate data is a kernel function; this neat trick allows the construction of a hyperplane in arbitrarily high-dimensional space, still using only the variables in $\mathbf{x}$.
If our $\mathbf{x}$ is 2-dimensional, we can imagine expanding the variables in $\mathbf{x}$ through the transformation:
$$
\begin{equation}
(x_1,x_2) \rightarrow (x_1,x_2,x_1^2+x_2^2)
\label{eq:5.17}
\end{equation}
$$
If these points were on a 2-dimensional plane with coordinates x,y, and we could not linearly separate them, we can now imagine transforming them and adding a z-axis where the points from both classes drift apart and are linearly separable along the z-axis.
It might seem that dimensionality would become a problem; however, in the dual formulation we take $\mathbf{x}^{(i)T}\mathbf{x}^{(j)}$, which reduces to a scalar.
If we denote a transformation as $\varphi(\mathbf{x})$ then the new product becomes:
$$
\begin{equation}
\varphi(\mathbf{x}^{(i)})^T\varphi(\mathbf{x}^{(j)})
\label{eq:5.18}
\end{equation}
$$
A concrete example of this is a polynomial function where the vector $\varphi(\mathbf{x})$ becomes:
$$
\begin{equation}
(b,\sqrt{2ab}x_1,\sqrt{2ab}x_2,ax_1^2,ax_2^2,\sqrt{2}ax_1x_2)
\label{eq:5.19}
\end{equation}
$$
Resulting in a product:
$$
\begin{gather}
\varphi(\mathbf{x}^{(i)})^T\varphi(\mathbf{x}^{(j)}) = \nonumber \\[8pt]
b^2 + 2abx_1^{(i)}x_1^{(j)} + 2abx_2^{(i)}x_2^{(j)} + a^2x_1^{(i)2}x_1^{(j)2} + a^2x_2^{(i)2}x_2^{(j)2} + 2a^2x_1^{(i)}x_2^{(i)}x_1^{(j)}x_2^{(j)} \nonumber \\[8pt]
= (a\mathbf{x}^{(i)T}\mathbf{x}^{(j)} + b)^2
\label{eq:5.20}
\end{gather}
$$
So in equation 17 our kernel function was:
$$
\begin{equation}
K(\mathbf{x}^{(i)},\mathbf{x}^{(j)}) = \mathbf{x}^{(i)T}\mathbf{x}^{(j)}
\label{eq:5.21}
\end{equation}
$$
We have now transformed it to be:
$$
\begin{equation}
K(\mathbf{x}^{(i)},\mathbf{x}^{(j)}) = (a\mathbf{x}^{(i)T}\mathbf{x}^{(j)} + b)^2
\label{eq:5.22}
\end{equation}
$$
We can now express equation 17 generally as:
$$
\begin{equation}
L=\sum_{n=1}^{N} {\alpha_n} - \frac{1}{2}\sum_{i=1}^{N} {\sum_{j=1}^{N} {\alpha_i\alpha_j y^{(i)}y^{(j)} K(\mathbf{x}^{(i)},\mathbf{x}^{(j)})}}
\label{eq:5.23}
\end{equation}
$$
The powerful result of the kernel is only possible because we express our data using an inner product; this allows us to separate the data using infinitely high dimensionality without dramatically increasing the computational cost.
With everything defined we can now look at this method applied to the iris dataset (apart from the sneak peek provided earlier).
In the first cell, the support vector function is defined; it is called SVC because it is a support vector classifier.
```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

scaler_svm = StandardScaler()
svm_simple = SVC(kernel='linear')
svm = SVC(kernel='rbf',C=1.0,gamma='scale')
x_01 = x[:100]
y_01 = y[:100]
pca_01 = PCA(n_components=2)
if(prl > 1):
    print("x_01:",x_01,"y_01:",y_01,sep='\n')
x_01_train,x_01_test,y_01_train,y_01_test = train_test_split(x_01,y_01,test_size=0.25,stratify=y_01)
x_01_train_scale = scaler_svm.fit_transform(x_01_train)
if(prl > 1):
    print("x_01_train_scale:",x_01_train_scale,"x_01_test:",x_01_test,"y_01_train:",y_01_train,"y_01_test:",y_01_test,sep='\n')
x_01_train_scale_pca = pca_01.fit_transform(x_01_train_scale)
if(prl > 1):
    print("x_01_train_scale:",x_01_train_scale,sep='\n')
svm_simple.fit(x_01_train_scale_pca,y_01_train)
svm.fit(x_train_scale_pca2,y_train)
```
The svm_simple variable defines a simple linear function for the hyperplane; this is used to produce the first figure in this section.
The svm variable makes use of the radial basis function where the kernel is defined as $K(\mathbf{x}^{(i)},\mathbf{x}^{(j)}) = \exp(-\gamma\lvert\mathbf{x}^{(i)}-\mathbf{x}^{(j)}\rvert^2)$, $C$ is set to 1.0, and $\gamma$ is set to scale, which sets $\gamma = \frac{1}{n_{features}\cdot Var(\mathbf{x})}$.
For visualization purposes, only the first two principal components are used.

![SVM RBF Plot-light](/assets/img/2026-07-08-website_post/svm_rbf_light.png){: .light }
![SVM RBF Plot-dark](/assets/img/2026-07-08-website_post/svm_rbf_dark.png){: .dark }

Next, 3 arrays are constructed using the radial basis function, with all 4 independent variables being utilized; the first array is held at constant $\gamma = \text{'scale'}$.
The following two cells can be used as a general template for how all 3 plots were constructed.
```python
svm_c_arr = []
c_iter = 25
c_vals = []
for i in range(c_iter):
    ci = (i+1)
    c_vals.append(ci)
    svm_c_arr.append(SVC(kernel='rbf',C=ci,gamma='scale'))
if(prl > 1):
    print(svm_c_arr)
```
```python
predict_c_arr = []
accuracy_c = []
for i in range(c_iter):
    svm_c_arr[i].fit(x_train_scale,y_train)
    temp_arr_c = svm_c_arr[i].predict(x_test_scale)
    predict_c_arr.append(temp_arr_c)
    count_c = 0
    for j in range(len(y_test)):
        if(y_test[j] == temp_arr_c[j]):
            count_c += 1
    accuracy_c.append(count_c/len(y_test))
if(prl > 1):
    print(y_test)
    print(predict_c_arr)
    print(accuracy_c)

fig = plt.figure()

plt.plot(c_vals,accuracy_c)
plt.xlabel("C")
plt.ylabel("Accuracy")
```
The test set (x_test_scale) is used here to optimize the parameters $C$ and $\gamma$ and is therefore treated as a validation set.
Validation sets are a third partition of the data used to find the optimal hyperparameters for the model.
The test set is then used in the final stage, since using the same set for hyperparameter optimization and testing could yield an artificial improvement in accuracy.
It is important to save data for final testing which the model has not seen.
For SVM applied to the iris dataset, I will not compute accuracy since I did not generate a validation set and the number of data points is already small; here I will just show hyperparameter optimization.
The first plot shows results varying with $C$ from 1.0 to 25.0 in increments of 1.0 and $\gamma = \text{'scale'}$.

![SVM RBF gc Plot-light](/assets/img/2026-07-08-website_post/svm_rbf_gc_light.png){: .light }
![SVM RBF gc Plot-dark](/assets/img/2026-07-08-website_post/svm_rbf_gc_dark.png){: .dark }

The accuracy remains relatively consistent as $C$ varies; in principle, if $C$ were increased further, it would eventually overfit the data by giving cluster violations too much weight.
Since the iris dataset is well-behaved, it becomes more difficult to observe the loss of accuracy as the margin is shrunk with increasing $C$.
The second plot shows results varying with $\gamma$ from 0.0001 to 10.0 in increments of $10^i$ where $i=0,1,2,3,4,5,6$ and $C = 1.0$.

![SVM RBF cc Plot-light](/assets/img/2026-07-08-website_post/svm_rbf_cc_light.png){: .light }
![SVM RBF cc Plot-dark](/assets/img/2026-07-08-website_post/svm_rbf_cc_dark.png){: .dark }

Accuracy with $\gamma$ changes more significantly here since with smaller $\gamma$ each data point influences every other data point as the function becomes nearly linear (it becomes a constant 1.0 at $\gamma = 0.0$).
While a small $\gamma$ results in underfitting, a large $\gamma$ causes each point to have no influence on other points, which hinders the formation of clustered regions.
This is best illustrated with the following figure:

![SVM RBF gamma Plot-light](/assets/img/2026-07-08-website_post/rbf_gamma_light.png){: .light }
![SVM RBF gamma Plot-dark](/assets/img/2026-07-08-website_post/rbf_gamma_dark.png){: .dark }
 
The final array is a matrix composed of combinations of the previously screened $\gamma$ and $C$ parameters, which allows for the plotting of a surface to determine which combination of $\gamma$ and $C$ maximizes accuracy.

![SVM RBF gcv Plot-light](/assets/img/2026-07-08-website_post/svm_rbf_gcv_light.png){: .light }
![SVM RBF gcv Plot-dark](/assets/img/2026-07-08-website_post/svm_rbf_gcv_dark.png){: .dark }

This surface should show generally that $C$ only varies (qualitatively) at low $\gamma$, while a $\gamma$ of ~0.1 maximizes the accuracy for the iris dataset.
It is important to emphasize that no overall accuracy heuristic was obtained for the SVM method.
In principle, one can separate the iris data into 3 sets for training, validation, and testing and compare the accuracy of all the methods while tinkering with hyperparameters.
I leave this as an exercise to the reader, as the best way to get comfortable with these concepts and their impact on data is to experiment.

## Conclusion

We have learned about principal component analysis, K-nearest neighbors, K-means clustering, and support vector machines as they apply to classification methods.
I encourage the reader to experiment further with the notebook and read about the different options for the hyperparameters in the Scikit-learn documentation.
I will briefly discuss how to identify applications in which these methods will either thrive or fail.

KNN requires no training; it simply works based upon whatever data it has for reference.
The lack of training is also a downside, because you then have to store all that data; there is no inherent compression in the method.
KNN works well for things like recommendations, where people who watch a certain genre of show similar to yours on Netflix might predict shows you would enjoy.
It could also be that some clusters of show preferences are deeply mingled and noisy, which would degrade the performance of KNN.

KMC is the only unsupervised learning method we used here; thus, it is an obvious choice (among these 3 methods) for situations where outputs are not available.
It could be useful for grouping YouTube videos, stocks, etc., into categories based upon the input data.
On the other hand, these categories may not be obvious; determining the number of clusters to use can be difficult if there is no well-defined elbow in the scree plot. 

SVM can be useful for problems in which the number of features exceeds the number of data points.
An example of this could be in DNA: gene expression determines the type of tissue formed; to probe this, one might look at the activity of ~10,000 genes, and there may only be 100s of samples.
Neural networks would perform poorly as they would not have enough samples to constrain the equation with 10,000 unknowns.
SVM gets around this by using a margin to separate the data; it chooses the safest boundaries between classes by maximizing the margin.
That being said, for most problems today, which involve structured data and many samples–things like text or images–neural networks are very popular and perform well.

These methods provide the starting points from which other methods build and form the foundations of ML.
There are several other fundamental methods for the interested reader to look into as well, which may be covered in future posts, such as the naive Bayes classifier, self-organizing maps, and trees.
