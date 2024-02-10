# ML_Recommender_Systems
Comparing machine learning methods in heterogeneous graph recommender systems

## Project Overview

This project compares two approaches in formulating a Graph Learning Recommender System to make predictions on unseen nodes in a heterogeneous graph and determine the superior method in an experiment on benchmark data. The first approach applies an extension of the GraphSAGE family of algorithms to make predictions based on low-dimensional link embeddings of the graph structure. The second approach uses a matrix of one-hot encoded features to create unique vectors for user-item combinations and then predicts user ratings as node attributes via a feed forward multilayer perceptron. Given that a multilayer perceptron with at least one hidden layer can universally approximate any measurable function, can an MLP regressor (with appropriate feature engineering) perform comparably to an implementation of the GraphSAGE family? Comparing these two models will illustrate which best solves the simultaneous issue of handling dynamically evolving large graphs and handling cold-start data points in a network.

## Findings

All the MLP regressor iterations outperformed the HinSAGE experiment. The worst performing variation (single layer, node attributes excluded, training partition of 30%) produced an RMSE 6.5% better than the HinSAGE results (0.968 vs 1.036). The best performing iteration (2 layers with node attributes and a training partition of 90%) produced an RMSE improvement of 9.4% (0.939 vs 1.036). However, said training partition is rather large in practice and would raise questions of over-fitting with poor generalization in production. As a result, the best comparision to the HinSAGE experiment is a training partition at 70% of the dataframe.

<div align="center">
    <img src="https://github.com/adamkicklighter/ML_Recommender_Systems/assets/97848631/2dbd4653-71e4-4299-9eaa-ef53678b1b26" alt="image">
</div>

<div>
    <img src="https://github.com/adamkicklighter/ML_Recommender_Systems/assets/97848631/5ebf4251-37b5-4598-859c-e82d7a7fc99f" alt="image" style="display: inline-block; margin-right: 10px;">
    <img src="https://github.com/adamkicklighter/ML_Recommender_Systems/assets/97848631/82fa4592-f69e-4a9e-9d64-0cacb6e3644a" alt="image" style="display: inline-block;">
</div>

## References

- [Wang et al., 2021] Shoujin Wang, Liang Hu, Yan Wang, et al. Graph Learning based Recommender Systems: A 
Review. arXiv preprint arXiv:2105.06339v1 [cs.IR] 13 May 2021
- [Berg, et al., 2017] Rianne van den Berg, Thomas N. Kipf, and Max Welling. Graph Convolutional Matrix 
Completion. arXiv preprint arXiv:1706.02263v2 [stat.ML] 25 Oct 2017.
- [Hamilton et al., 2018]. William L. Hamilton, Rex Ying, and Jure Leskovec. Inductive Representation Learning on 
Large Graphs. arXiv preprint arXiv:1706.02216v4 [cs.SI] 10 Sep 2018.
- [Maksimov et al., 2020] Ivan Maksimov, Rodrigo Rivera-Castro, and Evgeny Burnaev. Addressing Cold Start in 
Recommender Systems with Hierarchical Graph Neural Networks. arXiv preprint arXiv:2009.03455v2 [cs.LG] 1 Dec 2020.
- [James Currier et al., 2022] James Currier and the NFX Team. The Network Effects Bible. 
https://www.nfx.com/post/network-effects-bible
- Russell Reed, Robert J Marks II. Neural Smithing: Supervised Learning in Feedforward Artificial Neural Networks. 
Bradford Books. 1999.
- Link Prediction with Heterogenous GraphSAGE (HinSAGE). https://stellargraph.readthedocs.io/en/stable/demos/linkprediction/hinsage-link-prediction.html
- F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions 
on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages.
DOI=http://dx.doi.org/10.1145/2827872
- MovieLens 100K READ ME. https://files.grouplens.org/datasets/movielens/ml-100k-README.txt
- StellarGraph Documentation. Heterogenous GraphSAGE (HinSAGE). 
https://stellargraph.readthedocs.io/en/stable/hinsage.html
- MLPRegressor Documentation. https://scikitlearn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
