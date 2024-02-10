# ML_Recommender_Systems
Comparing machine learning methods in heterogeneous graph recommender systems

## Project Overview

This project compares two approaches in formulating a Graph Learning Recommender System to make predictions on unseen nodes in a heterogeneous graph and determines the superior method in an experiment on benchmark data. The first approach applies an extension of the GraphSAGE family of algorithms to make predictions based on low-dimensional link embeddings of the graph structure. The second approach uses a matrix of one-hot encoded features to create unique vectors for user-item combinations and then predicts user ratings as node attributes via a feed forward multilayer perceptron. Given that a multilayer perceptron with at least one hidden layer can universally approximate any measurable function, can an MLP regressor (with appropriate feature engineering) perform comparably to an implementation of the GraphSAGE family? Comparing these two models will illustrate which best recommends links between data points in a network.

## Data

This experiment was conducted on the MovieLens 100K dataset. This data comes from the GroupLens Research Group founded within the University of Minnesota. It contains ratings from users to movies on a 5-star rating for 100,000 interactions created by 943 users for 1,682 movies, collected between September 1997 and April 1998 by MovieLens, a movie recommendation service. Users were selected at random for inclusion but all had rated at least 20 movies and must have had complete demographic information available. Each user is represented by an id. All of the movies included had at least one rating provided, also represented by an id. Movies may come from any of the following genres: Action, Adventure, Animation, Children’s, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western, or no genre listed. The mean rating given by users was a 3.53 out of 5, with a median and mode of 4. The highest mean rating of any user was 4.87 (23 movies rated) and the lowest was 1.49 (435 movies rated). The distribution of the count of movies rated followed a power law, with the average user leaving 106 ratings (mean) but with a median of 20. The highest number of ratings left by any user was 737 and 36% of the users left more ratings than the mean average. <br>

The summary statistics for the list of movies were similar. The highest mean rating of any movie was 5 (with no more than 3 reviews for perfectly rated movies) and the lowest was 1 (with no more than 5 reviews for such films). This count distribution was likewise a power law, with the mean average number of ratings left was 59 (median 27 and mode of 1). The highest number of ratings for a single movie was 583 (with the lowest of 1), and 32% of movies had more ratings than the mean. Three datafiles were used to compile the training dataset. The first principal file is a dataframe of user-movie ratings, where each line of data represents one rating of one movie by one user, and has the format `userId,movieId,rating,timestamp`. Users and items are numbered consecutively from 1 and the data is randomly ordered. Ratings are made on a 5-star scale from 1 to 5. Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970. The second principal file is a dataframe of movie features, which includes each movie listed sequentially by its id, along with its title, release date, URL to its imdb page, and a vector encoding of which genres it can be categorized, corresponding to the genres listed previously in this description. The genre categorical variables were already transformed 
into one-hot encodings by the GroupLens team. The final principal file is a dataframe of user features including the user id (also listed sequentially), along with the user’s age, gender, occupation, and zip code. Ages range from 7 to 73 and possible occupations may be administrator, artist, doctor, educator, engineer, entertainment, executive, healthcare, homemaker, lawyer, librarian, marketing, none, other, programmer, retired, salesman, scientist, student, technician, or writer.

## Findings

All the MLP regressor iterations outperformed the HinSAGE experiment. The worst performing variation (single layer, node attributes excluded, training partition of 30%) produced an RMSE 6.5% better than the HinSAGE results (0.968 vs 1.036). The best performing iteration (2 layers with node attributes and a training partition of 90%) produced an RMSE improvement of 9.4% (0.939 vs 1.036). However, said training partition is rather large in practice and would raise questions of over-fitting with poor generalization in production. As a result, the best comparision to the HinSAGE experiment is a training partition at 70% of the dataframe.

<div align="center">
    <img src="https://github.com/adamkicklighter/ML_Recommender_Systems/assets/97848631/2dbd4653-71e4-4299-9eaa-ef53678b1b26" alt="image">
</div>

<div>
    <img src="https://github.com/adamkicklighter/ML_Recommender_Systems/assets/97848631/5ebf4251-37b5-4598-859c-e82d7a7fc99f" alt="image" style="display: inline-block; margin-right: 10px;">
    <img src="https://github.com/adamkicklighter/ML_Recommender_Systems/assets/97848631/82fa4592-f69e-4a9e-9d64-0cacb6e3644a" alt="image" style="display: inline-block;">
</div>


## Conclusions

The MLP regressor produced superior results to the HinSAGE algorithm and would thus lead to a better Graph Learning based Recommender System. Given its lower root mean square error and mean absolute error across all architecture varieties, the perceptron algorithm was better able to predict the rating a user would assign to a movie. Implementing this capability into a GLRS would mean that the MLP regressor could predict the score a user would assign an unseen movie and the system could rank those unseen movies from high-to-low before presenting the user with the movies with the highest predicted ratings as recommendations for viewing. <br>
But what inferences could one draw about why this was the case? One possibility is that what the HinSAGE approach gains in flexibility and speed via node neighborhood sub-sampling and learning from aggregated graph features it losses in thoroughness. In other words, the MLP regressor is training on the entirety of the feature set (+2,700 dimensions in this work) while the HinSAGE method is optimizing a function that learns how to aggregate features from near-byneighborhoods. This additional layer of abstraction might compromise the model’s ability to competitively generalize. This is essentially a comparison of a low-dimensional approach against a high-dimensional approach. That said, is it wise to emphasize the MLP regressor’s superior evaluation metrics over the HinSAGE algorithm’s flexibility? Not necessarily. In the constraints of this experiment, it’s convenient to draw that conclusion, but in high-throughput industry applications, it could very well be the case that the HinSAGE approach produces better network adoption and user retention through runtime optimization and “good enough” recommendations. Similar to a PageRank study, is having the most precise rank of an item set more performant than simply having the correct rank in order? As is often the case, a real-world comparison of two GLRS built from these different algorithms may come down to product analysis, such as A/B testing, user retention, cost to acquire new users, time spent during sessions, and other financial or business KPIs, rather than statistical evaluation metrics. In industry, the best evaluation metric is sometimes humorously said to be revenue, and such an analysis may lead to different conclusions than the ones presented in this experiment.

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
