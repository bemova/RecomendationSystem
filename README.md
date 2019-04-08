# RecomendationSystem

# README #

This README describes scripts and folder functionality.

### What is this repository for? ###

*  implementing collaborative filtering for [movilense data-set](https://grouplens.org/datasets/movielens/1m/) in spark and try to improve the accuracy with considering the movies' profile in CF algorithm.
*  implement Matrix Factorization with tensorflow
*  implement and useing a deep learning model for recomendation purpose with tensorflow and keras. 
*  implement Autorec paper which is a recomendation solution with autoencoder model with tensorflow.
*  implement linUCB and HybridLinUCB algorithms for cold start situation. 
*  implementing recommendation system with using MLlib ALS.
*  implementing recommendation system with using Mahout recommendation engine.

### Some Results: ###

#### Matrix Factorization MSE

![matrix_factorization](https://user-images.githubusercontent.com/38594307/55699012-b0f01000-5996-11e9-87b1-2f8ce6a58c6c.png)


#### Deep Learning Model MSE

![deeplearning](https://user-images.githubusercontent.com/38594307/55699033-d2e99280-5996-11e9-8f15-7e4a6cd79bc4.png)

#### Autorec(Autoencoder approach) Model MSE

![autorec](https://user-images.githubusercontent.com/38594307/55698993-99188c00-5996-11e9-821b-6eb714daebda.png)

### Cold Start Situation: using reinforcement learning and Bandits algorithms:

#### linUCB Average Rewards for 100 user after 100 steps:

![linucb](https://user-images.githubusercontent.com/38594307/55698996-a03f9a00-5996-11e9-979a-ce44d93d2398.png)

#### HybridLinUCB Average Rewards for 100 user after 100 steps:

![hybridlinucb](https://user-images.githubusercontent.com/38594307/55698998-a46bb780-5996-11e9-936c-032cf4a86dd0.png)

#### HybridLinUCB and linUCB Average Rewards comparison:

![coldstart_total](https://user-images.githubusercontent.com/38594307/55699003-aa619880-5996-11e9-872c-a503cfbce6b2.png)



### File Description ###

* [deeplearning_and_bandits.ipynb](https://github.com/bemova/RecomendationSystem/blob/master/deeplearning_and_bandits.ipynb): It is Jupyter notebook that contains the Matrix Factorization, Deeplearning model, Autorec(autoencoder model for recomendation), linUCB, and HybridLinUCB algorithm for cold start situation.

* [MahoutRecommendation](https://github.com/bemova/RecomendationSystem/tree/master/MahoutRecommendation/src/main/java): This folder contains Mahout recommendation implementation.

* [mahout-dataset-creator.py](https://github.com/bemova/RecomendationSystem/blob/master/mahout-dataset-creator.py): This python script is responsible for creating comma separated data-set for Mahout recommendation implementation because it accept only comma separated file as a input.

* [mllib-mahout-cf-result-chart.py](https://github.com/bemova/RecomendationSystem/blob/master/mllib-mahout-cf-result-chart.py): This python script is responsible for creating chart for comparing optimised cf, MLlib ALS result, and Mahout result for 5, 10, 20, 50, and 100 rates test set. 

* [movie-recommendations-als](https://github.com/bemova/RecomendationSystem/blob/master/movie-recommendations-als.py): This script is responsible for MLlib ALS implementation for recommendation system in our project.

* [movie-similarity-pairs-file-creator.py](https://github.com/bemova/RecomendationSystem/blob/master/movie-similarity-pairs-file-creator.py): This script is responsible for creating movie similarity pairs that is used for pre-processing in our project in order to reduce the execution time in collaborative filtering algorithm.

* [movie-similarity.py](https://github.com/bemova/RecomendationSystem/blob/master/movie-simillarity.py): This script is responsible for collaborative filtering implementation(all four attempt that we had in the project).

* [movies-features-illustrator.py](https://github.com/bemova/RecomendationSystem/blob/master/movies-features-illustrator.py): This script is responsible for extracting numerical and categorical features from movies.dat file which is in the 1M movielense data-set.

* [ratings-features-illustrator.py](https://github.com/bemova/RecomendationSystem/blob/master/ratings-features-illustrator.py): This script is responsible for extracting numerical and categorical features from ratings.dat file which is in the 1M movielense data-set. 

* [training-test-set-creator.py](https://github.com/bemova/RecomendationSystem/blob/master/training-test-set-creator.py): This script is responsible for creating test set and training set from ratings.dat file.

* [users-features-illustrator.py](https://github.com/bemova/RecomendationSystem/blob/master/users-features-illustrator.py): This script is responsible for extracting numerical and categorical features from users.dat file which is in the 1M movielense data-set.

