#! /usr/bin/env python3
import sys
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.mllib.evaluation import RegressionMetrics
import time
import argparse

names_dictionary = {}


def load_movie_names():
    names = {}
    with open("/Users/arz/Desktop/bigdata-project/ml-1m/movies.dat", encoding="ISO-8859-1") as f:
        for line in f:
            fields = line.split('::')
            try:
                names[int(fields[0])] = (str(fields[1]), fields[2])
            except:
                pass
    return names


def get_recommended_movies(selected_user_id):
    conf = SparkConf().setMaster("local[*]").setAppName("Movie Recommended with ALS")
    sc = SparkContext(conf=conf)

    print("\nLoading movie names...")
    name_dictionary = load_movie_names()

    data = sc.textFile("/Users/arz/Desktop/bigdata-project/ml-1m/ratings_training_5.dat")

    ratings = data.map(lambda l: l.split("::")).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
    rank = 10
    num_iterations = 20
    model = ALS.train(ratings, rank, num_iterations)

    print("\nTop 10 recommendations:")
    recommendations = model.recommendProducts(selected_user_id, 10)
    for recommendation in recommendations:
        print(recommendation)
        print(name_dictionary[int(recommendation[1])][0] + " score " + str(recommendation[2]))
    sc.stop()


def get_movie_rate():
    conf = SparkConf().setMaster("local[*]").setAppName("Movies Recommended Rates with ALS")
    sc = SparkContext(conf=conf)

    data = sc.textFile("/Users/arz/Desktop/bigdata-project/ml-1m/ratings_training_5.dat")

    ratings = data.map(lambda l: l.split("::")).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

    random_user_pairs_data = sc.textFile("/Users/arz/Desktop/bigdata-project/ml-1m/random_pairs_5").map(lambda x: x.split("::"))
    random_user_pairs = random_user_pairs_data.map(lambda x: (x[0], x[1])).cache()
    # print(random_user_pairs.collect())
    rank = 10
    num_iterations = 20
    alpha = 0.01
    model = ALS.train(ratings, rank, num_iterations, alpha)
    predictions = model.predictAll(random_user_pairs).map(lambda r: ((r[0], r[1]), r[2])).cache()
    rating_tuples = random_user_pairs_data.map(lambda x: ((int(x[0]), int(x[1])), float(x[2])))
    scores = predictions.join(rating_tuples)
    print(scores.collect())
    score_labels = scores.map(lambda x: x[1])
    metrics = RegressionMetrics(score_labels)
    root_mean_square_error = str(metrics.rootMeanSquaredError)
    sc.stop()
    return root_mean_square_error


def main():
    parser = argparse.ArgumentParser(description='users features illustrator help:')
    # defines the query number arguments which should be answered, and the query number arguments help message
    parser.add_argument('-q', '--question', type=int,
                        help='you should enter the question or query number.', required=True)
    # parse script argument and find mistakes in calling script and return appropriate messages
    args = parser.parse_args()
    # if the range of query and question number is bigger that 2, application return ERROR messages and shows help.
    if args.question and args.question > 2:
        print("ERROR: question number is in rage of 1 to 2")
        parser.print_help()
        sys.exit(0)
    # answer query 1 if user enters query number 1 after '-q'
    if args.question == 1:
        start = time.time()
        global names_dictionary
        names_dictionary = load_movie_names()
        root_mean_square_error = get_movie_rate()
        print("Root Mean Squared Error = " + root_mean_square_error)
        end = time.time()
        print("Execution time = " + str(end - start))
    # answer query 2 if user enters query number 2 after '-q'
    elif args.question == 2:
        start = time.time()
        selected_movie_id = 1193
        selected_user_id = 1
        get_recommended_movies(selected_user_id)
        end = time.time()
        print("Execution time = " + str(end - start))


if __name__ == "__main__":
    main()
