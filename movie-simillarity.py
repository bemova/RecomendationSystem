#! /usr/bin/env python3
import sys
from pyspark import SparkContext, SparkConf
from ast import literal_eval
from math import sqrt
import time


rates_threshold = 0.2
co_occurrence_threshold = 10
selected_movie_id = 0
selected_user_id = 0
names_dictionary = {}
same_genres = False


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


def pair_creator(string):
    movies, sim_count = literal_eval(string)
    return movies, sim_count


def filter_movies(pair):
    movies, sim_count = pair
    return (movies[0] == selected_movie_id or movies[1] == selected_movie_id) and sim_count[0] > rates_threshold \
           and sim_count[1] > co_occurrence_threshold


def filter_movies_same_genres(pair):
    movies, sim_count = pair
    paired_movie_genres = []
    if movies[0] == selected_movie_id or movies[1] == selected_movie_id:
        if movies[0] != selected_movie_id:
            paired_movie_genres = names_dictionary[movies[0]][1].split("|")
        else:
            paired_movie_genres = names_dictionary[movies[1]][1].split("|")
    else:
        return False
    intersect = set(paired_movie_genres).intersection(selected_movie_genres)
    return sim_count[0] > rates_threshold \
           and sim_count[1] > co_occurrence_threshold and len(intersect) > 0


def get_similar_movies(user_id, movie_id, same_genres):
    global selected_movie_id
    selected_movie_id = movie_id
    global selected_user_id
    selected_user_id = user_id
    if selected_movie_id:
        # scoreThreshold = 0.97
        # coOccurenceThreshold = 50
        conf = SparkConf().setMaster("local").setAppName("MovieSimilarities22").set("spark.ui.port", "4048")
        sc = SparkContext(conf=conf)
        print("\nLoading movie names...")
        selected_movie_id = int(selected_movie_id)
        global selected_movie_genres
        selected_movie_genres = names_dictionary[selected_movie_id][1].split("|")

        movie_pair_similarities = sc.textFile("/Users/arz/Desktop/bigdata-project/ml-1m/movies-similarities") \
            .map(pair_creator).cache()
        # data =movie_pair_similarities.first()
        # print(data)

        # Filter for movies with this sim that are "good" as defined by
        # our quality thresholds above (pair,sim)
        filtered_results = None
        if same_genres:
            filtered_results = movie_pair_similarities.filter(filter_movies_same_genres).cache()
            if filtered_results.count() < 5:
                filtered_results = movie_pair_similarities.filter(filter_movies)
        else:
            filtered_results = movie_pair_similarities.filter(filter_movies)

        results = filtered_results.map(lambda x: (x[1], x[0])).map(similarity_pair_creator)

        ratings_data = sc.textFile(training_path)
        users = ratings_data.map(lambda x: x.split("::")).filter(lambda x: (int(x[0]) == user_id))\
            .map(lambda x: (int(x[1]), int(x[2]))).cache()

        user_rates = users.join(results).collect()
        user_rates_for_similar_movie = []
        for user_rate in user_rates:
            pair = user_rate[1]
            movie_id = user_rate[0]
            movie_sim = pair[1]
            movie_rate = pair[0]
            user_rates_for_similar_movie.append((movie_id, movie_rate, movie_sim))
        sc.stop()
        sorted_rates_for_similar_movie = sorted(user_rates_for_similar_movie, key=lambda x: x[2])
        return sorted_rates_for_similar_movie[:10]


def similarity_pair_creator(result):
    (sim, pair) = result
    movie_pair_ids = pair[0]
    if movie_pair_ids == selected_movie_id:
        movie_pair_ids = pair[1]
    return movie_pair_ids, sim[0]


def get_overall_mean_movie():
    conf = SparkConf().setMaster('local').setAppName('inverted index6767').set("spark.ui.port", "4045")
    sc = SparkContext(conf=conf)
    ratings_data = sc.textFile(training_path)
    rates_data = ratings_data.map(lambda line: line.split("::"))
    rates = rates_data.map(lambda fields: int(fields[2])).cache()
    rates_count = rates.count()
    summation = rates.reduce(lambda x, y: x + y)
    mean_rating = summation / rates_count
    sc.stop()
    return mean_rating


def movie_rate_calculator(similar_movies_rates):
    total_sim = 0
    total_sim_rates = 0
    for item in similar_movies_rates:
        total_sim = total_sim + item[2]
        total_sim_rates = total_sim_rates + (item[1] * item[2])
    return total_sim_rates / total_sim


def movie_baseline_rate_calculator(similar_movies_rates, overall_mean_movie):
    total_sim = 0
    total_sim_rates = 0
    user_avg_rate = get_user_avg_rate(selected_user_id)
    movie_avg_rate = get_movie_avg_rate(selected_movie_id)
    bx = user_avg_rate - overall_mean_movie
    bi = movie_avg_rate - overall_mean_movie
    bxi = overall_mean_movie + bx + bi
    for item in similar_movies_rates:
        avg_movie_j_rate = get_movie_avg_rate(item[0])
        bj = avg_movie_j_rate - overall_mean_movie
        bxj = overall_mean_movie + bx + bj
        total_sim = total_sim + item[2]
        total_sim_rates = total_sim_rates + (item[2] * (item[1] - bxj))
    result = bxi + (total_sim_rates / total_sim)
    if result >= 5:
        return (selected_user_id, selected_movie_id), 5
    else:
        return (selected_user_id, selected_movie_id), bxi + (total_sim_rates / total_sim)


def get_user_avg_rate(user_id):
    conf = SparkConf().setMaster('local').setAppName('inverted index18898').set("spark.ui.port", "4047")
    sc = SparkContext(conf=conf)
    ratings_data = sc.textFile(training_path)
    rates_data = ratings_data.map(lambda line: line.split("::"))
    rates = rates_data.filter(lambda x: int(x[0]) == user_id).map(lambda fields: int(fields[2])).cache()
    rates_count = rates.count()
    summation = rates.reduce(lambda x, y: x + y)
    user_avg_rate = summation / rates_count
    sc.stop()
    return user_avg_rate


def get_movie_avg_rate(movie_id):
    conf = SparkConf().setMaster('local').setAppName('inverted index29797').set("spark.ui.port", "4049")
    sc = SparkContext(conf=conf)
    ratings_data = sc.textFile(training_path)
    rates_data = ratings_data.map(lambda line: line.split("::"))
    rates = rates_data.filter(lambda x: int(x[1]) == movie_id).map(lambda fields: int(fields[2])).cache()
    rates_count = rates.count()
    summation = rates.reduce(lambda x, y: x + y)
    movie_avg_rate = summation / rates_count
    sc.stop()
    return movie_avg_rate


def get_recommended_movies(selected_user_id, selected_movie_id, same_genres):
    recommended_movies = get_similar_movies(selected_user_id, selected_movie_id, same_genres)
    for movie in recommended_movies:
        print(movie)
        print(str(names_dictionary[movie[0]]) + "\tscore: " + str(movie[1]) + "\tstrength: " + str(movie[2]))


def rate_calculator(pair):
    user_id = pair[0]
    movie_id = pair[1][0]
    actual_rate = pair[1][1]
    recommended_movies = get_similar_movies(user_id, movie_id, same_genres)
    if len(recommended_movies) > 0:
        rate_user_movie = movie_rate_calculator(recommended_movies)
        return (user_id, movie_id), (rate_user_movie, actual_rate)


def rate_baseline_calculator(pair):
    user_id = int(pair[0])
    movie_id = int(pair[1][0])
    actual_rate = pair[1][1]
    # print((user_id, movie_id))
    recommended_movies = get_similar_movies(user_id, movie_id, same_genres)
    if len(recommended_movies) > 0:
        rate_user_movie = movie_baseline_rate_calculator(recommended_movies, overall_mean_movie)
        return (user_id, movie_id), (rate_user_movie[1], actual_rate)


def calculate_rates_from_file(same_genres):
    conf = SparkConf().setMaster("local").setAppName("MovieRecommendationsALS22").set("spark.ui.port", "4051")
    sc = SparkContext(conf=conf)
    random_user_pairs = sc.textFile(test_path) \
        .map(lambda x: x.split("::")).map(lambda x: (x[0], (x[1], x[2]))).cache()
    items = random_user_pairs.collect()
    sc.stop()
    for item in items:
        global selected_user_id
        selected_user_id = item[0]
        global selected_movie_id
        selected_movie_id = item[1][0]
        actual_rate = item[1][1]
        pair = (selected_user_id, selected_movie_id, actual_rate)
        rate_calculator(pair)


def calculate_rates():
    conf = SparkConf().setMaster("local").setAppName("MovieRecommendationsALS33").set("spark.ui.port", "4053")
    sc = SparkContext(conf=conf)
    random_user_pairs = sc.textFile(test_path) \
        .map(lambda x: x.split("::")).map(lambda x: (int(x[0]), (int(x[1]), int(x[2]))))
    if biases:
        recommended_actual_rate_pairs = random_user_pairs.map(rate_baseline_calculator).cache()
    else:
        recommended_actual_rate_pairs = random_user_pairs.map(rate_calculator).cache()
    mean_square_error = recommended_actual_rate_pairs.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
    print("Mean Squared Error = " + str(mean_square_error))
    print("Root Mean Squared Error = " + str(sqrt(mean_square_error)))
    items = recommended_actual_rate_pairs.collect()
    sc.stop()
    print(items)


def get_movie_user_recommendation():
    conf = SparkConf().setMaster("local").setAppName("MovieRecommendationsALS33").set("spark.ui.port", "4053")
    sc = SparkContext(conf=conf)
    random_user_pairs = sc.textFile(test_path) \
        .map(lambda x: x.split("::")).map(lambda x: (int(x[0]), (int(x[1]), int(x[2]))))
    recommended_actual_rate_pairs = random_user_pairs.map(rate_baseline_calculator).cache()
    mean_square_error = recommended_actual_rate_pairs.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
    print("Mean Squared Error = " + str(mean_square_error))
    print("Root Mean Squared Error = " + str(sqrt(mean_square_error)))
    items = recommended_actual_rate_pairs.collect()
    sc.stop()
    print(items)


def main():
    """
    instruction for retrieving top ten most similar movies:
    ./movie-simillarity.py [genre_effect] pair [userId] [movieId]
    sample: ./movie-simillarity.py True pair 119 3948

    instruction for calculating rates for a test set:
    ./movie-simillarity.py [genre_effect] set [training_set] [test_set]
    sample: ./movie-simillarity.py True set /Users/arz/Desktop/bigdata-project/ml-1m/ratings_training_5.dat /Users/arz/Desktop/bigdata-project/ml-1m/random_pairs_5
    """
    print(sys.argv)
    start = time.time()
    global names_dictionary
    names_dictionary = load_movie_names()
    global same_genres
    global biases
    global training_path
    global test_path
    rates_threshold = 0.1
    co_occurrence_threshold = 10
    if sys.argv[1] == 'True':
        same_genres = True
    else:
        same_genres = False
    if len(sys.argv) > 5:
        biases = True
    else:
        biases = False
    if sys.argv[2] == "set":
        training_path = sys.argv[3]
        test_path = sys.argv[4]
        global overall_mean_movie
        overall_mean_movie = get_overall_mean_movie()
        print(biases)
        calculate_rates()
    elif sys.argv[2] == "pair":
        print('pair')
        user_id = int(sys.argv[3])
        movie_id = int(sys.argv[4])
        global selected_movie_id
        global selected_user_id
        selected_movie_id = movie_id
        selected_user_id = user_id
        get_recommended_movies(user_id, movie_id, same_genres)


if __name__ == "__main__":
    main()
