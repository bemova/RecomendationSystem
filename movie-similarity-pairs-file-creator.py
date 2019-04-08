#! /usr/bin/env python3
from pyspark import SparkContext, SparkConf
from math import sqrt


def remove_duplicates(pair):
    user_id, ratings = pair
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return movie1 < movie2


def make_movies_rates_pairs(pair):
    user, ratings = pair
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return (movie1, movie2), (rating1, rating2)


def get_cosine_similarity(pairs):
    pairs_count = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in pairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        pairs_count += 1
    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)
    score = 0
    if (denominator):
        score = (numerator / (float(denominator)))
    return score, pairs_count


def create_similarity():
    conf = SparkConf().setMaster("local[*]").setAppName("MovieSimilarities11").set("spark.ui.port", "4041")
    sc = SparkContext(conf=conf)
    data = sc.textFile("/Users/arz/Desktop/bigdata-project/ml-1m/ratings.dat")

    # Map ratings to key / value pairs: user ID => movie ID, rating
    ratings = data.map(lambda x: x.split("::")).map(lambda x: (int(x[0]), (int(x[1]), float(x[2]))))

    # Emit every movie rated together by the same user.
    # Self-join to find every combination.
    # each will be (4096, ((2987, 4.0), (2987, 4.0)))
    user_joined_rates = ratings.join(ratings)

    # At this point our RDD consists of userID => ((movieID, rating), (movieID, rating))

    # Filter out duplicate pairs, then key by (movie1, movie2) pairs. pair is ((movie1_id, movie2_id), (rate1, rate2))
    movie_pairs = user_joined_rates.filter(remove_duplicates).map(make_movies_rates_pairs)

    # We now have (movie1, movie2) => (rating1, rating2)
    # Now collect all ratings for each movie pair and compute similarity
    # We now have (movie1, movie2) = > (rating1, rating2), (rating1, rating2) ...
    movie_pair_ratings = movie_pairs.groupByKey()

    # We now have (movie1, movie2) = > (rating1, rating2), (rating1, rating2) ...
    # We Can now compute similarities.
    movie_pair_similarities = movie_pair_ratings.mapValues(get_cosine_similarity).cache()

    # Save the results if desired
    movie_pair_similarities.sortByKey()
    movie_pair_similarities.repartition(1).saveAsTextFile("/Users/arz/Desktop/bigdata-project/ml-1m/movies-similarities")
    sc.stop()


def main():
    create_similarity()


if __name__ == "__main__":
    main()
