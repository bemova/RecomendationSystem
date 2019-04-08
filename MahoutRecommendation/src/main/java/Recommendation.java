import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.EuclideanDistanceSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.common.RandomUtils;

/**
 * Created by arz on 2017-04-01.
 */


public class Recommendation {
    public static void main(String[] args) throws Exception {
        long startTime = System.nanoTime();
        RecommenderBuilder itemSimRecommendationBuilder = new RecommenderBuilder() {
            @Override
            public Recommender buildRecommender(DataModel model) throws TasteException {
//              DataModel model = new FileDataModel(new File("/Users/arz/Desktop/bigdata-project/ml-1m/mahout-dataset.csv"));
                ItemSimilarity similarity = new PearsonCorrelationSimilarity(model);
//              UserNeighborhood neighborhood = new NearestNUserNeighborhood(10, userSimilarity, model);
                Recommender recommender = new GenericItemBasedRecommender(model, similarity);
                return recommender;

            }
        };
        RandomUtils.useTestSeed();
        FileDataModel model = new FileDataModel(new File("/Users/arz/Desktop/bigdata-project/ml-1m/mahout-dataset.csv"));

        RecommenderEvaluator evaluator = new RMSRecommenderEvaluator();
        double v = 1 - 100.0 / 1000209;
        //for obtaining User Similarity Evaluation Score
        double userSimEvaluationScore = evaluator.evaluate(itemSimRecommendationBuilder,null,model, v, 1.0);
        System.out.println("User Similarity Evaluation score : "+userSimEvaluationScore);
        long endTime = System.nanoTime();
        long duration = (endTime - startTime);
        System.out.println(duration / 1000000000);

    }
}