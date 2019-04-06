import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;


import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;

public class SimpleApp {
    public static void main(String[] args){
        JavaSparkContext ctx = new JavaSparkContext("yarn", "JavaWordCount",
                System.getenv("SPARK_HOME"), JavaSparkContext.jarOfClass(SimpleApp.class));

        System.out.println("spark home:" + ctx.getSparkHome());
    }
}
