import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.*;
import scala.Tuple2;

import java.util.Arrays;
import java.util.Iterator;

public class SimpleApp {
    public static void main(String[] args){
        SparkConf conf = new SparkConf()
                .setAppName("SimpleApp")
                .setMaster("yarn");

        JavaSparkContext sc = new JavaSparkContext(conf);
        JavaRDD<String> lines = sc.textFile("hdfs:///data/lr.txt",1);
        JavaRDD<String> Words = lines.flatMap(new FlatMapFunction<String, String>() {
            @Override
            public Iterator<String> call(String line) throws Exception {
                return (Iterator<String>) Arrays.asList(line.split(" "));
            }
        });

        JavaPairRDD<String, Integer> pairs = Words.mapToPair(new PairFunction<String, String, Integer>() {
            @Override
            public Tuple2<String, Integer> call(String word) throws Exception {
                return new Tuple2<String, Integer>(word, 1);
            }
        });
        JavaPairRDD<String, Integer> wordscount = pairs.reduceByKey(new Function2<Integer, Integer, Integer>() {
            @Override
            public Integer call(Integer v1, Integer v2) throws Exception {
                return v1+v2;
            }
        });

        wordscount.foreach(new VoidFunction<Tuple2<String, Integer>>() {
            @Override
            public void call(Tuple2<String, Integer> pairs) throws Exception {
                System.out.println(pairs._1+":"+pairs._2);
            }
        });


    }
}
