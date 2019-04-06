import org.datavec.api.writable.Writable;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.*;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.deeplearning4j.spark.datavec.DataVecDataSetFunction;
import org.nd4j.linalg.dataset.DataSet;
import scala.Tuple2;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class SimpleApp {
    public static void main(String[] args){
        SparkConf conf = new SparkConf()
                .setAppName("SimpleApp")
                .setMaster("yarn");

        JavaSparkContext sc = new JavaSparkContext(conf);
        JavaRDD<String> rddString = sc.textFile("hdfs:///data/fashion-mnist_test.csv");



    }
}
