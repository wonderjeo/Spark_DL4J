import org.apache.spark.SparkConf;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.sql.Dataset;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.deeplearning4j.spark.datavec.DataVecDataSetFunction;
import org.nd4j.linalg.dataset.DataSet;
import scala.Tuple2;


import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;

public class SimpleApp {
    public static void main(String[] args){
        SparkConf conf = new SparkConf()
                .setAppName("SimpleApp")
                .setMaster("yarn");
        String filePath = "hdfs:///data/fashion-mnist_test.csv";
        JavaSparkContext sc = new JavaSparkContext(conf);
        JavaRDD<String> rddString = sc.textFile(filePath);
        RecordReader recordReader = new CSVRecordReader(',');
        JavaRDD<List<Writable>> rddWriteables = rddString.map(new StringToWritablesFunction(recordReader));

        int labelIndex = 1;
        int numLabelClasses = 10;
        JavaRDD<DataSet> rddDataSetClassification = rddWriteables.map(new DataVecDataSetFunction(labelIndex, numLabelClasses, false));



    }
}
