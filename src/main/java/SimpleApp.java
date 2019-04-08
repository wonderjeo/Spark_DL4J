import org.apache.avro.generic.GenericData;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;
import org.datavec.api.writable.Writable;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.*;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.spark.datavec.DataVecDataSetFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.apache.hadoop.fs.FileSystem;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class SimpleApp {
    public static void main(String[] args) throws IOException {
        SparkConf conf = new SparkConf()
                .setMaster("yarn")  //local mode
                .set("spark.kryo.registrator", "org.nd4j.Nd4jRegistrator")
                .setAppName("Mnist Java Spark (Java)");
        JavaSparkContext jsc = new JavaSparkContext(conf);

        final List<String> lstLabelNames = Arrays.asList("零","一","二","三","四","五","六","七","八","九");  //Chinese Label
        final ImageLoader imageLoader = new ImageLoader(28, 28, 1);             //Load Image
        final DataNormalization scaler = new ImagePreProcessingScaler(0, 1);    //Normalize

        String srcPath = "hdfs:///data/test";
        FileSystem hdfs = FileSystem.get(URI.create(srcPath),jsc.hadoopConfiguration());    //hdfs read local file system
        FileStatus[] fileList = hdfs.listStatus(new Path(srcPath));
        List<String> lstFilePath = new ArrayList<>();
        for( FileStatus fileStatus :  fileList){
            lstFilePath.add(srcPath + "/" + fileStatus.getPath().getName());
        }
        JavaRDD<String> javaRDDImagePath = jsc.parallelize(lstFilePath);
        JavaRDD<DataSet> javaRDDImageTrain = javaRDDImagePath.map(new Function<String, DataSet>() {

            @Override
            public DataSet call(String imagePath) throws Exception {
                FileSystem fs = FileSystem.get(new Configuration());
                DataInputStream in = fs.open(new Path(imagePath));
                INDArray features = imageLoader.asRowVector(in);            //features tensor
                String[] tokens = imagePath.split("\\/");
                String label = tokens[tokens.length-1].split("\\_")[0];
                int intLabel = Integer.parseInt(label);
                INDArray labels = Nd4j.zeros(10);                           //labels tensor
                labels.putScalar(0, intLabel, 1.0);
                DataSet trainData = new DataSet(features, labels);          //DataSet, wrapper of features and labels
                trainData.setLabelNames(lstLabelNames);
                scaler.preProcess(trainData);                               //normalize
                fs.close();
                return trainData;
            }
        });
        javaRDDImageTrain.saveAsObjectFile("hdfs:///mnistNorm.dat");
    }
}
