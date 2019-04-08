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
    public static void main(String[] args){
        SparkConf conf = new SparkConf()
                .setAppName("SimpleApp")
                .set("spark.kryo.registrator", "org.nd4j.Nd4jRegistrator")
                .setMaster("yarn");

        JavaSparkContext sc = new JavaSparkContext(conf);

        final List<String> lstLabelNames = Arrays.asList("0","1","2","3","4","5","6","7","8","9");
        final ImageLoader imageLoader = new ImageLoader(28,28,1);
        //System.out.println("Loading"+imageLoader);
        final DataNormalization scaler = new ImagePreProcessingScaler(0,1);

        String srcPath = "hdfs:///data/test";
        FileSystem hdfs = null;
        try {
            hdfs = FileSystem.get(URI.create(srcPath), sc.hadoopConfiguration());
        } catch (IOException e) {
            e.printStackTrace();
        }
        FileStatus[] fileList = null;
        try {
            fileList = hdfs.listStatus(new Path(srcPath));
        } catch (IOException e) {
            e.printStackTrace();
        }
        List<String> lstFilePath = new ArrayList<>();
        for(FileStatus fileStatus : fileList){
            lstFilePath.add(srcPath + "/" + fileStatus.getPath().getName());
        }
        JavaRDD<String> javaRDDImagePath = sc.parallelize(lstFilePath);
        JavaRDD<DataSet> javaRDDImageTrain = javaRDDImagePath.map(new Function<String, DataSet>() {
            @Override
            public DataSet call(String imagePath) throws Exception {
                FileSystem fs = FileSystem.get(new Configuration());
                DataInputStream in = fs.open(new Path(imagePath));
                INDArray features = imageLoader.asRowVector(in);
                //String[] tokens = imagePath.split("\\/");
                //String label = tokens[tokens.length-1].split("\\.")[0];
                int intLabel = 1;
                INDArray labels = Nd4j.zeros(10);
                labels.putScalar(0,intLabel, 1.0);
                DataSet trainData = new DataSet(labels,features);
                trainData.setLabelNames(lstLabelNames);
                scaler.preProcess(trainData);
                fs.close();
                return trainData;
            }
        });

        javaRDDImageTrain.saveAsObjectFile("hdfs:///mnistNorm.dat");
    }
}
