import org.apache.avro.generic.GenericData;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
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
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.datavec.DataVecDataSetFunction;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.apache.hadoop.fs.FileSystem;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.DataInputStream;
import java.io.File;
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
        final int numBatch = 32;
        int nChannels = 1;
        int outputNum = 10;
        int iterations = 1;
        int seed = 123;
        int nEpochs = 1;

        String srcPath = "hdfs:///data/food";
        Configuration hconf = new Configuration();
        Path path = new Path(srcPath);
        FileSystem hdfs = path.getFileSystem(hconf);    //hdfs read local file system
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
        //javaRDDImageTrain.saveAsObjectFile("hdfs:///mnistNorm.dat");
        ParameterAveragingTrainingMaster trainMaster = new ParameterAveragingTrainingMaster.Builder(numBatch)   //weight average service
                .workerPrefetchNumBatches(0)
                .saveUpdater(true)
                .averagingFrequency(5)
                .batchSizePerWorker(numBatch)
                .build();
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.01, 0.9))
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28,28,1)) //See note below
                .backprop(true).pretrain(false);

        //new ConvolutionLayerSetup(builder,28,28,1);
        MultiLayerConfiguration netconf = builder.build();
        MultiLayerNetwork net = new MultiLayerNetwork(netconf);
        net.setListeners(new ScoreIterationListener(1));
        net.init();
        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(jsc, net, trainMaster);
        //train the network on Spark
        for( int i = 0; i < nEpochs; ++i ){
            sparkNetwork.fit(javaRDDImageTrain);
            System.out.println("----- Epoch " + i + " complete -----");
            Evaluation evalActual = sparkNetwork.evaluate(javaRDDImageTrain);
            System.out.println(evalActual.stats());
        }
        FileSystem fhdfs = FileSystem.get(jsc.hadoopConfiguration());
        Path hdfsPath = new Path("hdfs:///minist-model.zip");
        FSDataOutputStream outputStream = fhdfs.create(hdfsPath);
        MultiLayerNetwork trainedNet = sparkNetwork.getNetwork();
        ModelSerializer.writeModel(trainedNet, outputStream, true);
    }
}
