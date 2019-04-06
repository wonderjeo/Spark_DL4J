import org.apache.hadoop.conf.Configuration;
import java.io.IOException;
import org.apache.hadoop.fs.FileSystem;

public class SimpleApp {
    public static void main(String[] args){
        Configuration conf = new Configuration();
        conf.set("fs.defaultFS", "hdfs://gpu18:9000/data/lr.txt");
        try {
            FileSystem fs = FileSystem.get(conf);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
