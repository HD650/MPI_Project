import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

import java.io.IOException;
import java.util.*;

/*
 * Main class of the TFIDF MapReduce implementation.
 * Author: Tyler Stocksdale
 * Date:   10/18/2017
 */
public class TFIDF {

    public static void main(String[] args) throws Exception {
        // Check for correct usage
        if (args.length != 1) {
            System.err.println("Usage: TFIDF <input dir>");
            System.exit(1);
        }
		
		// Create configuration
		Configuration conf = new Configuration();
		
		// Input and output paths for each job
		Path inputPath = new Path(args[0]);
		Path wcInputPath = inputPath;
		Path wcOutputPath = new Path("output/WordCount");
		Path dsInputPath = wcOutputPath;
		Path dsOutputPath = new Path("output/DocSize");
		Path tfidfInputPath = dsOutputPath;
		Path tfidfOutputPath = new Path("output/TFIDF");
		
		// Get/set the number of documents (to be used in the TFIDF MapReduce job)
        FileSystem fs = inputPath.getFileSystem(conf);
        FileStatus[] stat = fs.listStatus(inputPath);
		String numDocs = String.valueOf(stat.length);
		conf.set("numDocs", numDocs);
		
		// Delete output paths if they exist
		FileSystem hdfs = FileSystem.get(conf);
		if (hdfs.exists(wcOutputPath))
			hdfs.delete(wcOutputPath, true);
		if (hdfs.exists(dsOutputPath))
			hdfs.delete(dsOutputPath, true);
		if (hdfs.exists(tfidfOutputPath))
			hdfs.delete(tfidfOutputPath, true);
		
		// Create and execute Word Count job
		Job word_count = Job.getInstance(conf, "word_count");
		word_count.setJarByClass(TFIDF.class);
		word_count.setMapperClass(WCMapper.class);
		word_count.setReducerClass(WCReducer.class);
		word_count.setOutputKeyClass(Text.class);
		word_count.setOutputValueClass(IntWritable.class);
		FileInputFormat.addInputPath(word_count, wcInputPath);
		FileOutputFormat.setOutputPath(word_count, wcOutputPath);
		word_count.waitForCompletion(true);

		// Create and execute Document Size job
		Job doc_size = Job.getInstance(conf, "doc_size");
		doc_size.setJarByClass(TFIDF.class);
		doc_size.setMapperClass(DSMapper.class);
		doc_size.setReducerClass(DSReducer.class);
		doc_size.setOutputKeyClass(Text.class);
		doc_size.setOutputValueClass(Text.class);
		FileInputFormat.addInputPath(doc_size, dsInputPath);
		FileOutputFormat.setOutputPath(doc_size, dsOutputPath);
		doc_size.waitForCompletion(true);
		
		//Create and execute TFIDF job
		Job tfidf = Job.getInstance(conf, "tfidf");
		tfidf.setJarByClass(TFIDF.class);
		tfidf.setMapperClass(TFIDFMapper.class);
		tfidf.setReducerClass(TFIDFReducer.class);
		tfidf.setOutputKeyClass(Text.class);
		tfidf.setOutputValueClass(Text.class);
		FileInputFormat.addInputPath(tfidf, tfidfInputPath);
		FileOutputFormat.setOutputPath(tfidf, tfidfOutputPath);
		tfidf.waitForCompletion(true);
		
    }
	
	/*
	 * Creates a (key,value) pair for every word in the document 
	 *
	 * Input:  ( byte offset , contents of one line )
	 * Output: ( (word@document) , 1 )
	 *
	 * word = an individual word in the document
	 * document = the filename of the document
	 */
	public static class WCMapper extends Mapper<Object, Text, Text, IntWritable> {
		
		private final static IntWritable one = new IntWritable(1);
		private Text word = new Text();

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			// get doc name
			String doc_name = ((FileSplit)context.getInputSplit()).getPath().getName();
			StringTokenizer itr = new StringTokenizer(value.toString());
			// for every word in doc, output a key-value pair
			while (itr.hasMoreTokens())
			{
				word.set(itr.nextToken()+"@"+doc_name);
				context.write(word, one);
			}
		}
    }

    /*
	 * For each identical key (word@document), reduces the values (1) into a sum (wordCount)
	 *
	 * Input:  ( (word@document) , 1 )
	 * Output: ( (word@document) , wordCount )
	 *
	 * wordCount = number of times word appears in document
	 */
	public static class WCReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
		
		private IntWritable result = new IntWritable();
		public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
			int sum = 0;
			// count the same word
			for (IntWritable val : values)
			{
				sum += val.get();
			}
			result.set(sum);
			context.write(key, result);
		}
    }
	
	/*
	 * Rearranges the (key,value) pairs to have only the document as the key
	 *
	 * Input:  ( (word@document) , wordCount )
	 * Output: ( document , (word=wordCount) )
	 */
	public static class DSMapper extends Mapper<Object, Text, Text, Text> {
		private Text out_key = new Text(), out_value = new Text();	
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			String[] token = ((Text)value).toString().split("\t");
			String[] item = token[0].split("@");
			// rearrange the key-value pair
			out_key.set(item[1]);
			out_value.set(item[0]+"="+token[1]);
			context.write(out_key, out_value);
		}
    }

    /*
	 * For each identical key (document), reduces the values (word=wordCount) into a sum (docSize) 
	 *
	 * Input:  ( document , (word=wordCount) )
	 * Output: ( (word@document) , (wordCount/docSize) )
	 *
	 * docSize = total number of words in the document
	 */
	public static class DSReducer extends Reducer<Text, Text, Text, Text> {
		private Text out_key = new Text(), out_value = new Text();
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			int sum = 0;
			ArrayList<String> words = new ArrayList<String>();
			ArrayList<String> counts = new ArrayList<String>();
			// get the total number of word for every doc
			for (Text val : values)
			{
				String [] token = ((Text)val).toString().split("=");
				sum += Integer.parseInt(token[1]);
				words.add(token[0]);
				counts.add(token[1]);
			}
			// calculate wordCount/docSize
			for (int i=0; i<words.size();i++)
			{
				out_key.set(words.get(i)+"@"+key);
				out_value.set(counts.get(i)+"/"+String.valueOf(sum));
				context.write(out_key, out_value);
			}
		}
    }
	
	/*
	 * Rearranges the (key,value) pairs to have only the word as the key
	 * 
	 * Input:  ( (word@document) , (wordCount/docSize) )
	 * Output: ( word , (document=wordCount/docSize) )
	 */
	public static class TFIDFMapper extends Mapper<Object, Text, Text, Text> {
		private Text out_key = new Text(), out_value = new Text();	
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			String[] token = ((Text)value).toString().split("\t");
			String[] key_item = token[0].split("@");
			// rearranges key-value pair
			out_key.set(key_item[0]);
			out_value.set(key_item[1]+"="+token[1]);
			context.write(out_key, out_value);
		}
    }

    /*
	 * For each identical key (word), reduces the values (document=wordCount/docSize) into a 
	 * the final TFIDF value (TFIDF). Along the way, calculates the total number of documents and 
	 * the number of documents that contain the word.
	 * 
	 * Input:  ( word , (document=wordCount/docSize) )
	 * Output: ( (document@word) , TFIDF )
	 *
	 * numDocs = total number of documents
	 * numDocsWithWord = number of documents containing word
	 * TFIDF = (wordCount/docSize) * ln(numDocs/numDocsWithWord)
	 *
	 * Note: The output (key,value) pairs are sorted using TreeMap ONLY for grading purposes. For
	 *       extremely large datasets, having a for loop iterate through all the (key,value) pairs 
	 *       is highly inefficient!
	 */
	public static class TFIDFReducer extends Reducer<Text, Text, Text, Text> {
		
		private static int numDocs;
		private Map<Text, Text> tfidfMap = new HashMap<>();
		
		// gets the numDocs value and stores it
		protected void setup(Context context) throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
			numDocs = Integer.parseInt(conf.get("numDocs"));
		}
		
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			ArrayList<String> documents = new ArrayList<String>();
			ArrayList<String> docsizes = new ArrayList<String>();
			int numDocsWithWord = 0;
			// count the number of doc with this specific word
			for (Text val : values)
			{
				String [] token = ((Text)val).toString().split("=");
				numDocsWithWord += 1;
				documents.add(token[0]);
				docsizes.add(token[1]);
			}
			// calculate tfidf of this word
			for (int i=0; i<documents.size();i++)
			{
				String[] formulate = docsizes.get(i).split("/");
				double result = ((double)Integer.parseInt(formulate[0]))/((double)Integer.parseInt(formulate[1]))*Math.log(((double)numDocs)/((double)numDocsWithWord));
				Text out_key = new Text(documents.get(i)+"@"+key);
				Text out_value = new Text(String.valueOf(result));
				tfidfMap.put(out_key, out_value);
			}
		}
		
		// sorts the output (key,value) pairs that are contained in the tfidfMap
		protected void cleanup(Context context) throws IOException, InterruptedException {
            Map<Text, Text> sortedMap = new TreeMap<Text, Text>(tfidfMap);
			for (Text key : sortedMap.keySet()) {
                context.write(key, sortedMap.get(key));
            }
        }
		
    }
}
