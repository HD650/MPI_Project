# put data into hdfs
hdfs dfs -put input /user/zzhang63/input
# compile hadoop code
make
# run hadoop job
hadoop jar TFIDF.jar TFIDF /user/zzhang63/input &> hadoop_output.txt
# get output from hdfs
hdfs dfs -get /user/zzhang63/output ./job_output
