# put data into hdfs
hdfs dfs -put input /user/zzhang63/input
# compile hadoop code
make
# run hadoop job
spark-submit --class TFIDF TFIDF.jar input
