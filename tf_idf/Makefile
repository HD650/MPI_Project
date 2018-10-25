all: jar_file

clean: 
	rm *.jar *.class

jar_file: class_file
	jar cf TFIDF.jar TFIDF*.class

class_file: TFIDF.java
	javac TFIDF.java
