# Bigdata-project
Explainable Machine Learning for Twitter Sentiment Analysis: Understanding Why Tweets are Classified as Positive or Negative

More step into project and you requirment library on google colab before run code: 
Step 1 run this: /n
!rm -rf /content/spark-3.5.0-bin-hadoop3*
!rm -rf /content/*.tgz
!wget -q https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz -O /content/spark-3.5.0-bin-hadoop3.tgz

Step 2 Dected Java home and Spark home: 
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.5.0-bin-hadoop3"

Step 3 Run this to download spark: 
!tar xf /content/spark-3.5.0-bin-hadoop3.tgz -C /content/

Step 4 Run code in file called: 
FileProject.py
