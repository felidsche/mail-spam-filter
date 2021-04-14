# mail-spam-filter
- a binary classifier (Logistic Regression + SVM) to mark emails as spam or not spam

## Setup on your local machine

### Download Apache Spark 2.4.6 distribution pre-built for Apache Hadoop 2.7 [link](http://spark.apache.org/downloads.html).
- unpack the archive
- set the `$SPARK_HOME` environment variable `export SPARK_HOME=$(pwd)`
### add the Apache Spark librariers to an IDE (i.e. PyCharm)
- navigate to `PyCharm → Preferences ... → Project spark-demo → Project Structure → Add Content Root` in the main menu
- select all `.zip` files from `$SPARK_HOME/python/lib` 
- click apply and save changes

### create a new run configuration in your IDE
- navigate to `Run → Edit Configurations → + → Python` in the main menu
- select `email_spam_filter.py` for `Script`
- name it `email_spam_filter`
### add environment variables in the run configuration
- `PYSPARK_PYTHON=python3`
- `PYTHONPATH=$SPARK_HOME/python`
- `PYTHONUNBUFFERED=1`

### provide the input data
- the training data `nospam_training.txt`, `spam_training.txt`, as well as the testing data `nospam_testing.txt`, `pam_testing.txt` need to be under `../spam-datasets/*.txt` relative to the script path

### run the script within Apache Spark context
- click `Run → Run 'email_spam_filter'` in the main menu

### check the [webUI](http://localhost:4040) to monitor a running Apache Spark job

