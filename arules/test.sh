#!/bin/bash

PYTHON_INTERPRETER_PATH="$(which python3)"

spark-submit \
	--master local \
	--deploy-mode client \
	--conf spark.pyspark.driver.python=$PYTHON_INTERPRETER_PATH \
	--conf spark.pyspark.python=$PYTHON_INTERPRETER_PATH \
	--num-executors 2 \
	--executor-memory 1G \
	--executor-cores 2 \
	--driver-memory 1G \
	--py-files arules.py \
	test.py