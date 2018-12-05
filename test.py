'''
spark-submit \
	--master yarn \
	--deploy-mode client \
	--conf spark.pyspark.driver.python=/hadoop/opt/python_anaconda/3-5.1.0/bin/python \
	--conf spark.pyspark.python=/hadoop/opt/python_anaconda/3-5.1.0/bin/python \
	--conf spark.executorEnv.PYTHONHASHSEED=321 \
	--num-executors 2 \
	--executor-memory 1G \
	--executor-cores 2 \
	--driver-memory 1G \
	--py-files arules.py \
	test.py
'''

import unittest

from pyspark.sql import SparkSession, functions as func
from pyspark.conf import SparkConf

from arules import AssociationRules

class TestAssociationRules(unittest.TestCase):

	def setUp(self):

		self.spark = (
			SparkSession.builder
			.config(conf=SparkConf())
			.appName('arules-test')
			.getOrCreate()
		)

		sc = self.spark.sparkContext

		data = [
			['apple', 'orange', 'banana'],
			['apple', 'juice', 'bread'],
			['orange', 'banana', 'bread'],
			['beer', 'diapers'],
			['beer', 'apple']
		]

		rdd = sc.parallelize(data)
		self.arules = AssociationRules(max_length=5)
		self.arules.fit(rdd, 0.00001)
		self.arules.itemsets_df = self.arules.itemsets_df.withColumn('antecedent_str', func.udf(lambda x: '|'.join(x))(func.col('antecedent')))
		self.arules.itemsets_df = self.arules.itemsets_df.withColumn('consequent_str', func.udf(lambda x: '|'.join(x))(func.col('consequent')))

	def tearDown(self):
		pass
	
	def rule_metric_grabber(self, rule_col, rule_df, antecedent, consequent):
		mask = (rule_df['antecedent_str'] == antecedent) & (rule_df['consequent_str'] == consequent)
		return rule_df.where(mask).select(rule_col).take(1)[0].asDict()[rule_col]

	def test_support(self):
		support = self.arules.support()
		self.assertEqual(self.rule_metric_grabber('support', support, 'banana', 'bread'), 0.2)
		self.assertEqual(self.rule_metric_grabber('support', support, 'diapers', 'beer'), 0.2)

	def test_confidence(self):
		confidence = self.arules.confidence()
		self.assertEqual(self.rule_metric_grabber('confidence', confidence, 'banana', 'bread'), 0.5)
		self.assertEqual(self.rule_metric_grabber('confidence', confidence, 'diapers', 'beer'), 1.0)

	def test_lift(self):
		lift = self.arules.lift()
		self.assertEqual(self.rule_metric_grabber('lift', lift, 'banana', 'bread'), 1.25)
		self.assertEqual(self.rule_metric_grabber('lift', lift, 'diapers', 'beer'), 2.5)

	def test_conviction(self):
		conviction = self.arules.conviction()
		self.assertEqual(self.rule_metric_grabber('conviction', conviction, 'banana', 'bread'), 1.2)
		self.assertEqual(self.rule_metric_grabber('conviction', conviction, 'diapers', 'beer'), None)

if __name__ == '__main__':
	unittest.main()