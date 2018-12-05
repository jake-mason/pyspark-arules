from pyspark.mllib.fpm import FPGrowth
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql import functions as func

def tuple_str_handler(row):
	if isinstance(row, str):
		return (row,)
	else:
		return tuple(row)
	
def get_antecedent():
	def _get_antecedent(row):
		if len(row) > 1:
			return tuple_str_handler(row[:-1])
		else:
			return tuple_str_handler(row)
	return func.udf(_get_antecedent, ArrayType(StringType()))

def get_consequent():
	def _get_consequent(row):
		if len(row) > 1:
			return tuple_str_handler(row[-1])
		else:
			return tuple_str_handler(row)
	return func.udf(_get_consequent, ArrayType(StringType()))

class AssociationRules:

	def __init__(self, max_length=4):
		'''
		Initialize `arules.AssociationRules` object

		param:
			`max_length`: int, maximum itemset length. Smaller values should correspond to faster processing.
		'''
		self.max_length = max_length

	def fit(self, rdd, min_support):
		'''
		Mine frequent itemsets, using `pyspark.mllib.fpm.FPGrowth`

		param:
			`rdd`: PythonRDD, transactions
			`min_support`: float in [0, 1) or int in [1, inf]. If former, percentage of records; latter, number of records
			`n_partitions`: int, number of partitions
		'''
		self.rdd = rdd
		self.n = rdd.count()

		# Allow for passing "number of records" or percentage
		if min_support >= 1:
			min_support /= self.n

		model = FPGrowth.train(rdd, min_support, rdd.getNumPartitions())
		self.itemsets_df = model.freqItemsets().toDF()
		self._addl_itemset_setup()

	def _addl_itemset_setup(self):
		''' Accoutrements for `itemsets_df` '''
		self.itemsets_df = self.itemsets_df.withColumn('antecedent', get_antecedent()(func.col('items')))
		self.itemsets_df = self.itemsets_df.withColumn('consequent', get_consequent()(func.col('items')))
		self.itemsets_df = self.itemsets_df.withColumn('rule_len', func.udf(lambda x: len(x))(func.col('items')))
		self.itemsets_df = self.itemsets_df.withColumn('antecedent_len', func.udf(lambda x: len(x))(func.col('antecedent')))

		lookup_df = self.itemsets_df.selectExpr('items AS items_key', 'freq AS items_freq').distinct()

		self.itemsets_df = (
			self.itemsets_df
			.join(lookup_df, self.itemsets_df['antecedent'] == lookup_df['items_key'], 'left')
			.drop('items_key')
			.withColumnRenamed('items_freq', 'antecedent_freq')
		)

		self.itemsets_df = (
			self.itemsets_df
			.join(lookup_df, self.itemsets_df['consequent'] == lookup_df['items_key'], 'left')
			.drop('items_key')
			.withColumnRenamed('items_freq', 'consequent_freq')
		)

		self.itemsets_df = self.itemsets_df.where(func.col('rule_len') <= self.max_length)

	def _handle_single_itemset_metric(self, metric, df):
		''' Should not calculate multi-item metrics like confidence, lift, conviction, etc. for one-item sets '''
		return df.withColumn(metric, func.when(df['rule_len'] < 2, None).otherwise(df[metric]))

	def support(self):
		''' Support: |A & B| / n '''
		support = self.itemsets_df['freq'] / self.n
		df = self.itemsets_df.withColumn('support', support)
		return df

	def confidence(self):
		''' Confidence: |A & B| / |A| '''
		confidence = self.itemsets_df['freq'] / self.itemsets_df['antecedent_freq']
		df = self.itemsets_df.withColumn('confidence', confidence)
		df = self._handle_single_itemset_metric('confidence', df)
		return df

	def lift(self):
		''' Lift: (|A & B| / |A|) / (|B| / n) '''
		lift = (self.itemsets_df['freq'] / self.itemsets_df['antecedent_freq']) / (self.itemsets_df['consequent_freq'] / self.n)
		df = self.itemsets_df.withColumn('lift', lift)
		df = self._handle_single_itemset_metric('lift', df)
		return df

	def conviction(self):
		'''
		Conviction
		(1 - (|B| / n)) / (1 - (|A & B| / |A|)) = (1 - SUP(B)) / (1 - CONF(A, B))
		'''
		conviction = (1 - (self.itemsets_df['consequent_freq'] / self.n)) / (1 - (self.itemsets_df['freq'] / self.itemsets_df['antecedent_freq']))
		df = self.itemsets_df.withColumn('conviction', conviction)
		df = self._handle_single_itemset_metric('conviction', df)
		return df

	def calculate_all(self):
		''' Calculate all currently-defined association rules '''
		for metric_calculator in [self.support, self.confidence, self.lift, self.conviction]:
			self.itemsets_df = metric_calculator()
		return self.itemsets_df
