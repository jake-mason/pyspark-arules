# arules
Association rules on PySpark. Since the implementation of more sophisticated association rules seems to be a [back-burner item for Spark](https://issues.apache.org/jira/browse/SPARK-10697), hopefully this library can provide some much-needed a-rules utilities to those in search.

## How to get started

Here is some faux, carb-loaded transactional data. We'll create a [Resilient Distributed Dataset (RDD)](https://spark.apache.org/docs/latest/rdd-programming-guide.html) out of the nested iterable structure:

```python
data = [
    ['apple', 'orange', 'banana'],
    ['apple', 'juice', 'bread'],
    ['orange', 'banana', 'bread'],
    ['beer', 'diapers'],
    ['beer', 'apple']
]

rdd = sc.parallelize(data)
```

Next, we can import the `arules.AssociationRules` class and fit the [`pyspark.mllib.fpm.FPGrowth`](https://spark.apache.org/docs/2.3.0/mllib-frequent-pattern-mining.html) model for mining frequent itemsets:

```python
from arules.arules import AssociationRules

arules = AssociationRules()

arules.fit(rdd, min_support=0.00001)
```

There are currently three association rule measures implemented:

### Support

```python
arules.support().show()
# +--------------------+----+----------------+----------+--------+--------------+---------------+---------------+-------+
# |               items|freq|      antecedent|consequent|rule_len|antecedent_len|antecedent_freq|consequent_freq|support|
# +--------------------+----+----------------+----------+--------+--------------+---------------+---------------+-------+
# |            [orange]|   2|        [orange]|  [orange]|       1|             1|              2|              2|    0.4|
# |     [orange, apple]|   1|        [orange]|   [apple]|       2|             1|              2|              3|    0.2|
# |             [apple]|   3|         [apple]|   [apple]|       1|             1|              3|              3|    0.6|
# |      [juice, apple]|   1|         [juice]|   [apple]|       2|             1|              1|              3|    0.2|
# |[orange, banana, ...|   1|[orange, banana]|   [apple]|       3|             2|              2|              3|    0.2|
# |     [banana, apple]|   1|        [banana]|   [apple]|       2|             1|              2|              3|    0.2|
# |      [bread, apple]|   1|         [bread]|   [apple]|       2|             1|              2|              3|    0.2|
# |       [beer, apple]|   1|          [beer]|   [apple]|       2|             1|              2|              3|    0.2|
# |[juice, bread, ap...|   1|  [juice, bread]|   [apple]|       3|             2|              1|              3|    0.2|
# |             [juice]|   1|         [juice]|   [juice]|       1|             1|              1|              1|    0.2|
# |           [diapers]|   1|       [diapers]| [diapers]|       1|             1|              1|              1|    0.2|
# |    [orange, banana]|   2|        [orange]|  [banana]|       2|             1|              2|              2|    0.4|
# |            [banana]|   2|        [banana]|  [banana]|       1|             1|              2|              2|    0.4|
# |     [orange, bread]|   1|        [orange]|   [bread]|       2|             1|              2|              2|    0.2|
# |      [juice, bread]|   1|         [juice]|   [bread]|       2|             1|              1|              2|    0.2|
# |[orange, banana, ...|   1|[orange, banana]|   [bread]|       3|             2|              2|              2|    0.2|
# |     [banana, bread]|   1|        [banana]|   [bread]|       2|             1|              2|              2|    0.2|
# |             [bread]|   2|         [bread]|   [bread]|       1|             1|              2|              2|    0.4|
# |     [diapers, beer]|   1|       [diapers]|    [beer]|       2|             1|              1|              2|    0.2|
# |              [beer]|   2|          [beer]|    [beer]|       1|             1|              2|              2|    0.4|
# +--------------------+----+----------------+----------+--------+--------------+---------------+---------------+-------+
```

### Confidence

```python
arules.confidence().show()
# +--------------------+----+----------------+----------+--------+--------------+---------------+---------------+----------+
# |               items|freq|      antecedent|consequent|rule_len|antecedent_len|antecedent_freq|consequent_freq|confidence|
# +--------------------+----+----------------+----------+--------+--------------+---------------+---------------+----------+
# |            [orange]|   2|        [orange]|  [orange]|       1|             1|              2|              2|      null|
# |     [orange, apple]|   1|        [orange]|   [apple]|       2|             1|              2|              3|       0.5|
# |             [apple]|   3|         [apple]|   [apple]|       1|             1|              3|              3|      null|
# |      [juice, apple]|   1|         [juice]|   [apple]|       2|             1|              1|              3|       1.0|
# |[orange, banana, ...|   1|[orange, banana]|   [apple]|       3|             2|              2|              3|       0.5|
# |     [banana, apple]|   1|        [banana]|   [apple]|       2|             1|              2|              3|       0.5|
# |      [bread, apple]|   1|         [bread]|   [apple]|       2|             1|              2|              3|       0.5|
# |       [beer, apple]|   1|          [beer]|   [apple]|       2|             1|              2|              3|       0.5|
# |[juice, bread, ap...|   1|  [juice, bread]|   [apple]|       3|             2|              1|              3|       1.0|
# |             [juice]|   1|         [juice]|   [juice]|       1|             1|              1|              1|      null|
# |           [diapers]|   1|       [diapers]| [diapers]|       1|             1|              1|              1|      null|
# |    [orange, banana]|   2|        [orange]|  [banana]|       2|             1|              2|              2|       1.0|
# |            [banana]|   2|        [banana]|  [banana]|       1|             1|              2|              2|      null|
# |     [orange, bread]|   1|        [orange]|   [bread]|       2|             1|              2|              2|       0.5|
# |      [juice, bread]|   1|         [juice]|   [bread]|       2|             1|              1|              2|       1.0|
# |[orange, banana, ...|   1|[orange, banana]|   [bread]|       3|             2|              2|              2|       0.5|
# |     [banana, bread]|   1|        [banana]|   [bread]|       2|             1|              2|              2|       0.5|
# |             [bread]|   2|         [bread]|   [bread]|       1|             1|              2|              2|      null|
# |     [diapers, beer]|   1|       [diapers]|    [beer]|       2|             1|              1|              2|       1.0|
# |              [beer]|   2|          [beer]|    [beer]|       1|             1|              2|              2|      null|
# +--------------------+----+----------------+----------+--------+--------------+---------------+---------------+----------+
```

### Lift

```python
arules.lift().show()
# +--------------------+----+----------------+----------+--------+--------------+---------------+---------------+------------------+
# |               items|freq|      antecedent|consequent|rule_len|antecedent_len|antecedent_freq|consequent_freq|              lift|
# +--------------------+----+----------------+----------+--------+--------------+---------------+---------------+------------------+
# |            [orange]|   2|        [orange]|  [orange]|       1|             1|              2|              2|              null|
# |     [orange, apple]|   1|        [orange]|   [apple]|       2|             1|              2|              3|0.8333333333333334|
# |             [apple]|   3|         [apple]|   [apple]|       1|             1|              3|              3|              null|
# |      [juice, apple]|   1|         [juice]|   [apple]|       2|             1|              1|              3|1.6666666666666667|
# |[orange, banana, ...|   1|[orange, banana]|   [apple]|       3|             2|              2|              3|0.8333333333333334|
# |     [banana, apple]|   1|        [banana]|   [apple]|       2|             1|              2|              3|0.8333333333333334|
# |      [bread, apple]|   1|         [bread]|   [apple]|       2|             1|              2|              3|0.8333333333333334|
# |       [beer, apple]|   1|          [beer]|   [apple]|       2|             1|              2|              3|0.8333333333333334|
# |[juice, bread, ap...|   1|  [juice, bread]|   [apple]|       3|             2|              1|              3|1.6666666666666667|
# |             [juice]|   1|         [juice]|   [juice]|       1|             1|              1|              1|              null|
# |           [diapers]|   1|       [diapers]| [diapers]|       1|             1|              1|              1|              null|
# |    [orange, banana]|   2|        [orange]|  [banana]|       2|             1|              2|              2|               2.5|
# |            [banana]|   2|        [banana]|  [banana]|       1|             1|              2|              2|              null|
# |     [orange, bread]|   1|        [orange]|   [bread]|       2|             1|              2|              2|              1.25|
# |      [juice, bread]|   1|         [juice]|   [bread]|       2|             1|              1|              2|               2.5|
# |[orange, banana, ...|   1|[orange, banana]|   [bread]|       3|             2|              2|              2|              1.25|
# |     [banana, bread]|   1|        [banana]|   [bread]|       2|             1|              2|              2|              1.25|
# |             [bread]|   2|         [bread]|   [bread]|       1|             1|              2|              2|              null|
# |     [diapers, beer]|   1|       [diapers]|    [beer]|       2|             1|              1|              2|               2.5|
# |              [beer]|   2|          [beer]|    [beer]|       1|             1|              2|              2|              null|
# +--------------------+----+----------------+----------+--------+--------------+---------------+---------------+------------------+
```

### Conviction

```python
arules.conviction.show()
# +--------------------+----+----------------+----------+--------+--------------+---------------+---------------+----------+
# |               items|freq|      antecedent|consequent|rule_len|antecedent_len|antecedent_freq|consequent_freq|conviction|
# +--------------------+----+----------------+----------+--------+--------------+---------------+---------------+----------+
# |            [orange]|   2|        [orange]|  [orange]|       1|             1|              2|              2|      null|
# |     [orange, apple]|   1|        [orange]|   [apple]|       2|             1|              2|              3|       0.8|
# |             [apple]|   3|         [apple]|   [apple]|       1|             1|              3|              3|      null|
# |      [juice, apple]|   1|         [juice]|   [apple]|       2|             1|              1|              3|      null|
# |[orange, banana, ...|   1|[orange, banana]|   [apple]|       3|             2|              2|              3|       0.8|
# |     [banana, apple]|   1|        [banana]|   [apple]|       2|             1|              2|              3|       0.8|
# |      [bread, apple]|   1|         [bread]|   [apple]|       2|             1|              2|              3|       0.8|
# |       [beer, apple]|   1|          [beer]|   [apple]|       2|             1|              2|              3|       0.8|
# |[juice, bread, ap...|   1|  [juice, bread]|   [apple]|       3|             2|              1|              3|      null|
# |             [juice]|   1|         [juice]|   [juice]|       1|             1|              1|              1|      null|
# |           [diapers]|   1|       [diapers]| [diapers]|       1|             1|              1|              1|      null|
# |    [orange, banana]|   2|        [orange]|  [banana]|       2|             1|              2|              2|      null|
# |            [banana]|   2|        [banana]|  [banana]|       1|             1|              2|              2|      null|
# |     [orange, bread]|   1|        [orange]|   [bread]|       2|             1|              2|              2|       1.2|
# |      [juice, bread]|   1|         [juice]|   [bread]|       2|             1|              1|              2|      null|
# |[orange, banana, ...|   1|[orange, banana]|   [bread]|       3|             2|              2|              2|       1.2|
# |     [banana, bread]|   1|        [banana]|   [bread]|       2|             1|              2|              2|       1.2|
# |             [bread]|   2|         [bread]|   [bread]|       1|             1|              2|              2|      null|
# |     [diapers, beer]|   1|       [diapers]|    [beer]|       2|             1|              1|              2|      null|
# |              [beer]|   2|          [beer]|    [beer]|       1|             1|              2|              2|      null|
# +--------------------+----+----------------+----------+--------+--------------+---------------+---------------+----------+
```

### Calculate *all* currently-supported rules

```python
arules.calculate_all().show()
# +--------------------+----+----------------+----------+--------+--------------+---------------+---------------+-------+----------+------------------+----------+
# |               items|freq|      antecedent|consequent|rule_len|antecedent_len|antecedent_freq|consequent_freq|support|confidence|              lift|conviction|
# +--------------------+----+----------------+----------+--------+--------------+---------------+---------------+-------+----------+------------------+----------+
# |            [orange]|   2|        [orange]|  [orange]|       1|             1|              2|              2|    0.4|      null|              null|      null|
# |     [orange, apple]|   1|        [orange]|   [apple]|       2|             1|              2|              3|    0.2|       0.5|0.8333333333333334|       0.8|
# |             [apple]|   3|         [apple]|   [apple]|       1|             1|              3|              3|    0.6|      null|              null|      null|
# |      [juice, apple]|   1|         [juice]|   [apple]|       2|             1|              1|              3|    0.2|       1.0|1.6666666666666667|      null|
# |[orange, banana, ...|   1|[orange, banana]|   [apple]|       3|             2|              2|              3|    0.2|       0.5|0.8333333333333334|       0.8|
# |     [banana, apple]|   1|        [banana]|   [apple]|       2|             1|              2|              3|    0.2|       0.5|0.8333333333333334|       0.8|
# |      [bread, apple]|   1|         [bread]|   [apple]|       2|             1|              2|              3|    0.2|       0.5|0.8333333333333334|       0.8|
# |       [beer, apple]|   1|          [beer]|   [apple]|       2|             1|              2|              3|    0.2|       0.5|0.8333333333333334|       0.8|
# |[juice, bread, ap...|   1|  [juice, bread]|   [apple]|       3|             2|              1|              3|    0.2|       1.0|1.6666666666666667|      null|
# |             [juice]|   1|         [juice]|   [juice]|       1|             1|              1|              1|    0.2|      null|              null|      null|
# |           [diapers]|   1|       [diapers]| [diapers]|       1|             1|              1|              1|    0.2|      null|              null|      null|
# |    [orange, banana]|   2|        [orange]|  [banana]|       2|             1|              2|              2|    0.4|       1.0|               2.5|      null|
# |            [banana]|   2|        [banana]|  [banana]|       1|             1|              2|              2|    0.4|      null|              null|      null|
# |     [orange, bread]|   1|        [orange]|   [bread]|       2|             1|              2|              2|    0.2|       0.5|              1.25|       1.2|
# |      [juice, bread]|   1|         [juice]|   [bread]|       2|             1|              1|              2|    0.2|       1.0|               2.5|      null|
# |[orange, banana, ...|   1|[orange, banana]|   [bread]|       3|             2|              2|              2|    0.2|       0.5|              1.25|       1.2|
# |     [banana, bread]|   1|        [banana]|   [bread]|       2|             1|              2|              2|    0.2|       0.5|              1.25|       1.2|
# |             [bread]|   2|         [bread]|   [bread]|       1|             1|              2|              2|    0.4|      null|              null|      null|
# |     [diapers, beer]|   1|       [diapers]|    [beer]|       2|             1|              1|              2|    0.2|       1.0|               2.5|      null|
# |              [beer]|   2|          [beer]|    [beer]|       1|             1|              2|              2|    0.4|      null|              null|      null|
# +--------------------+----+----------------+----------+--------+--------------+---------------+---------------+-------+----------+------------------+----------+
```
## Testing

To test the calculations, run the following command (you may have to change things a bit to accommodate your particular Hadoop environment):

```
spark-submit \
	--master yarn \
	--deploy-mode client \
	--conf spark.pyspark.driver.python=/hadoop/opt/python_anaconda/3-5.1.0/bin/python \
	--conf spark.pyspark.python=/hadoop/opt/python_anaconda/3-5.1.0/bin/python \
	--conf spark.executorEnv.PYTHONHASHSEED=321 \
	--num-executors 5 \
	--executor-memory 2G \
	--executor-cores 2 \
	--driver-memory 1G \
	--py-files arules.py \
	test.py
```

## Contributing

Pull requests are welcome, especially those proposing new association rule metrics.
