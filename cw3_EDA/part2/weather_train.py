import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, types
from pyspark.sql import functions as fn
spark = SparkSession.builder.appName('weather_train').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
assert spark.version >= '2.4' # make sure we have Spark 2.4+

from pyspark.ml import Pipeline
from pyspark.ml.feature import  VectorAssembler,SQLTransformer
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

tmax_schema = types.StructType([
    types.StructField('station', types.StringType()),
    types.StructField('date', types.DateType()),
    types.StructField('latitude', types.FloatType()),
    types.StructField('longitude', types.FloatType()),
    types.StructField('elevation', types.FloatType()),
    types.StructField('tmax', types.FloatType()),
])

USE_YTD_TEMP_FEATURE = False

def main(inputs,model_file):
    data = spark.read.option('encoding','UTF-8').csv(inputs, schema=tmax_schema)
    ################ FEATURE ENGINEERING: add yesterday tmax #####################
    if USE_YTD_TEMP_FEATURE:
        syntax = """SELECT today.latitude,today.longitude,today.elevation,today.date,
                           today.tmax, yesterday.tmax AS yesterday_tmax
                    FROM __THIS__ as today
                    INNER JOIN __THIS__ as yesterday
                    ON date_sub(today.date, 1) = yesterday.date
                       AND today.station = yesterday.station"""
        sql_trans = SQLTransformer(statement= syntax)
        data = sql_trans.transform(data)
    #############################################################################
    data = data.withColumn('day_of_year', fn.dayofyear('date'))
    train, validation = data.randomSplit([0.75, 0.25])
    train = train.cache()
    validation = validation.cache()

    if USE_YTD_TEMP_FEATURE:
        train_feature_assembler = VectorAssembler(inputCols=['yesterday_tmax','day_of_year','latitude','longitude','elevation'],outputCol='features')
    else:
        train_feature_assembler = VectorAssembler(inputCols=['day_of_year','latitude','longitude','elevation'],outputCol='features')

    ############# DIFFERENT ML ALGORITHMS TO BE USED ####################
    # classifier = GeneralizedLinearRegression(featuresCol = 'features', labelCol='tmax' )
    # classifier = GBTRegressor( maxDepth=5,featuresCol = 'features', labelCol='tmax' )
    classifier = RandomForestRegressor(numTrees=7, maxDepth=8,featuresCol = 'features', labelCol='tmax' )
    #####################################################################

    train_pipeline = Pipeline(stages=[train_feature_assembler, classifier])
    weather_model = train_pipeline.fit(train)

    prediction = weather_model.transform(validation)
    # print(prediction.show())
    evaluator = RegressionEvaluator(predictionCol="prediction",labelCol='tmax',metricName='r2') #rmse
    score = evaluator.evaluate(prediction)
    print('Validation score for weather model: %g' % (score, ))

    weather_model.write().overwrite().save(model_file)

if __name__ == '__main__':
    inputs = sys.argv[1]
    output_model = sys.argv[2]
    main(inputs,output_model)
