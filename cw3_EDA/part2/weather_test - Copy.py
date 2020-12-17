import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('weather test').getOrCreate()
assert spark.version >= '2.3' # make sure we have Spark 2.3+
spark.sparkContext.setLogLevel('WARN')
from pyspark.sql import functions as fn
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import SQLTransformer
import elevation_grid as eg
import numpy as np
import pandas as pd
from datetime import date
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt


tmax_schema = types.StructType([
    types.StructField('station', types.StringType()),
    types.StructField('date', types.DateType()),
    types.StructField('latitude', types.FloatType()),
    types.StructField('longitude', types.FloatType()),
    types.StructField('elevation', types.FloatType()),
    types.StructField('tmax', types.FloatType()),
])

USE_YTD_TEMP_FEATURE = False

def test_model(model_file, inputs):
    # get the data
    test_tmax = spark.read.csv(inputs, schema=tmax_schema)
    #########################################################################
    if USE_YTD_TEMP_FEATURE:
        syntax = """SELECT today.latitude,today.longitude,today.elevation,today.date,
                           today.tmax, yesterday.tmax AS yesterday_tmax
                    FROM __THIS__ as today
                    INNER JOIN __THIS__ as yesterday
                    ON date_sub(today.date, 1) = yesterday.date
                       AND today.station = yesterday.station"""
        sql_trans = SQLTransformer(statement= syntax)
        test_tmax = sql_trans.transform(test_tmax)
    #######################################################################
    test_tmax = test_tmax.withColumn('day_of_year', fn.dayofyear('date'))
    # load the model
    model = PipelineModel.load(model_file)

    # -------------------------------------------------------------------------------------------------
    '''#################################################################################'''
    '''########## B1 plot the Temperature Heatmap from trained model ####################'''
    '''##################################################################################'''
    lats, lons = np.meshgrid(np.arange(-90,90,.5),np.arange(-180,180,.5))
    elevs = [eg.get_elevations(np.array([late,lone]).T) for late,lone in zip(lats,lons)]

    num_row = lats.shape[0]
    num_col = lats.shape[1]
    total_pixel = num_row * num_col

    # Col = 3 because of 'latitude,longitude,elevation'
    grid_lats_lons_elev = np.zeros(shape=(total_pixel,3))

    print(grid_lats_lons_elev.shape)

    index_row_grid = 0
    for i in range(num_row):
        for j in range(num_col):
            grid_lats_lons_elev[index_row_grid] =  np.array([lats[i][j], lons[i][j], elevs[i][j]])
            index_row_grid += 1

    df_lats_lons_elev = pd.DataFrame(grid_lats_lons_elev, columns=['latitude', 'longitude', 'elevation'])

    # Assume the simulated data comes from today
    df_date = pd.DataFrame(np.arange(total_pixel), columns=['date'])
    df_date['date'] = date.today()

    df_final = pd.concat([ df_date, df_lats_lons_elev], axis=1)
    print(df_final)

    simulated_tmax_schema = types.StructType([
        types.StructField('date', types.DateType()),
        types.StructField('latitude', types.FloatType()),
        types.StructField('longitude', types.FloatType()),
        types.StructField('elevation', types.FloatType())
    ])

    df_simulated_tmax = spark.createDataFrame(df_final, schema = simulated_tmax_schema)
    df_simulated_tmax = df_simulated_tmax.withColumn('day_of_year', fn.dayofyear('date'))

    predictions = model.transform(df_simulated_tmax)

    print(predictions.show())

    df_predictions = predictions.toPandas()
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    geometry = [Point(xy) for xy in zip(df_predictions['longitude'], df_predictions['latitude'])]

    df_predictions = df_predictions.drop(['longitude', 'latitude'], axis=1)
    crs = {'init': 'epsg:4326'}
    gdf = GeoDataFrame(df_predictions, crs=crs, geometry=geometry)

    base = gdf.plot(column='prediction', marker='o',cmap='jet', markersize=5,legend=True,legend_kwds={'label': "Temperature in Celcius",'orientation': "horizontal"})
    world.boundary.plot(ax=base,edgecolor='black')

    plt.title('Predicted Temperature of Jan 2020')
    # plt.show()
    plt.savefig("heatmap")
    plt.close()
    ''' ####################---- END of B1 ----###################################### '''
    ''' ############################################################################# '''
    #---------------------------------------------------------------------------------------------------------------


    '''#################################################################################'''
    '''########## B2 plot the Error Distribution of Temperature  ########################'''
    '''##################################################################################'''
    # use the model to make predictions
    predictions = model.transform(test_tmax)
    predictions = predictions.withColumn('error', predictions['prediction']-predictions['tmax'])
    df_long_lat = predictions.toPandas()
    predictions.show()

    geometry = [Point(xy) for xy in zip(df_long_lat['longitude'], df_long_lat['latitude'])]
    df_long_lat = df_long_lat.drop(['longitude', 'latitude'], axis=1)
    crs = {'init': 'epsg:4326'}
    gdf = GeoDataFrame(df_long_lat, crs=crs, geometry=geometry)

    base = world.plot(color='white',edgecolor='black',figsize=(20, 12))
    gdf.plot(column='error',ax=base, marker='o',cmap='jet', markersize=15,legend=True,legend_kwds={'label': "Error of Temperature in Celcius",'orientation': "horizontal"})
    plt.title('Distribution of Temperature Prediction Error')
    # plt.show()
    plt.savefig('dist_temp_error')



    '''
    # evaluate the predictions
    r2_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='tmax',
            metricName='r2')
    r2 = r2_evaluator.evaluate(predictions)

    rmse_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='tmax',
            metricName='rmse')
    rmse = rmse_evaluator.evaluate(predictions)

    print('r2 =', r2)
    print('rmse =', rmse)
    # If you used a regressor that gives .featureImportances, maybe have a look...
    print(model.stages[-1].featureImportances)
    '''

if __name__ == '__main__':
    model_file = sys.argv[1]
    inputs = sys.argv[2]
    test_model(model_file, inputs)
