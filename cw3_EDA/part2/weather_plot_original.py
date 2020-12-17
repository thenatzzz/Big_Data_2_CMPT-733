import sys
import matplotlib; matplotlib.use('Agg') # don't fail when on headless server
import matplotlib.pyplot as plt
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, types
from pyspark.sql import functions as fn
spark = SparkSession.builder.appName('weather_plot').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
assert spark.version >= '2.4' # make sure we have Spark 2.4+

from pyspark.ml.feature import  SQLTransformer
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
import numpy as np

tmax_schema = types.StructType([
    types.StructField('station', types.StringType()),
    types.StructField('date', types.DateType()),
    types.StructField('latitude', types.FloatType()),
    types.StructField('longitude', types.FloatType()),
    types.StructField('elevation', types.FloatType()),
    types.StructField('tmax', types.FloatType()),
])

USE_YTD_TEMP_FEATURE = True

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
        df = sql_trans.transform(data)
    #############################################################################
    df = data.withColumn('day_of_year', fn.dayofyear('date'))
    df = df.withColumn('year', fn.year('date'))

    df_long_lat = df[['station','longitude','latitude','tmax','year']].toPandas()
    count_year = df_long_lat['year'].value_counts().to_dict()


    YEAR_SELECTED = 1900
    YEAR_DURATION = 120
    df_long_lat= df_long_lat.loc[(df_long_lat['year'] > YEAR_SELECTED) & (df_long_lat['year'] < YEAR_SELECTED+YEAR_DURATION)]


    df_long_lat['avg_temp'] = df_long_lat.groupby(['station','year'])['tmax'].transform('mean')
    df_long_lat.drop_duplicates(subset=['station','year'],inplace=True)
    print(df_long_lat)

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    geometry = [Point(xy) for xy in zip(df_long_lat['longitude'], df_long_lat['latitude'])]

    df_long_lat = df_long_lat.drop(['longitude', 'latitude'], axis=1)
    crs = {'init': 'epsg:4326'}
    gdf = GeoDataFrame(df_long_lat, crs=crs, geometry=geometry)

    base = world.plot(color='white',edgecolor='black',figsize=(20, 12))
    gdf.plot(column='avg_temp',ax=base, marker='o',cmap='jet', markersize=15,legend=True,legend_kwds={'label': "Temperature in Celcius",'orientation': "horizontal"})
    plt.title('Distribution of Temperature between '+ str(YEAR_SELECTED)+ " and "+str(YEAR_SELECTED+YEAR_DURATION))
    plt.savefig(inputs+"_"+str(YEAR_SELECTED)+"-"+str(YEAR_SELECTED+YEAR_DURATION))

if __name__ == '__main__':
    inputs = sys.argv[1]
    output_model = sys.argv[2]
    main(inputs,output_model)
