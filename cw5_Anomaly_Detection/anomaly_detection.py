# anomaly_detection.py
import pandas as pd
import numpy as np
import statistics
from sklearn.preprocessing import StandardScaler

class AnomalyDetection():

    def scaleNum(self, df, indices):
        """
            Write your code!
        """
        def scale_num(single_list_features, which_col,mean,std):
            single_list_features[which_col] = float(single_list_features[which_col])-mean
            single_list_features[which_col] = single_list_features[which_col]/std
            return single_list_features

        def get_mean_std(df_features, which_col):
            # Function to get mean and std of single specified column
            num_total_list = len(df_features)
            temp_list = []

            for i in range(num_total_list):
                temp_list.append(df_features[i][which_col])
            # print("Unique elements in list:",set(temp_list))

            #Cast int to every element in list
            temp_list =[float(x) for x in temp_list]

            # Find mean
            mean = sum(temp_list)/num_total_list

            # Find std (there are 2 ways: np.std(), statistics.stdev())
            # std = np.std(temp_list)
            std =statistics.stdev(temp_list)
            print("mean: ",mean, " ---- std: ",std)

            # These 2 lines of Code to check with built-in StandardScaler function
            # NOTE!!! StandardScaler() uses np.std() while our assignment uses statistics.stdev()
            # sc = StandardScaler()
            # print(sc.fit_transform(np.asarray(temp_list).reshape(-1,1)), " --------------")

            return mean,std

        num_col_for_scaleNum = len(indices)
        for index_col in range(num_col_for_scaleNum):
            # Scaling function on each individual column per each loop
            mean, std = get_mean_std(df['features'],indices[index_col])
            df_result = df['features'].apply(scale_num,args=[indices[index_col],mean,std])

        return pd.DataFrame(df_result, columns=['features']).rename_axis('id')

    def cat2Num(self, df, indices):
        """
            Write your code!
            # Note: this function only works for this assignment where col 0,1 is categorical features,
                    if we want more general form, I need to adjust some code (column code)
        """

        def df_categorical_to_num(single_elem,unique_list):
            # Function to do One-hot-encoding
            num_distinct = len(unique_list)
            one_hot_encoding = [0]*num_distinct
            for i in range(num_distinct):
                if single_elem == unique_list[i]:
                    one_hot_encoding[i] = 1
                    return one_hot_encoding
            #in case can't find any match, return list of 0
            return [0]*num_distinct

        # We first extract non-categorical column dataframe
        INDEX_START_NON_CATEGORICAL = 2
        df_without_categorical_col = df['features'].apply(lambda x: x[INDEX_START_NON_CATEGORICAL:])

        # Get categorical columns
        series_categorical_col = df['features'].apply(lambda x: x[0:2])
        # Append list in every column into one big list of row list
        list_series_categorical_col= series_categorical_col.values.tolist()
        df_categorical = pd.DataFrame(list_series_categorical_col,columns=indices)
        for i in range(len(indices)):
            # Find unique element in each column
            unique_elem = df_categorical[i].unique()
            # Turn categorical element to one_hot_encoding element [Column by Column]
            df_categorical[i] = df_categorical[i].apply(df_categorical_to_num,args=[unique_elem])

        # Join and rename: 2 df (categorical df with one_hot_encoding and non-categorical df)
        df_result = pd.concat([df_categorical,df_without_categorical_col],axis=1)
        df_result.rename(columns={'features':'non_cat_features'}, inplace=True)

        # Join transformed categorical col with non-categorical column dataframe
        df_result['features'] = df_result[0]+df_result[1]+df_result['non_cat_features']
        df_result.drop([0,1,'non_cat_features'],axis=1,inplace=True)
        return df_result


    def detect(self, df, k, t):
        """
            Write your code!
        """
        from sklearn.cluster import KMeans

        def calculate_score(single_elem,frequency_map,large_N,small_N):
            for i in range(len(frequency_map.index)):
                cluster=frequency_map.iloc[i]['index_col']
                if single_elem == cluster:
                    score_nom = large_N-frequency_map.iloc[i]['cluster_size']
                    score_denom = large_N-small_N
                    return  score_nom/score_denom

        kmeans = KMeans(n_clusters=k)
        list_series= df.features.values.tolist()
        np_features = np.array(list_series)
        kmeans.fit(np_features)

        cluster_map = pd.DataFrame()
        # Dataframe to assign each row to different clusters
        cluster_map['data_index'] = df.index.values
        cluster_map['cluster'] = kmeans.labels_

        # Dataframe and Function to calculate Frequency
        count_frequency = pd.DataFrame(cluster_map['cluster'].value_counts())
        count_frequency.rename(columns={'cluster':'cluster_size'}, inplace=True)
        count_frequency['index_col'] = count_frequency.index

        largest_cluster_size = count_frequency.iloc[0]['cluster_size']
        smallest_cluster_size = count_frequency.iloc[-1]['cluster_size']
        print("largest_cluster_size: ",largest_cluster_size, " -- smallest_cluster_size: ",smallest_cluster_size)

        # Apply function to get Score
        cluster_map['score'] = cluster_map['cluster'].apply(calculate_score,args=[count_frequency,largest_cluster_size,smallest_cluster_size])
        print(cluster_map)

        df = pd.concat([df,cluster_map['score']],axis=1).rename_axis('id')
        print(df)

        # Remove rows with score less than threshold
        df_final = df[df.score >= t]
        return df_final

if __name__ == "__main__":
    # For logs-features-sample.csv
    df = pd.read_csv('logs-features-sample.csv').set_index('id')
    df['features'] = df['features'].astype(str)
    #Turn elements in Features column from String to List
    df['features'] = df.features.apply(lambda x: x.strip('[]').split(','))

    # For matrix exmaple (Comment this part to use logs-features-sample.csv)
    data = [(0, ["http", "udt", 4]), \
            (1, ["http", "udf", 5]), \
            (2, ["http", "tcp", 5]), \
            (3, ["ftp", "icmp", 1]), \
            (4, ["http", "tcp", 4])]
    df = pd.DataFrame(data=data, columns = ["id", "features"])


    ad = AnomalyDetection()

    df1 = ad.cat2Num(df, [0,1])
    print(df1, '\n')

    df2 = ad.scaleNum(df1, [6])
    print(df2, '\n')

    # df3 = ad.detect(df2, 8, 0.97)
    df3 = ad.detect(df2, 2, 0.9)
    print(df3)
