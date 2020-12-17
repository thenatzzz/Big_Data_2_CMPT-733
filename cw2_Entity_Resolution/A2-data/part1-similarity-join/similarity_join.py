import re
import pandas as pd

class SimilarityJoin:
    def __init__(self, data_file1, data_file2):
        self.df1 = pd.read_csv(data_file1)
        self.df2 = pd.read_csv(data_file2)

    def preprocess_df(self, df, cols):
        """
            Write your code!
        """
        df[cols[0]] =df[cols[0]].fillna('')
        df[cols[1]] =df[cols[1]].fillna('')

        df['joinKey'] = df[cols[0]].astype(str)+" "+df[cols[1]].astype(str)
        df['joinKey'] = df['joinKey'].apply(lambda x: x.lower())
        df['joinKey'] = df['joinKey'].apply(lambda x:re.split(r'\W+',x))

        #Remove empty string from list
        df['joinKey'] = df['joinKey'].apply(lambda x:list(filter(None,x)) )

        return df

    def filtering(self, df1, df2):
        """
            Write your code!
        """

        df1_flatten = df1.explode('joinKey')
        df1_flatten = df1_flatten[['id','joinKey']].rename(columns={'id':'id1'})
        df2_flatten = df2.explode('joinKey')
        df2_flatten = df2_flatten[['id','joinKey']].rename(columns={'id':'id2'})

        can_df = df1_flatten.merge(df2_flatten,on='joinKey')
        # Remove duplicate
        can_df = can_df.drop_duplicates().reset_index(drop=True)

        df1 = df1.rename(columns={'id':'id1','joinKey':'joinKey1'})
        can_df = can_df.merge(df1,on='id1')
        can_df = can_df[['id1','id2','joinKey1']]

        df2 = df2.rename(columns={'id':'id2','joinKey':'joinKey2'})
        can_df = can_df.merge(df2,on='id2')
        can_df = can_df[['id1','joinKey1','id2','joinKey2']]
        # Remove duplicate again if there are left
        can_df=can_df.loc[can_df.astype(str).drop_duplicates().index]

#         can_df.to_csv('test.csv', encoding='utf-8', index=False)
        return can_df


    def verification(self, cand_df, threshold):
        """
            Write your code!
        """
        def jaccard_similarity(list1, list2):
            s1 = set(list1)
            s2 = set(list2)
            return len(s1.intersection(s2)) / len(s1.union(s2))

        # Copy all elements from can_df dataframe to new dataframe called result_df
        result_df = cand_df.copy()

        # Compute Jaccard Similarity row by row on Pandas dataframe and assign to new column called 'jaccard'
        result_df['jaccard'] = cand_df.apply(lambda x: jaccard_similarity(x['joinKey1'],x['joinKey2']),axis=1)

        result_df = result_df[result_df.jaccard >= threshold ]

        return result_df

    def evaluate(self, result, ground_truth):
        """
            Write your code!
        """
        tuple_result = [tuple(t) for t in result]
        tuple_ground_truth = [tuple(t) for t in ground_truth]

        # Calculate number of intersection between our result and ground_truth
        number_match = len(set(tuple_result) & set(tuple_ground_truth))

        precision = number_match/len(result)
        recall = number_match/len(ground_truth)
        f_score = (2*precision*recall)/(precision+recall)

        return (precision,recall,f_score)

    def jaccard_join(self, cols1, cols2, threshold):
        new_df1 = self.preprocess_df(self.df1, cols1)
        new_df2 = self.preprocess_df(self.df2, cols2)
        print ("Before filtering: %d pairs in total" %(self.df1.shape[0] *self.df2.shape[0]))
        cand_df = self.filtering(new_df1, new_df2)
        print ("After Filtering: %d pairs left" %(cand_df.shape[0]))

        result_df = self.verification(cand_df, threshold)
        print ("After Verification: %d similar pairs" %(result_df.shape[0]))

        return result_df


if __name__ == "__main__":
    er = SimilarityJoin("Amazon_sample.csv", "Google_sample.csv")
    amazon_cols = ["title", "manufacturer"]
    google_cols = ["name", "manufacturer"]
    result_df = er.jaccard_join(amazon_cols, google_cols, 0.5)

    result = result_df[['id1', 'id2']].values.tolist()
    ground_truth = pd.read_csv("Amazon_Google_perfectMapping_sample.csv").values.tolist()
    print ("(precision, recall, fmeasure) = ", er.evaluate(result, ground_truth))
