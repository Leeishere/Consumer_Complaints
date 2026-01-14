import pandas as pd
import numpy as np

import regex as re
import itertools

from collections import defaultdict

#!pip install -U transformers sentence-transformers #accelerate [where accelerate is for large self.models and multiple GPUs]
import transformers
import sentence_transformers
from sentence_transformers import SentenceTransformer
from IPython.display import display, clear_output

import warnings

import sys, subprocess, pathlib
from pathlib import Path






#==================================================================================================================================================================
# Not yet a class
# functions that process a column and looks for semantic similarity in every combination of unique values
# call get_similarity_comparison_df() to return a df with similarity_scores
# call get_user_annotation() to annotate in jupyter, but an app would be better: variable_point_similarities_annotator.py
#==================================================================================================================================================================


#THE self.model: https://huggingface.co/MongoDB/mdbr-leaf-mt/blob/main/README.md?code=true




class SemanticSentenceMatching:

    def __init__(self, model_name="MongoDB/mdbr-leaf-mt"):
        self.model = SentenceTransformer(model_name)

    # 3 functions used in get_similarity_comparison_df()
    # 1
    def get_memory_efficient_semantic_similarities_scores(self,
                                                        observation_point_values,
                                                        filter:None|float=0.8,
                                                        min_memory:bool=False):  # Rare edge case. Memory issues are not likely to arise here. They are likely to arise when embedding large texts
        """
        where observation_point_values is pd.Series.unique() 
        self.model is a transformer self.model to compare sentences such as from #https://huggingface.co/MongoDB/mdbr-leaf-mt/blob/main/README.md?code=true
        min_memory iterates over a generator object if True, else creates a tuple of tuples 
        testcases with 273 unique values and avg word count < 10-ish:  
            combos are held inside func as list and scores computed one-by-one
                filter=None, min_memory=False)-> 2m 51.5s  result shape = (36856, 3)
                filter=0.8, min_memory=False)->  2m 57.2s  result shape = (444, 3) 
            combose are called from a generator object and computed one-by-one
                filter=None, min_memory=True)->  6m 17.1s  result shape = (36856, 3)   
                filter=0.8, min_memory=True)->   6m 11.6s  result shape = (444, 3)   
        """
        individual_vals=observation_point_values[~pd.isna(observation_point_values)] 
        combos=itertools.combinations(individual_vals,2)
        if min_memory==False:
            pairs=list(combos)

        col_a,col_b,scores = [],[],[]

        while True:
            if min_memory==False:
                if len(pairs)<1:
                    break
                pair=pairs.pop()

            else:
                try:
                    pair=next(combos)
                except StopIteration:
                    break
            embedding_a=self.model.encode(pair[0])
            embedding_b=self.model.encode(pair[1])
            score=float(self.model.similarity(embedding_a,embedding_b)[0][0])
            if (filter is not None) and (score>filter):
                col_a.append(pair[0])
                col_b.append(pair[1])
                scores.append(score)
            elif filter is None:
                col_a.append(pair[0])
                col_b.append(pair[1])
                scores.append(score)
        return  round(pd.DataFrame({'column_a':col_a, 'column_b':col_b, 'similarity_score':scores}).sort_values(by='similarity_score',ascending=False).reset_index(drop=True),6)
    

    # 2
    def get_embedding_matrix(self,observation_point_values):
        """
        where observation_point_values is pd.Series.unique() 
        self.model is a transformer self.model to compare sentences such as from #https://huggingface.co/MongoDB/mdbr-leaf-mt/blob/main/README.md?code=true
        """
        embeddings=self.model.encode(observation_point_values)
        similarities=self.model.similarity(embeddings, embeddings)
        return similarities


    # 3
    def get_semantic_similarity_scores(self,similarity_matrix, observation_point_values):
        """
        where similarity matrix is an outer product of embeddings with ones in the diagonal 
        and observation_point_values are unique values from pd.series.unique()
        returns a dataframe
        """

        var1=[]
        var2=[]
        score=[]
        row=0
        while row < similarity_matrix.shape[0]-1:
            sim_list=similarity_matrix[row,row+1:].tolist()
            score+=sim_list
            var1+=[observation_point_values[row]]*len(sim_list)
            var2+=list(observation_point_values[row+1:])
            row+=1        
        semantic_scores=pd.DataFrame({'column_a':var1,'column_b':var2,'similarity_score':score}).sort_values(by='similarity_score',ascending=False).reset_index(drop=True)
        semantic_scores['similarity_score']=round(semantic_scores['similarity_score'],6)
        return semantic_scores

    #--------------------------------------------------------------------------------------------------------------------
    #a function that returns a dataframe of every unique pair combination and similarity score, it can be filtered 
    # it is used in create_similarity_cvs_dir()
    def get_similarity_comparison_df(self,pd_series_to_examine,filter:None|float=0.8,conserve_memory:bool=False,clean_text:bool=False):
        """
        this takes a pandas.Series() with values to compare and gets unique values inside this function
        the self.model call .encode() and .similarity() and returns an outer product with ones in the diagonal. A model such as from #https://huggingface.co/MongoDB/mdbr-leaf-mt/blob/main/README.md?code=true
        These are converted to a sorted dataframe with columns=['column_a','column_b','similarity_score']
        it is returned with scores rounded to 6 places
        if filter is not False, it should be a float that is the minimum similarity score to return. default is 0.8 
        conserve_memory:bool=False   ==> if true get_memory_efficient_semantic_similarities_scores() is called to compute scores one-by-one
        """
        if clean_text==True:
            pd_series_to_examine=pd_series_to_examine.astype(str).str.lower().str.replace(r"[^A-Za-z ]"," ",regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
        sentence_values=pd_series_to_examine.copy().dropna().unique()
        if conserve_memory==True:
            return self.get_memory_efficient_semantic_similarities_scores(sentence_values,
                                                        filter=filter,
                                                        min_memory=False)
        else:
            similarity_matrix=self.get_embedding_matrix(sentence_values)
            scores_df=self.get_semantic_similarity_scores(similarity_matrix, sentence_values)
            if filter==None:
                return scores_df
            else: 
                return scores_df.loc[scores_df['similarity_score']>filter].reset_index(drop=True)
            
    # a function that loads similarity df's into a dir
    # calls get_similarity_comparison_df()
    #
    def create_similarity_cvs_dir(self,dataframe:pd.DataFrame, columns:list|str, directory:str,filter=0.7):
        """
        retrieve similar, but not identical entries per columns
        where directory is where the csv files will be stored
        filter filters similarity scores
        """
        if type(columns)!=list:
            columns=[columns]
        directory=directory.lstrip('.')
        directory=directory.lstrip('/')
        directory='./'+directory
        dir_path = pathlib.Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        for col in columns:
            try:
                sim_df=self.get_similarity_comparison_df(dataframe[col],filter=filter)
                file=dir_path / f"{col}_similarity_annotation_df.csv"
                sim_df.to_csv(file,index=False)
            except Exception as e:
                warnings.warn(f"The '{col}' column didn't work for similarity matching: {e}", category=UserWarning)

    #----------------------------------------------------------------------------------------------------------
    # two functions that annotate a boolean column based on similarity scores in df's returned by get_similarity_comparison_df()
    # the first is lightweight and can be used in .ipynb, the second calls run on a streamlit app
    # 1
    # a function that takes user feedback and creates an annotation T/F is_similar column
    def get_user_annotation(self,similarity_df,return_filtered:bool=True,num_observations_to_display:int=20,default_false_max:float=0.85):
        """
        this uses basic python input() to collect indexes from the user
        Takes a similarity df and iterates through ten rows at a time by default: num_observations_to_display:int=10
        the user should type s then enter to move to the next ten
        or type the numeric index value associated with the row then enter for each dissimilar row
        return_filtered can be set to false to return both True and False similarity annotations, otherwise, it only return True similarities
        default_false_max:float=0.85 sets all obs with scores < float val to False. that makes the anotators job easier
        """
        similarity_df['is_similar']=True   #because it's filtered at 0.8 by default, so many will be True. Plus, the user is asked to negate
        similarity_df.loc[similarity_df['similarity_score']<=default_false_max,'is_similar']=False
        try:
            final_true_index=int(similarity_df.loc[similarity_df['similarity_score']>default_false_max].index[-1])
        except:
            final_true_index=int(similarity_df.index[0])  # creates an edge case that a later if condition checks when there is user input
        dissimilar_indexes=[]
        similar_indexes=[]
        i_index=0
        while True:
            curr_max_index=num_observations_to_display+i_index
            if curr_max_index>similarity_df.shape[0]:
                curr_max_index=similarity_df.shape[0]
            text="If any 'is_similar' is incorrect, enter those index number(s) on the left one-by-one. Pres ENTER 2 times for next dataframe section."
            next_df_partition_is_signaled=0
            while True:
                clear_output(wait=True)
                display_df=similarity_df.iloc[i_index:curr_max_index,:][[ 'is_similar','column_a', 'column_b', 'similarity_score']]
                display(display_df)#display the selection for the user
                user_input = input(text)
                if user_input!="":
                    if str(user_input).isnumeric() and int(user_input) in display_df.index:
                        user_input=int(user_input)
                        if user_input>final_true_index or (user_input==final_true_index and similarity_df.loc[user_input,'is_similar']==False):#where or catches edge case where there is no True by default
                            similar_indexes+=[user_input]  #change a false to a true
                        else: 
                            dissimilar_indexes+=[user_input]   #change a true to a false
                        text=f"✅ {user_input} recieved."
                    else:
                        text=f"❌  {user_input} is not valid."
                #break inner loop
                else:
                    next_df_partition_is_signaled+=1
                    if next_df_partition_is_signaled>=2:
                        next_df_partition_is_signaled=0 
                        break
                    else:
                        text=f"🔲 press enter again."
            #break outer loop
            if curr_max_index==similarity_df.shape[0]: break   # this value that was set in the first if
            i_index=curr_max_index

        similarity_df.loc[dissimilar_indexes,'is_similar']=False
        similarity_df.loc[similar_indexes,'is_similar']=True
        if return_filtered==False:
            return similarity_df
        else: return similarity_df.loc[similarity_df['is_similar']==True]
            
    # 2
    #load TrueFalse_Dataframe_Annotator.py
    def run_streamlit_annotator(self,path='./TrueFalse_Dataframe_Annotator.py',quit:bool=False):
        if quit==True:
            sys.exit()
        else:
            try:
                subprocess.run(["streamlit", "run", str(Path(path))])
            except:
                None
                #!streamlit run {pathlib.Path(path)}


    # -------------------------------------------------------------------------------------
    # 3 funcs to take data from the dir create_similarity_cvs_dir() put similarity df's into, and update the main dataframe
    # and one helper func

    # uses the same directory as create_similarity_cvs_dir()
    def zip_paths_and_files_by_extension(self,directory:str='./similarity_dfs', extension:str='csv'):
        """
        a function to extract filenames from files created by function: create_similarity_cvs_s() 
        WARNING: does not discriminate. Every file in the directory will be used to update the man dataframe
        """
        directory=directory.lstrip('.')
        directory=directory.lstrip('/')
        directory='./'+directory
        files = list(Path(directory).glob(f"*.{extension.lstrip('.')}"))
        features=[str(i)[15:][:-29] for i in files]
        return zip(features,files)
        # a helper function to use in .apply() inside: updata_feature_columns_based_on_similar_entries()
        


    #iterate through similarities dataframes and return a dict of similar variable groups
    def get_similar_entries(self,zipped_features_to_filepaths):
        """
        O(n) graph approach 
        the files should be csv, the boolean column should be titled "is_similar", features should be in 'column_a' and 'column_b'
        takes a list of similarity dataframes [is_similar, col_a, col_b,similarity_score] as zip(feature_column,filepath_to_entry_similarity_dataframes)
        and uses the boolean column 'is_similar' (derived by annotation functions) to return a dictionary of lists of sets for each similarity group
        returns dict such as {'feature':[[group 1],[group 2]], 'feature2':...}
        """

        similar_map = {}

        for feature_column, filepath in zipped_features_to_filepaths:
            df = pd.read_csv(filepath)
            df = df[df['is_similar']][['column_a', 'column_b']]

            graph = defaultdict(set)
            for a, b in zip(df['column_a'], df['column_b']):
                graph[a].add(b)
                graph[b].add(a)

            visited = set()
            groups = []

            for node in graph.keys():
                if node in visited:
                    continue

                stack = [node]
                component = set()

                while stack:
                    curr = stack.pop()
                    if curr in visited:
                        continue
                    visited.add(curr)
                    component.add(curr)
                    stack.extend(graph[curr] - visited)

                groups.append(list(component))

            similar_map[feature_column] = groups

        return similar_map








    def choose_group_representative(self,list_of_group_lists):
        """
        """
        if not list_of_group_lists: return None
        lookup_dict={}
        for list_of_vars in list_of_group_lists:
            if not list_of_vars:
                continue
            new_vals={var:list_of_vars[0] for var in list_of_vars}
            lookup_dict.update(new_vals)
        return lookup_dict


    # a funciton to update the dataframe
    def update_feature_columns_based_on_similar_entries(self,dataframe:pd.DataFrame, similar_entries_maps):
        """
        takes the main dataframe and the similar entries dict from function-> get_similar_entries()
        picks the first entry to use to replace all other entries
        """
        for column,groups in similar_entries_maps.items():
            mapping=self.choose_group_representative(groups)
            if mapping is None:
                continue
            dataframe[column]=dataframe[column].map(mapping).fillna(dataframe[column])
        return dataframe