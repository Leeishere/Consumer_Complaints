

import zipfile
import os
import pathlib
from pathlib import Path


import pandas as pd
import streamlit as st

#================================================================================================================================
# session state
#================================================================================================================================
if 'filepath' not in st.session_state:
    st.session_state.filepath = None
if "files_list" not in st.session_state:
    st.session_state.files_list = None
if 'isfile_isdir' not in st.session_state:
    st.session_state.isfile_isdir = None
if 'path_to_selected_file' not in st.session_state:
    st.session_state.path_to_selected_file = None
if 'selected_dataframe' not in st.session_state:
    st.session_state.selected_dataframe = None
if 'functions_available' not in st.session_state:
    st.session_state.functions_available = ['interactive_true_false_column']
if 'curr_function' not in st.session_state:
    st.session_state.curr_function = None
if 'current_checkbox_column' not in st.session_state:
    st.session_state.current_checkbox_column = None
if 'load_saved' not in st.session_state:
    st.session_state.load_saved = False
if 'feature_thresholds' not in st.session_state:
    st.session_state.feature_thresholds = {}
if 'curr_feature' not in st.session_state:
    st.session_state.curr_feature = None

#================================================================================================================================
# file handling functions
#================================================================================================================================
# compute the similarities on the fly
def fetch_zipped_data(filepath):
    """
    """
    df = None
    filepath = Path(filepath)

    with zipfile.ZipFile(filepath) as myzip:
        contents = myzip.namelist()

        selected_files = st.multiselect( "Select the correct file", contents, default=contents[:1], key="extract_file_from_zip" )

        if selected_files:
            filename = selected_files[0]
            with myzip.open(filename) as f:
                df = pd.read_csv(f)
    return df


def list_of_file_s(filepath):
    """
    returns a list of filepaths that are in the folder, or puts the file filepath into a one element list and returns it
    """
    p=pathlib.Path(filepath)
    if p.suffix=='':
        return sorted(p.glob('*'))
    return [p]

def get_valid_filepath(filepath:str='./similarity_dfs'):
    """
    accepts a string such as '/utils/file.csv'
    returns a path that works on whatever operating system is running it
    """
    filepath=filepath.lstrip('.')
    filepath=filepath.lstrip('/')
    filepath='./'+filepath
    return pathlib.Path(filepath)



# similarity frames are stored in the filesystem
def fetch_similarity_data(filepath):
    """
    takes filpath to a .csv file
    """
    return pd.read_csv(filepath)


# case if user is asked for the filepath 
# not presently supported
def is_valid_user_input(filepath):
     return (pathlib.Path(filepath).exists() and (pathlib.Path(filepath).is_file()) or pathlib.Path(filepath).is_dir())

def is_file_or_is_dir(filepath):
    return "file" if pathlib.Path(filepath).is_file() else "folder" if pathlib.Path(filepath).is_dir() else "neither"



#================================================================================================================================
# load data
#================================================================================================================================
#get_file, select_annotation_function, load_default_or_previous_save , submit_selected_annotation_function = st.columns([.4,.4,.2,.2],  gap="small", vertical_alignment="top", border=True)   

with st.expander('file loader'):
    #with get_file:
    if st.session_state.files_list==None:
        st.session_state.filepath=get_valid_filepath() #----------#----------#----------#----------#----------#----------# filepath
        st.session_state.isfile_isdir=is_file_or_is_dir(st.session_state.filepath)
        if st.session_state.isfile_isdir=='neither':
            raise ValueError(f"The file path must be a file or a folder containing valid files")


    if st.session_state.isfile_isdir=='file':
        st.session_state.path_to_selected_file=st.session_state.filepath
    elif st.session_state.isfile_isdir=='folder':
        st.session_state.files_list=list_of_file_s(st.session_state.filepath)
        file=st.selectbox('Select a file',st.session_state.files_list,index=0,key='select_file')
        st.session_state.path_to_selected_file=pathlib.Path(file)#pathlib.Path(st.session_state.filepath)/file
    
    st.session_state.curr_feature = ''.join(''.join(str(st.session_state.path_to_selected_file).split('/')[1:]).split('.')[:-1])

    if (st.session_state.files_list is not None) or (st.session_state.path_to_selected_file is not None):
        st.session_state.selected_dataframe=fetch_similarity_data(st.session_state.path_to_selected_file)

    # select an annotation function
    #with select_annotation_function:
    selected_function = st.selectbox("Select an Annotation Function", st.session_state.functions_available, key='select_a_function_for_data_loading' , label_visibility="visible", width="stretch")
    #with load_default_or_previous_save:
    #where left if False and right is True
    st.session_state.load_saved = st.toggle("Load saved version, or reset to T/F threshold.", value=False, key='load_saved_or_default_toggle', label_visibility="visible")
    #with submit_selected_annotation_function:
    submit_function = st.button('Submit', key='retrieve_interactive_dataframe', type="secondary")
    if submit_function:
        st.session_state.curr_function = selected_function



#================================================================================================================================
# display an editable T/F dataframe
#================================================================================================================================

# the function referenced in st.session_state.functions_available
def true_false_checkbox_column(dataframe,threshold:float=0.85,upper_is_true:bool=True,similarity_col:str='similarity_score',true_false_col:str='is_similar',reset:bool=False):
    """
    where similarity_col is the column containing similarity scores
    takes a dataframe as input and adds a bool column of checkboxes
    if upper_is_true==True: threshold is max value for a default False value in the new column, else it is the max value for True
    returns a dataframe with an added column of interactive checkboxes and the name of the added column 
    WARNING: if reset==True, any column with a header == true_false_col parameter will be replaced with the new column, else it returns the dataframe unchanged
    """
    original_columns=list(dataframe.columns)
    if reset==False and (true_false_col in original_columns):
        return dataframe, true_false_col
    elif reset==True and (true_false_col in original_columns):
        dataframe = dataframe.drop(columns=[true_false_col])
        original_columns = list(dataframe.columns) 
    if upper_is_true==True:
        dataframe[true_false_col]=True   
        dataframe.loc[dataframe[similarity_col]<=threshold,true_false_col]=False
    else:
        dataframe[true_false_col]=False   
        dataframe.loc[dataframe[similarity_col]<=threshold,true_false_col]=True       
    return dataframe[[true_false_col]+original_columns], true_false_col




if st.session_state.selected_dataframe is not None and st.session_state.curr_function == 'interactive_true_false_column': 
    #interactive_true_false_column_editable_df_display = st.columns([1],  gap="small", vertical_alignment="top", border=True) 

    #interactive_true_false_column_editable_df_display=interactive_true_false_column_editable_df_display[0]
    #with interactive_true_false_column_editable_df_display:
    # where toggle left==False, right==True: toggle -> "Load saved version, or reset to T/F threshold." -> st.session_state.load_saved
    if st.session_state.load_saved == True:
        curr_threshold = st.session_state.feature_thresholds.get(st.session_state.curr_feature,None)
        if curr_threshold is not None:
            st.session_state.selected_dataframe, true_false_col = true_false_checkbox_column(st.session_state.selected_dataframe,threshold= curr_threshold , reset=True)
        else:
            st.session_state.selected_dataframe, true_false_col = true_false_checkbox_column(st.session_state.selected_dataframe,threshold= curr_threshold , reset=True)  
    else:
        st.session_state.selected_dataframe, true_false_col = true_false_checkbox_column(st.session_state.selected_dataframe, reset=False) 

    edited_dataframe = st.data_editor(
        st.session_state.selected_dataframe,
        column_config={
            true_false_col: st.column_config.CheckboxColumn(
                label=None,
                help="Checked Boxes Should be Similar. Uncheck Should be Dissimilar",
                default=False,
                        )
                        },
        disabled=[col for col in st.session_state.selected_dataframe if col != true_false_col],
        hide_index=False,
        )
    
 
submit_annotations , update_feature_threshold, submit_threshold= st.columns([.33,.33,.34] , gap="small", vertical_alignment="top", border=True)  
with submit_annotations:
    st.markdown('Submitting will overwrite the file')
    submit_the_annotated_df = st.button('Submit File', key='submit_and_save_data', type="secondary")
    if submit_the_annotated_df:
        edited_dataframe.to_csv(st.session_state.path_to_selected_file, index=False, encoding="utf-8")

if st.session_state.curr_feature is not None:
    with update_feature_threshold:
        input_float = st.number_input("T/F Threshold", min_value=0.000, max_value=1.000, value=0.850,  key='set_threshold')
    if input_float:
        input_float=float(input_float)
        with submit_threshold:
            submit_threshold = st.button('Submit Threshold')
            st.markdown('Toggle T/F threshold.')
            if submit_threshold:
                st.session_state.feature_thresholds[st.session_state.curr_feature]=input_float