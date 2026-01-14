

import regex as re
import pandas as pd


class DateTime:
    def __init__(self):
        self.invalid_date_map = {}

    def detect_invalid_dates(self,dataframe:pd.DataFrame,columns_to_examine:str|list,yearfirst:bool=False):
        """
        presently this assumes yearfirst because that's what the consumer_complaints dataset used
        self.invalid_date_map should be called after running this func to retrieve the count of invalid dates
        """
        if yearfirst!=True:
            raise ValueError('presently this only supports yearfirst')
        if type(columns_to_examine)==str:
            columns_to_examine=[columns_to_examine]

        for col in columns_to_examine:            
            invalid_dates = ~dataframe[col].astype(str).str.match( r'^(19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$' )
            self.invalid_date_map[col]=dataframe.loc[invalid_dates].shape[0]

    def normalize_datetime(self,dataframe:pd.DataFrame,date_columns:list|str,yearfirst:bool=False,format:str|None=None):
        """
        presently this assumes yearfirst because that is what the consumer_complaints dataset uses
        returns a datafame with input columns revised to datetime
        """

        if yearfirst==False:
            raise ValueError('presently supports yearfirst')
        elif format is not None:
            raise ValueError('presently supports yearfirst')
        if type(date_columns)==str:
            date_columns=[date_columns]
        for col in date_columns:
            dataframe[col] = pd.to_datetime(dataframe[col],yearfirst=True, errors="coerce").dt.normalize()
        return dataframe

        
        

