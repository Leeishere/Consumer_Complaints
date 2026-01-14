

import pandas as pd
import regex as re
import unicodedata


# Mapping of common confusables to normalized form
TRANSLATION = str.maketrans({
    # Dashes
    "–": "-", "—": "-", "−": "-",

    # Quotes
    "‘": "'", "’": "'",
    "“": '"', "”": '"',

    # Spaces / invisible
    "\u00A0": " ",   # non-breaking space '\xa0'
    "\u200B": "",
    "\u200C": "",
    "\u200D": "",
    "\u2060": "",
})

class TextNormalization:


    #====================================================================================================================================================================
    # two funcs to change column headers
    #====================================================================================================================================================================


    def format_headers(self,dataframe):
        """
        convert headers to lower and replace spaces with _
        """
        cols=dataframe.columns
        new_cols=cols.str.lower().str.replace(' ','_').str.strip()
        dataframe.columns=new_cols
        return dataframe

    def header_to_title(self,dataframe):
        """
        convert to title, replace _ with space
        """
        cols=dataframe.columns
        new_cols=cols.str.replace('_',' ').str.title().str.strip()
        dataframe.columns=new_cols
        return dataframe



    #====================================================================================================================================================================
    # a func to normalize to unicode
    #====================================================================================================================================================================

    #TQDM WOULD BE USEFUL FOR JUPYTER NOTEBOOK USE CASES  
    def normalize_dataframe_text(self, df: pd.DataFrame, columns:None | list = None, normalize_unicode: bool = True) -> pd.DataFrame:
        """
        Normalize text columns in a DataFrame:
        - Convert to string / Unicode
        - Optional Unicode normalization (NFKC)
        - Replace dashes, quotes, invisible characters
        - Remove 'nan' (case-insensitive) in cells that are only 'nan'
        - Collapse whitespace
        - Replace empty strings with pd.NA  ####<-------------------------does this after creating strings becaus

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        columns : list, optional
            List of columns to normalize. If None, all object columns are normalized.
        normalize_unicode : bool
            Whether to apply Unicode normalization (recommended)
        """

        global TRANSLATION
        # auto detect if no columns entered        
        if columns is None:
            columns = df.select_dtypes(include="object").columns.tolist()      
        # Unicode normalization if normalize_unicode True
        if normalize_unicode:
            for col in columns:
                nans=df[col].isna()
                df[col]=df[col].astype(str).apply(lambda x: unicodedata.normalize("NFKC", x)).str.translate(TRANSLATION).str.replace(r"\s+", " ", regex=True).str.strip()
                df.loc[nans,col]=pd.NA
        else:
            for col in columns:                
                nans=df[col].isna()
                df[col] = df[col].astype(str).str.translate(TRANSLATION).str.replace(r"\s+", " ", regex=True).str.strip()
                df.loc[nans,col]=pd.NA
        return df
            







