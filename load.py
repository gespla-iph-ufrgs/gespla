import pandas as pd


def insert_gaps(dataframe, date_field='Date', freq='D'):
    """
    This is a convenience function that standardizes a timeseries by inserting the missing gaps as actual records
    :param dataframe: pandas DataFrame object
    :param date_field: string datefield - Default: 'Date'
    :param freq: string frequency alias offset (see pandas documentation). Dafault: 'D' (daily)
    :return: pandas DataFrame object with inserted gaps records
    """
    # get data from DataFrame
    in_df = dataframe.copy()
    # ensure Date field is datetime
    in_df[date_field] = pd.to_datetime(in_df[date_field])
    # create start and end values
    start = in_df[date_field].min()
    end = in_df[date_field].max()
    # create the reference date index
    ref_dates = pd.date_range(start=start, end=end, freq=freq)
    # create the reference dataset
    ref_df = pd.DataFrame({'Date':ref_dates})
    # left join on datasets
    merge = pd.merge(ref_df, in_df, how='left', left_on='Date', right_on=date_field)
    return merge


def metadata_ana_flow(file):
    """
    This function loads a DataFrame based on the file created in  .download.metadata_ana_flow()
    :param file: file string
    :return: pandas DataFrame
    """
    def_df = pd.read_csv(file, sep=';', dtype={'CodEstacao': str}, parse_dates=['StartDate', 'EndDate'])
    # data set memory optimization
    def_df['Type'] = def_df['Type'].astype('category')
    def_df['State'] = def_df['State'].astype('category')
    def_df['Responsible'] = def_df['Responsible'].astype('category')
    return def_df


def metadata_ana_prec(file):
    """
    This function loads a DataFrame based on the file created in  .download.metadata_ana_prec()
    :param file: file string
    :return: pandas DataFrame
    """
    def_df = pd.read_csv(file, sep=';', dtype={'CodEstacao': str}, parse_dates=['StartDate', 'EndDate'])
    # data set memory optimization
    def_df['Type'] = def_df['Type'].astype('category')
    def_df['State'] = def_df['State'].astype('category')
    def_df['Responsible'] = def_df['Responsible'].astype('category')
    return def_df


def metadata_inmet(file):
    """
    This function loads a DataFrame based on the file created in .download.metadata_inmet()
    :param file: file string
    :return: pandas DataFrame
    """
    def_df = pd.read_csv(file, sep=';', dtype={'CodEstacao': str}, parse_dates=['StartOperation'])
    # data set memory optimization
    def_df['Type'] = def_df['Type'].astype('category')
    def_df['State'] = def_df['State'].astype('category')
    return def_df


def ana_flow(file):
    """
    This function loads a DataFrame from the .txt file created in .download.ana_flow()
    :param file: file string
    :return: pandas DataFrame
    """
    def_df = pd.read_csv(file, sep=';')
    def_df['Date'] = pd.to_datetime(def_df['Date'])
    def_merged = insert_gaps(def_df, freq='D')
    return def_merged


def ana_stage(file):
    """
    This function loads a DataFrame from the .txt file created in .download.ana_stage()
    :param file: file string
    :return: pandas DataFrame
    """
    def_df = pd.read_csv(file, sep=';')
    def_df['Date'] = pd.to_datetime(def_df['Date'])
    def_merged = insert_gaps(def_df, freq='D')  # make sure the gaps are in record
    return def_merged


def ana_prec(file):
    """
    This function loads a DataFrame from the .txt file created in .download.ana_prec()
    :param file: file string
    :return: pandas DataFrame
    """
    def_df = pd.read_csv(file, sep=';')
    def_df['Date'] = pd.to_datetime(def_df['Date'])
    def_merged = insert_gaps(def_df, freq='D')  # make sure the gaps are in record
    return def_merged


def inmet_daily(file):
    """
    This function loads a DataFrame from the .txt file created in .download.inmet_daily()
    :param file: file string
    :return: pandas DataFrame
    """
    def_df = pd.read_csv(file, sep=';')
    def_df['Date'] = pd.to_datetime(def_df['Date'])
    def_merged = insert_gaps(def_df, freq='D')  # make sure the gaps are in record
    return def_merged


def inmet_hourly(file):
    """
    This function loads a DataFrame from the .txt file created in .download.inmet_hourly()
    :param file: file string
    :return: pandas DataFrame
    """
    def_df = pd.read_csv(file, sep=';')
    def_df['Date'] = pd.to_datetime(def_df['Date'])
    def_merged = insert_gaps(def_df, freq='H')  # make sure the gaps are in record
    return def_merged



