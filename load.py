import pandas as pd


def metadata_ana_flow(file):
    """
    This function loads a DataFrame based on the file created in  .download.metadata_ana_flow()
    :param file: file string
    :return: pandas DataFrame
    """
    def_df = pd.read_csv(file, sep=';', index_col='CodEstacao', parse_dates=['StartDate', 'EndDate'])
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
    def_df = pd.read_csv(file, sep=';', index_col='CodEstacao', parse_dates=['StartDate', 'EndDate'])
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
    def_df = pd.read_csv(file, sep=';', index_col='CodEstacao', parse_dates=['StartOperation'])
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
    return def_df


def ana_stage(file):
    """
    This function loads a DataFrame from the .txt file created in .download.ana_stage()
    :param file: file string
    :return: pandas DataFrame
    """
    def_df = pd.read_csv(file, sep=';')
    def_df['Date'] = pd.to_datetime(def_df['Date'])
    return def_df


def ana_prec(file):
    """
    This function loads a DataFrame from the .txt file created in .download.ana_prec()
    :param file: file string
    :return: pandas DataFrame
    """
    def_df = pd.read_csv(file, sep=';')
    def_df['Date'] = pd.to_datetime(def_df['Date'])
    return def_df


def inmet_daily(file):
    """
    This function loads a DataFrame from the .txt file created in .download.inmet_daily()
    :param file: file string
    :return: pandas DataFrame
    """
    def_df = pd.read_csv(file, sep=';')
    def_df['Date'] = pd.to_datetime(def_df['Date'])
    return def_df


def inmet_hourly(file):
    """
    This function loads a DataFrame from the .txt file created in .download.inmet_hourly()
    :param file: file string
    :return: pandas DataFrame
    """
    def_df = pd.read_csv(file, sep=';')
    def_df['Date'] = pd.to_datetime(def_df['Date'])
    return def_df



