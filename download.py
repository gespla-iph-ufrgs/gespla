''' 
***** UNIVERSIDADE FEDERAL DO RIO GRANDE DO SUL *****
********** GESPLA IPH/UFGRS PYTHON LIBRARY **********

Website: https://www.ufrgs.br/warp/
Repository: https://github.com/gespla-iph-ufrgs/gespla

This file is under LICENSE: GNU General Public License v3.0
Permissions:
    Commercial use
    Modification
    Distribution
    Patent use
    Private use 
Limitations:
    Liability
    Warranty 
Conditions:
    License and copyright notice
    State changes
    Disclose source
    Same license 

Module description:
--Download data and metadata functions
--Files are saved in .txt format.

Authors:
Marcio Inada: https://github.com/mshigue
Ipora Possantti: https://github.com/ipo-exe

First commit: 20 of October of 2020

'''

import pandas as pd


def metadata_ana_flow(folder='.', suff='flow'):
    """
    This function downloads metadata for all flow stations registered in the  Brazilian
    National Agency of Water (ANA).
    
    File is saved in .txt format
    To the file name is attached the date of download in format: 'YYYY-MM-DD'
    
    Dependencies:
    * HydroBr as hb
    * Pandas as pd
    :param folder: [optional] string of output directory (ex: 'C:/Datasets/Hydrology' )
    :param suff: [optional] string suffix for file name
    :return: string of file path (ex: 'C:/Datasets/Hydrology/metadata_ANA-flow_2020-10-19.txt' )
    """
    import hydrobr as hb

    # function for timestamp
    def today(p0='-'):
        import datetime
        def_now = datetime.datetime.now()
        def_lst = [def_now.strftime('%Y'), def_now.strftime('%m'), def_now.strftime('%d')]
        def_s = str(p0.join(def_lst))
        return def_s
    #
    df = hb.get_data.ANA.list_flow_stations()
    df.rename(mapper={'Code': 'CodEstacao'}, axis='columns', inplace=True)
    df.set_index('CodEstacao', inplace=True)  # set the 'CodEstacao' as the index of the DataFrame
    df.sort_index(inplace=True)  # sort the DataFrame by index
    def_export_file = folder + '/' + 'metadata_' + 'ANA' + '-' + suff + '_' + today() + '.txt'
    df.to_csv(def_export_file, sep=';')
    return def_export_file


def metadata_ana_prec(folder='.', suff='prec'):
    """
    this function downloads metadata for all precipitation stations registered in the  Brazilian
    National Agency of Water (ANA).
    
    File is saved in .txt format
    To the file name is attached the date of download in format: 'YYYY-MM-DD'
    
    External dependencies:
    * HydroBr as hb
    * Pandas as pd
    :param folder: string of output directory (ex: 'C:/Datasets/Hydrology' )
    :param suff: string suffix for file name
    :return: string of file path (ex: 'C:/Datasets/Hydrology/metadata_ANA-prec_2020-10-19.txt' )
    """
    import hydrobr as hb

    # function for timestamp
    def today(p0='-'):
        import datetime
        def_now = datetime.datetime.now()
        def_lst = [def_now.strftime('%Y'), def_now.strftime('%m'), def_now.strftime('%d')]
        def_s = str(p0.join(def_lst))
        return def_s
    #
    df = hb.get_data.ANA.list_prec_stations()
    df.rename(mapper={'Code': 'CodEstacao'}, axis='columns', inplace=True)
    df.set_index('CodEstacao', inplace=True)  # set the 'CodEstacao' as the index of the DataFrame
    df.sort_index(inplace=True)  # sort the DataFrame by index
    def_export_file = folder + '/' + 'metadata_' + 'ANA' + '-' + suff + '_' + today() + '.txt'
    df.to_csv(def_export_file, sep=';')
    return def_export_file


def metadata_inmet(folder='.', opt='both'):
    """
    this function downloads metadata for all precipitation stations registered in the  Brazilian
    National Agency of Water (ANA) or the INMET    
    
    File is saved in .txt format
    To the file name is attached the date of download in format: 'YYYY-MM-DD'

    External dependencies:
    * HydroBr as hb
    * Pandas as pd

    :param folder: string of output directory (ex: 'C:/Datasets/Hydrology' )
    :param opt: string for type of station:
    'both' - all types;
    'automatic' - only automatic stations;
    'conventional' - only conventional stations;
    :return: string of file path (ex: 'C:/Datasets/Hydrology/metadata_INMET-both_2020-10-19.txt' )
    """
    import hydrobr as hb

    # function for timestamp
    def today(p0='-'):
        import datetime
        def_now = datetime.datetime.now()
        def_lst = [def_now.strftime('%Y'), def_now.strftime('%m'), def_now.strftime('%d')]
        def_s = str(p0.join(def_lst))
        return def_s

    # use the .get_data method
    df = hb.get_data.INMET.list_stations(station_type=opt)
    df.rename(mapper={'Code': 'CodEstacao',
                      'Start Operation': 'StartOperation',
                      'End Operation': 'EndOperation'}, axis='columns', inplace=True)
    df.set_index('CodEstacao', inplace=True)  # set the 'CodEstacao' as the index of the DataFrame
    df.sort_index(inplace=True)  # sort the DataFrame by index
    def_export_file = folder + '/' + 'metadata_' + 'INMET-' + opt + '_' + today() + '.txt'
    df.to_csv(def_export_file, sep=';')
    return def_export_file


def metadata_ana_telemetry(folder='.'):
    """
    Importa o inventário da rede Telemétrica da ANA
    Os dados das estações são importados do inventario do Hidro e Telemétrica da ANA, segundo o documento
    "DESCRIÇÃO PARA DISPONIBILIZAR DADOS HIDROMETEOROLÓGICOS DOS SISTEMAS TELEMETRIA 1 E HIDRO" da ANA, disponível
    em https://www.ana.gov.br/telemetria1ws/Telemetria1ws.pdf.
    A importação é realizada à partir do acesso ao portal da ANA via API (Application Programming Interface),
    onde, através de entradas de parâmetros "vazios" é retornada uma lista com todas as estações hidrometeorológicas.
    Portanto, este processo pode levar algum tempo para a execução, dependendo da capacidade computacional
    e da conexão de internet.
    Depois de gerada a lista no formato XML, o processo seguinte organiza os dados na forma de um data frame do
    Pandas e, então é exportado para um arquivo CSV.
    OBS: caso tenha a necessidade de fazer uma personalização na importação, é recomendável ler a documentação a cima.
    Neste caso, os dados de entrada devem ser inseridos no dicionário `params...` e as saídas podem ser
    habilitadas ou excluídas na lista `columns...`.
    --------------------------------------
    File is saved in .txt format
    To the file name is attached the date of download in format: 'YYYY-MM-DD'
        
    Dependencies:
        pandas as pd
    Internal dependencies:
        * tqdm

    :param folder: string for ouputfile directory (ex: 'C:/Datasets/ANA/RHN')
    :return: string of file path
    """
    import requests
    import xml.etree.ElementTree as ET
    from tqdm import tqdm
    from datetime import datetime

    # function for timestamp
    def today(p0='-'):
        def_now = datetime.now()
        def_lst = [def_now.strftime('%Y'), def_now.strftime('%m'), def_now.strftime('%d')]
        def_s = str(p0.join(def_lst))
        return def_s

    # IMPORTA INVENTÁRIO DA REDE TELEMÉTRICA
    # Parâmetros de entrada
    params = {'statusEstacoes': '', 'origem': ''}
    # Dados de saída
    columns_str = ['NomeEstacao',
                   'Operadora', 'Responsavel',
                   'Municipio-UF', 'NomeRio',
                   'Origem', 'StatusEstacao']
    columns_num = ['CodEstacao',
                   'Bacia', 'SubBacia', 'CodRio',
                   'Latitude', 'Longitude', 'Altitude']
    # Busca no API da ANA
    inventario_telemetrica = requests.get('http://telemetriaws1.ana.gov.br/ServiceANA.asmx/ListaEstacoesTelemetricas', params)
    # Organiza os dados no data frame
    tree = ET.ElementTree(ET.fromstring(inventario_telemetrica.content))
    root = tree.getroot()
    index = 1
    df_telemetrica = pd.DataFrame()
    for station in tqdm(root.iter('Table')):
        for column in columns_str:
            df_telemetrica.at[index, column] = str(getattr(station.find(column), 'text', None))
        for column in columns_num:
            df_telemetrica.at[index, column] = getattr(station.find(column), 'text', None)
        index += 1
    #
    df_telemetrica.set_index('CodEstacao', inplace=True)
    df_telemetrica.sort_index(inplace=True)
    #
    # Exporta os dados para o arquivo CSV (txt)
    export_file_name = folder + '/' + 'metadata_ANA_RHN-Telemetry_' + today() + '.txt'
    df_telemetrica.to_csv(export_file_name, sep=';')
    return export_file_name


def metadata_ana_rhn_inventory(folder='.'):
    """
    Importa o inventário da rede Hidro e Telemétrica da ANA
    Os dados das estações são importados do inventario do Hidro e Telemétrica da ANA, segundo o documento
    "DESCRIÇÃO PARA DISPONIBILIZAR DADOS HIDROMETEOROLÓGICOS DOS SISTEMAS TELEMETRIA 1 E HIDRO" da ANA, disponível
    em https://www.ana.gov.br/telemetria1ws/Telemetria1ws.pdf.
    A importação é realizada à partir do acesso ao portal da ANA via API (Application Programming Interface),
    onde, através de entradas de parâmetros "vazios" é retornada uma lista com todas as estações hidrometeorológicas.
    Portanto, este processo pode levar algum tempo para a execução, dependendo da capacidade computacional
    e da conexão de internet.
    Depois de gerada a lista no formato XML, o processo seguinte organiza os dados na forma de um data frame do
    Pandas e, então é exportado para um arquivo CSV.
    OBS: caso tenha a necessidade de fazer uma personalização na importação, é recomendável ler a documentação a cima.
    Neste caso, os dados de entrada devem ser inseridos no dicionário `params...` e as saídas podem ser
    habilitadas ou excluídas na lista `columns...`.
    --------------------------------------
    File is saved in .txt format
    To the file name is attached the date of download in format: 'YYYY-MM-DD'

    Dependencies:
        pandas as pd
    Internal dependencies:
        * requests
        * xml
        * tqdm

    :param folder: string for ouputfile directory (ex: 'C:/Datasets/ANA/RHN')
    :return: string of file path
    """
    import requests
    import xml.etree.ElementTree as ET
    from tqdm import tqdm
    from datetime import datetime
    #
    # function for timestamp
    def today(p0='-'):
        def_now = datetime.now()
        def_lst = [def_now.strftime('%Y'), def_now.strftime('%m'), def_now.strftime('%d')]
        def_s = str(p0.join(def_lst))
        return def_s

    # IMPORTA INVENTÁRIO DA REDE HIDRO ##
    #
    # Parâmetros de entrada
    params = {'codEstDE': '', 'codEstATE': '', 'tpEst': '', 'nmEst': '',
              'nmRio': '', 'codSubBacia': '', 'codBacia': '',
              'nmMunicipio': '', 'nmEstado': '',
              'sgResp': '', 'sgOper': '',
              'telemetrica': ''}
    #
    # Dados de saída
    columns_str = ['Nome',
                   'RioNome',
                   'nmEstado', 'nmMunicipio',
                   'ResponsavelSigla', 'OperadoraSigla',
                   'ResponsavelUnidade', 'ResponsavelJurisdicao',
                   'OperadoraUnidade', 'OperadoraSubUnidade',
                   'Descricao', 'NumImagens']
    columns_num = ['Codigo', 'TipoEstacao',
                   'CodigoAdicional', 'RioCodigo',
                   'BaciaCodigo', 'SubBaciaCodigo',
                   'EstadoCodigo', 'MunicipioCodigo',
                   'ResponsavelCodigo', 'OperadoraCodigo',
                   'Latitude', 'Longitude', 'Altitude', 'AreaDrenagem',
                   'TipoEstacaoEscala', 'TipoEstacaoRegistradorNivel', 'TipoEstacaoDescLiquida',
                   'TipoEstacaoSedimentos', 'TipoEstacaoQualAgua', 'TipoEstacaoPluviometro',
                   'TipoEstacaoRegistradorChuva', 'TipoEstacaoTanqueEvapo',
                   'TipoEstacaoClimatologica', 'TipoEstacaoPiezometria',
                   'TipoEstacaoTelemetrica',
                   'TipoRedeBasica', 'TipoRedeEnergetica', 'TipoRedeNavegacao',
                   'TipoRedeCursoDagua', 'TipoRedeEstrategica', 'TipoRedeCaptacao',
                   'TipoRedeSedimentos', 'TipoRedeQualAgua', 'TipoRedeClasseVazao',
                   'Operando']
    columns_date = ['PeriodoEscalaInicio', 'PeriodoEscalaFim',
                    'PeriodoRegistradorNivelInicio', 'PeriodoRegistradorNivelFim',
                    'PeriodoDescLiquidaInicio', 'PeriodoDescLiquidaFim',
                    'PeriodoSedimentosInicio', 'PeriodoSedimentosFim',
                    'PeriodoQualAguaInicio', 'PeriodoQualAguaFim',
                    'PeriodoPluviometroInicio', 'PeriodoPluviometroFim',
                    'PeriodoRegistradorChuvaInicio', 'PeriodoRegistradorChuvaFim',
                    'PeriodoTanqueEvapoInicio', 'PeriodoTanqueEvapoFim',
                    'PeriodoClimatologicaInicio', 'PeriodoClimatologicaFim',
                    'PeriodoPiezometriaInicio', 'PeriodoPiezometriaFim',
                    'PeriodoTelemetricaInicio', 'PeriodoTelemetricaFim',
                    'UltimaAtualizacao', 'DataIns', 'DataAlt']
    #
    # Busca no API da ANA
    inventario_hidro = requests.get('http://telemetriaws1.ana.gov.br/ServiceANA.asmx/HidroInventario', params)
    #
    # Organiza os dados no data frame
    tree = ET.ElementTree(ET.fromstring(inventario_hidro.content))
    root = tree.getroot()
    index = 1
    df_hidro = pd.DataFrame()
    for station in tqdm(root.iter('Table')):
        for column in columns_str:
            df_hidro.at[index, column] = str(getattr(station.find(column), 'text', None))
        for column in columns_num:
            df_hidro.at[index, column] = getattr(station.find(column), 'text', None)
        for column in columns_date:
            if station.find(column).text is not None:
                df_hidro.at[index, column] = pd.to_datetime(
                    datetime.strptime(station.find(column).text, '%Y-%m-%d %H:%M:%S'), errors='coerce')
        index += 1
    #
    df_hidro.rename(mapper={'Codigo': 'CodEstacao'}, axis='columns', inplace=True)  # rename 'Codigo' by 'CodEstacao'
    df_hidro.set_index('CodEstacao', inplace=True)  # set the 'CodEstacao' as the index of the DataFrame
    df_hidro.sort_index(inplace=True)
    #
    # Exporta os dados para o arquivo CSV (txt)
    export_file_name = folder + '/' + 'metadata_ANA_RHN-Full-Inventory_' + today() + '.txt'
    df_hidro.to_csv(export_file_name, sep=';')
    return export_file_name


def ana_flow(code, folder='.', suff='flow'):
    """
    This function downloads the time series of measured flow at a single station of ANA

    External dependencies:
    * HydroBr as hb
    * Pandas as pd

    :param code: string of the station code
    :param folder: string of output directory (ex: 'C:/Datasets/Hydrology' )
    :param suff: string suffix for file name
    :return: string of file path (ex: 'C:/Datasets/Hydrology/ANA-flow_61861000_2020-10-19.txt' )
    """
    import hydrobr as hb

    # function for timestamp
    def today(p0='-'):
        import datetime
        def_now = datetime.datetime.now()
        def_lst = [def_now.strftime('%Y'), def_now.strftime('%m'), def_now.strftime('%d')]
        def_s = str(p0.join(def_lst))
        return def_s

    # get the metadata first by using the list_flow_stations() method
    df_meta = hb.get_data.ANA.list_flow_stations()
    df_meta.set_index('Code', inplace=True)  # set the 'Code' as the index of the DataFrame
    df_meta.sort_index(inplace=True)  # sort the DataFrame by index
    error_str = ''
    if code in df_meta.index:
        # use the .flow_data method
        df = hb.get_data.ANA.flow_data([code])
    else:
        # create an error msg dataframe
        dct = {'Error': ['Station Code not found']}
        indx = [code]
        df = pd.DataFrame(dct, index=indx)
        error_str = 'error_'
    def_export_file = folder + '/' + error_str + 'ANA' + '-' + suff + '_' + code + '_' + today() + '.txt'
    df.to_csv(def_export_file, sep=';')
    return def_export_file


def ana_prec(code, folder='.', suff='prec'):
    """
    This function downloads the timeseries of measured precipitation at a single station of ANA

    External dependencies:
    * HydroBr as hb
    * Pandas as pd

    :param code: string of the station code
    :param folder: string of output directory (ex: 'C:/Datasets/Hydrology' )
    :param suff: string suffix for file name
    :return: string of file path (ex: 'C:/Datasets/Hydrology/ANA-prec_10064000_2020-10-19.txt' )
    """
    import hydrobr as hb

    # function for timestamp
    def today(p0='-'):
        import datetime
        def_now = datetime.datetime.now()
        def_lst = [def_now.strftime('%Y'), def_now.strftime('%m'), def_now.strftime('%d')]
        def_s = str(p0.join(def_lst))
        return def_s

    # get the metadata first by using the list_prec_stations() method
    df_meta = hb.get_data.ANA.list_prec_stations()
    df_meta.set_index('Code', inplace=True)  # set the 'Code' as the index of the DataFrame
    df_meta.sort_index(inplace=True)  # sort the DataFrame by index
    error_str = ''
    if code in df_meta.index:
        # use the .flow_data method
        df = hb.get_data.ANA.prec_data([code])
    else:
        # create an error msg dataframe
        dct = {'Error': ['Station Code not found']}
        indx = [code]
        df = pd.DataFrame(dct, index=indx)
        error_str = 'error_'
    def_export_file = folder + '/' + error_str + 'ANA' + '-' + suff + '_' + code + '_' + today() + '.txt'
    df.to_csv(def_export_file, sep=';')
    return def_export_file


def inmet_daily(code, folder='.'):
    """
    This function downloads the timeseries of daily measured climate variables at a single station of INMET
    --------------------------------------
    File is saved in .txt format
    To the file name is attached the date of download in format: 'YYYY-MM-DD'
    
    External dependencies:
    * HydroBr as hb
    * Pandas as pd

    :param code: string of the station code
    :param folder: string of output directory (ex: 'C:/Datasets/Climate )
    :return: string of file path (ex: 'C:/Datasets/Hydrology/INMET_10064000_2020-10-19.txt' )
    """
    import hydrobr as hb

    # function for timestamp
    def today(p0='-'):
        import datetime
        def_now = datetime.datetime.now()
        def_lst = [def_now.strftime('%Y'), def_now.strftime('%m'), def_now.strftime('%d')]
        def_s = str(p0.join(def_lst))
        return def_s

    # get the metadata first by using the list_flow_stations() method
    df_meta = hb.get_data.INMET.list_stations(station_type='both')
    df_meta.set_index('Code', inplace=True)  # set the 'Code' as the index of the DataFrame
    df_meta.sort_index(inplace=True)  # sort the DataFrame by index
    error_str = ''
    if code in df_meta.index:
        # use the .flow_data method
        df = hb.get_data.INMET.daily_data(code)
    else:
        # create an error msg dataframe
        dct = {'Error': ['Station Code not found']}
        indx = [code]
        df = pd.DataFrame(dct, index=indx)
        error_str = 'error_'
    def_export_file = folder + '/' + error_str + 'INMET-daily_' + code + '_' + today() + '.txt'
    df.to_csv(def_export_file, sep=';')
    return def_export_file


def inmet_hourly(code, folder='.'):
    """
    This function downloads the timeseries of hourly measured climate variables at a single station of INMET.
    Only available for automatic stations.
    --------------------------------------
    File is saved in .txt format
    To the file name is attached the date of download in format: 'YYYY-MM-DD'

    Dependencies:
    * HydroBr as hb
    * Pandas as pd

    :param code: string of the station code
    :param folder: string of output directory (ex: 'C:/Datasets/Climate )
    :return: string of file path (ex: 'C:/Datasets/Hydrology/INMET_10064000_2020-10-19.txt' )
    """
    import hydrobr as hb

    # function for timestamp
    def today(p0='-'):
        import datetime
        def_now = datetime.datetime.now()
        def_lst = [def_now.strftime('%Y'), def_now.strftime('%m'), def_now.strftime('%d')]
        def_s = str(p0.join(def_lst))
        return def_s

    # get the metadata first by using the list_flow_stations() method
    df_meta = hb.get_data.INMET.list_stations(station_type='automatic')
    df_meta.set_index('Code', inplace=True)  # set the 'Code' as the index of the DataFrame
    df_meta.sort_index(inplace=True)  # sort the DataFrame by index
    error_str = ''
    if code in df_meta.index:
        # use the .flow_data method
        df = hb.get_data.INMET.hourly_data(code)
    else:
        # create an error msg dataframe
        dct = {'Error': ['Station Code not found']}
        indx = [code]
        df = pd.DataFrame(dct, index=indx)
        error_str = 'error_'
    def_export_file = folder + '/' + error_str + 'INMET-hourly_' + code + '_' + today() + '.txt'
    df.to_csv(def_export_file, sep=';')
    return def_export_file
