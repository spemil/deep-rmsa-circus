import os
import json
import random
import numpy as np
import pandas as pd
import networkx as nx
import multiprocessing
import threading
from time import sleep


def dbmtolin(pch):
    return 10 ** ((pch - 30) / 10)


def load_germany17_undirected(init_dir=True):
    """
    :return: directed graph instance of Ger17 network [networkx DiGraph],
             link information [dictionary]
    """
    G = nx.read_gml(os.path.join('../pynp/topology/topology_files', 'nobel-germany.gml'))
    G17 = nx.Graph()
    G17.add_nodes_from(G.nodes)
    with open(os.path.join('../pynp/topology/topology_files', 'Links_Germany_17.json'),
              mode='r', encoding='utf-8') as f:
        data = json.load(f)
    LinkDict = {}
    idx = 0
    for linkID, linkDetails in data.items():
        src = linkDetails['startNode']
        dst = linkDetails['endNode']
        length = linkDetails['linkDist']
        LinkDict[idx] = {'linkID': idx, 'sourceNode': src, 'dstNode': dst, 'linkLength': length,
                         'LPlist': [], 'SpanDict': None, 'fiberType': 'SSMF', 'SCIDict': {}}
        G17.add_edge(src, dst, weight=length, linkID=idx)
        idx += 1
    # LinkDict = generic_spans(LinkDict)
    return G17, LinkDict, None, 'DemandOSNR_Germany_17_Normal__C-Band_YEAR5_2022-03-07 20-33-47.json'


def random_trafficdemand(G, lambda_time=None, lambda_req=12,
                         filename='DemandOSNR_Germany_17_Normal__C-Band_YEAR5_2022-03-07 20-33-47.json'):
    if lambda_time is None:
        lambda_time = [14]
    nodes = G.nodes  # Nodes in the graph G
    pathlength = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))  # Lengths of all shortest paths in G
    DemandDict = {}

    service_time = lambda_time[np.random.randint(0, len(lambda_time))]  # Packet service time parameter
    with open(os.path.join('/users/emil/Documents/00_Uni/00_MA/pynp/topology/topology_files', filename), mode='r',
              encoding='utf-8') as f:
        data = json.load(f)
    if filename == 'DemandOSNR_Germany_17_Normal__C-Band_YEAR5_2022-03-07 20-33-47.json':
        arrival_time = 0
        for demandID in range(len(data)):
            randID, randDetails = random.choice(list(data.items()))
            DemandDict[demandID] = {'demandID': demandID, 'srcNode': randDetails['srcNode'],
                                    'dstNode': randDetails['dstNode'],
                                    'shortest_path_length': pathlength[randDetails['srcNode']][randDetails['dstNode']],
                                    'reqDataRate': randDetails['yearlyDemandIncrease'][1], 'paths': None, 'LPlist': [],
                                    'eolSNR': None, 'provDataRate': 0, 'tOA': arrival_time, 'tOD': np.inf,
                                    'action': None}
            arrival_time += np.random.exponential(service_time)
    return DemandDict


def readconfigs(filename, pch_scaler=1, Txpenalty=0):
    with open(os.path.join('config_files', filename),
              mode='r', encoding='utf-8') as f:
        data = json.load(f)
    configs = []
    DataRates = []
    reqSNR = []
    IDs = []
    for configID, configDetails in data.items():
        DataRates.append(float(configDetails['Data_Rate']))
        configDetails['orig_pch'] = dbmtolin(configDetails['chLaunchPow'])
        configDetails['PSD'] = dbmtolin(configDetails['chLaunchPow']) * pch_scaler / (configDetails['SR'] * (
                1 + configDetails['rloff']))
        if 'reqOSNR' in configDetails.keys():
            configDetails['reqSNR'] = configDetails['reqOSNR'] - 10 * np.log10(configDetails['SR'] / 12.5)
        reqSNR.append(configDetails['reqSNR'])
        if not 'TxOSNR' in configDetails.keys():
            configDetails['TxOSNR'] = 0
        configs.append(np.hstack([configID, configDetails['Data_Rate'], configDetails['SR'], configDetails['SR'] *
                                  (1 + configDetails['rloff']), configDetails['chBW'], configDetails['reqSNR'],
                                  configDetails['PSD'], configDetails['qam'], dbmtolin(configDetails['chLaunchPow'])
                                  * pch_scaler, configDetails['orig_pch'], configDetails['TxOSNR'] - Txpenalty]))
        IDs.append(int(configID))
        # Reorder with ascending reqSNR
    columns = ['configID', 'DR', 'SR', 'BW', 'slot', 'reqSNR', 'PSD', 'modf', 'pch', 'orig_pch', 'TxOSNR']
    df = pd.DataFrame(data=np.asarray(configs)[np.argsort(np.asarray(reqSNR))], columns=columns,
                      index=np.asarray(IDs)[np.argsort(np.asarray(reqSNR))])
    return df.astype('float')
