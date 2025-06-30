#
import numpy as np
import pandas as pd
#
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

# Discrimination of Similar Machining Features
def DBSCAN_Discrimination(datapath_List):

    data1_np = np.array(pd.read_csv(datapath_List[0])).reshape(-1, 27)
    data2_np = np.array(pd.read_csv(datapath_List[1])).reshape(-1, 27)
    data3_np = np.array(pd.read_csv(datapath_List[2])).reshape(-1, 27)
    dataall_np = np.vstack((data1_np, data2_np, data3_np))
    col_name = list(pd.read_csv(datapath_List[0]))

    Scaler = MinMaxScaler()
    data_MM = Scaler.fit_transform(dataall_np)
    My_DBSCAN = DBSCAN(eps=0.5, min_samples=480)
    cluster_result = My_DBSCAN.fit_predict(data_MM)
    cluster_result = np.array(cluster_result).reshape(-1, 1)

    col_name += ["聚类结果"]
    result_np = np.hstack((dataall_np, cluster_result))
    result_pd = pd.DataFrame(result_np, columns=col_name)
    result_pd.to_csv("./Result/Cluster_result/.csv")

    return cluster_result

if __name__ == "__main__":
    datapath_List = [r".csv",
                     r".csv",
                     r".csv"]
    cluster_result = DBSCAN_Discrimination(datapath_List)