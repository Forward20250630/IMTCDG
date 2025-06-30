#
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def pca_distribution(datapath, d_name):
    print(d_name)
    data = pd.read_csv(datapath)
    data_np = np.array(data.iloc[:, 0:25])
    MS_X = MinMaxScaler()
    pca = PCA(n_components=1)

    X = MS_X.fit_transform(data_np)
    X_ = pca.fit_transform(X)
    X_df = pd.DataFrame(X_, columns=["data"])
    X_df.to_csv(r" "+ d_name + ".csv", encoding="utf-8", index=False)

    return None

if __name__ == "__main__":
    d_Name = ["main_src", "unseen_trg"]
    data_path1 = r".csv"
    data_path2 = r".csv"
    pca_distribution(datapath=data_path1, d_name=d_Name[0])
    pca_distribution(datapath=data_path2, d_name=d_Name[1])