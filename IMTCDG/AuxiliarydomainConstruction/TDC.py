#
import math
#
import torch
#
import TemporalDistance.distance as Distance
device = "cuda" if torch.cuda.is_available() else "cpu"

class DistanceFunction(object):
    def __init__(self, dist_type="Cosine"):
        self.dist_type = dist_type
        # self.input_dim = input_dim

    def Compute(self, X, Y):
        Loss = 0.0
        if self.dist_type == "MMD_Linear":
            mmd_loss = Distance.MMD_loss(kernel_type="linear")
            Loss = mmd_loss(X, Y)
        elif self.dist_type == "MMD_rbf":
            mmd_loss = Distance.MMD_loss(kernel_type="rbf")
            Loss = mmd_loss(X, Y)
        elif self.dist_type == "Coral":
            Loss = Distance.CORAL(X, Y)
        elif self.dist_type == "Cosine":
            Loss = 1 - Distance.Cosine(X, Y)
        elif self.dist_type == "KL":
            Loss = Distance.KL_Div(X, Y)
        elif self.dist_type == "JS":
            Loss = Distance.JS_Div(X, Y)

        return Loss

# Dynamic programming
# Time Distribution Characterization
def TDC(data, Num_domain, dist_type="Cosine"):
    data = data
    Start = 0
    End, Length = len(data), len(data)

    split_N = 10
    Feat = data[:, :]

    Selected = [0, 10]
    Candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    if Num_domain in [2, 4, 6, 8, 10]:
        while len(Selected) - 2 < Num_domain - 1:
            Dist_List = []
            for C in Candidate:
                Selected.append(C)
                Selected.sort()
                Dist_tmp = 0
                for i in range(1, len(Selected) - 1):
                    for j in range(i, len(Selected) - 1):
                        Part1_Index_Start = Start + math.floor(Selected[i-1] / split_N * Length)
                        Part1_Index_End = Start + math.floor(Selected[i] / split_N * Length)
                        Feat_Part1 = Feat[Part1_Index_Start:Part1_Index_End, :]
                        Part2_Index_Start = Start + math.floor(Selected[j] / split_N * Length)
                        Part2_Index_End = Start + math.floor(Selected[j+1] / split_N * Length)
                        Feat_Part2 = Feat[Part2_Index_Start:Part2_Index_End]
                        dist_Fuc = DistanceFunction(dist_type=dist_type)
                        Part_Distance = dist_Fuc.Compute(Feat_Part1, Feat_Part2)
                        Dist_tmp += Part_Distance
                Dist_List.append(Dist_tmp)
                Selected.remove(C)
            C_index = Dist_List.index(max(Dist_List))
            Selected.append(Candidate[C_index])
            Candidate.remove(Candidate[C_index])
        Selected.sort()
        print(Selected)
    return None

# if __name__ == "__main__":
#     samples = torch.randn((500, 5)).to(device)
#     dist_Func = TDC(samples, 6)