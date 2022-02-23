import copy

from pandas import DataFrame
import numpy as np
from scipy.spatial import distance_matrix

from two_bezier.Util.Cluster import Cluster
from two_bezier.Util.Road import Road
from two_bezier.index.GMRC import GRMC
from two_bezier.index.gray import Gray, get_G
import pandas as pd

def get_road_info_data():
    # 7. 模拟数据
    # import random
    #
    # road_section_list = []
    # for c in range(10):
    #     # 随机生成字段矢量[速度， 车流量， 矢量个数]
    #     road_section_list.append([random.randint(5, 120), random.randint(10, 100), random.randint(10, 200)])

    road_section_list = [[37, 17, 41], [36, 98, 73], [42, 18, 184], [41, 15, 47], [44, 78, 75], [38, 39, 82],
                         [33, 75, 108], [36, 28, 169], [34, 28, 190], [38, 49, 13]]

    i = 1
    for road_section in road_section_list:
        print("第", i, "条道路的信息为：", road_section)
        i = i + 1
    return road_section_list


# 将最优值（自由流速度参数列表）插入，获得A0...Ap矩阵
def insert_road_best(road_section_list, best_row):
    road_section_list_tmp = road_section_list
    road_section_list_tmp.insert(0, best_row)
    return road_section_list_tmp


def get_norm_A_matrixs(A):
    # 4 转为df，进行归一化操作
    road_section = A
    road_df2 = DataFrame(road_section)
    p_cols = ['v', 'flow', 'vector_nums']
    road_df2.columns = p_cols
    # 4.1 根据速度特点，进行归一化
    v_norm_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    road_df2[['v']] = abs(road_df2[['v']].apply(v_norm_scaler))
    # 4.2 根据车流量和矢量个数特点来进行归一化
    flow_vectors_scaler = lambda x: (x - np.max(x)) / (np.max(x) - np.min(x))
    road_df2[['flow']] = abs(road_df2[['flow']].apply(flow_vectors_scaler))
    road_df2[['vector_nums']] = abs(road_df2[['vector_nums']].apply(flow_vectors_scaler))
    # 4.3 输出归一化后的df信息
    # print("归一化后的df:", road_df2)
    return road_df2


def get_road_norm(road_section_list):
    # 得到df
    road_df1 = DataFrame(road_section_list)
    # 1. 取出矩阵中最大值与最小值
    max_loc = road_df1.max()
    min_loc = road_df1.min()
    # 2.根据每一列特点，获得最优值
    # 2.1 速度取最大值
    max_v = max_loc[0]
    # 2.2 车流量取最小值
    min_flow = min_loc[1]
    # 2.3 矢量个数取最小值
    min_vectors = min_loc[2]
    print("路段各参数的最优值为：", max_v, ",", min_flow, ",", min_vectors)
    
    # 3. 将最有参考值插入数据框的第一行
    X0 = [max_v, min_flow, min_vectors]

    # 将最有参考值插入数据框的第一行
    A0 = insert_road_best(road_section_list, X0)
    A1 = insert_road_best(road_section_list, Y1)
    A2 = insert_road_best(road_section_list, Y2)
    A3 = insert_road_best(road_section_list, Y3)
    A4 = insert_road_best(road_section_list, Y4)
    A5 = insert_road_best(road_section_list, Y5)
    # 4. 将以上矩阵进行归一化
    A0_norm = get_norm_A_matrixs(A0)
    A1_norm = get_norm_A_matrixs(A1)
    A2_norm = get_norm_A_matrixs(A2)
    A3_norm = get_norm_A_matrixs(A3)
    A4_norm = get_norm_A_matrixs(A4)
    A5_norm = get_norm_A_matrixs(A5)
    
    # 5.返回归一化后的矩阵A0...Ap
    return A0_norm, A1_norm, A2_norm, A3_norm, A4_norm, A5_norm


def get_distance(road_df):
    distance_df = pd.DataFrame(distance_matrix(road_df.values, road_df.values), index=road_df.index, columns=road_df.index)
    # print("欧式距离矩阵为：\n", distance_df)
    return distance_df


# 输入欧式距离矩阵D，返回归一化后的距离矩阵D_norm
# 按照每一行进行归一化
def get_diatance_norm(D):
    D_norm = abs(D.apply(lambda x: ((x - x.max())/(x.max() - x.min()))))
    # 转置之后，就将从按列进行归一化变为按行进行归一化
    D_norm_T = D_norm.T
    return D_norm_T


def get_S(G, D_norm):
    S = 0.5 * G + 0.5 * D_norm
    return S


def set_decision():
    # 1 计算P（RCi)
    for cluster in cluster_list:
        # (1) 获得第i个聚类结果中路段的数量rc
        rc = len(cluster)
        # (2) 获得总路段数量
        x = n
        # （3）计算概率
        P_RC = rc / n


def get_cluster_results(road_section_list, insert_row):
    # 1. 将最有参考值插入数据框的第一行
    road_section_list_tmp = copy.deepcopy(road_section_list)
    A = insert_road_best(road_section_list_tmp, insert_row)

    # 2. 将矩阵进行归一化
    A_norm = get_norm_A_matrixs(A)

    # 3.计算相关度RA
    RA = Gray(A_norm)
    # print("灰色关联矩阵为：\n", RA)
    # ShowGRAHeatMap(RA)

    # 4.计算A0矩阵的关联相似度矩阵G
    G = get_G(RA)
    # print("关联相似度矩阵G为：\n", G)
    # ShowGRAHeatMap(G)

    # 5. 计算欧式距离矩阵D
    D = get_distance(A_norm)

    # 6. 根据欧式距离特点，进行归一化
    D_norm = get_diatance_norm(D)
    # print("归一化后的欧式距离矩阵为：\n", D_norm)

    # 7. 根据灰色关联相似矩阵G 和欧式距离矩阵D ，权重系数为0.5，得到贴近度矩阵S
    S = get_S(G, D_norm)
    # print("贴近度矩阵为：\n", S)

    # 8. 聚类，定义的lambda=0.6
    cluster_list = GRMC(S)
    # print("改进后的GMRC聚类结果为：\n", cluster_list)
    # 5.返回归一化后的矩阵A0...Ap
    return cluster_list


def get_X0(road_section_list):
    # 得到df
    road_df1 = DataFrame(road_section_list)
    # 1. 取出矩阵中最大值与最小值
    max_loc = road_df1.max()
    min_loc = road_df1.min()
    # 2.根据每一列特点，获得最优值
    # 2.1 速度取最大值
    max_v = max_loc[0]
    # 2.2 车流量取最小值
    min_flow = min_loc[1]
    # 2.3 矢量个数取最小值
    min_vectors = min_loc[2]
    # print("路段各参数的最优值为：", max_v, ",", min_flow, ",", min_vectors)

    # 3. 将最有参考值插入数据框的第一行
    X0 = [max_v, min_flow, min_vectors]
    return X0


def get_road_obj(road_section_list):
    road_obj_list = []
    for i in range(len(road_section_list)):
        v = road_section_list[i][0]
        flow = road_section_list[i][1]
        vectors = road_section_list[i][2]
        road = Road(i+1, v, flow, vectors)
        print("road的index为：", road.index)
        road_obj_list.append(road)
    return road_obj_list


def get_cluster(index, A_cluster):
    for i in range(len(A_cluster)):
        # 若在某个聚类中，返回是第几聚类（聚类结果取值为：0-4）
        if index in A_cluster[i].get_index():
            return i
        else:
            continue


def get_road_gray_dist(road_section_list):
    road_section_list_tmp = copy.deepcopy(road_section_list)
    A_norm = pd.DataFrame(road_section_list_tmp)
    # 3.计算相关度RA
    RA = Gray(A_norm)
    print("灰色关联矩阵为：\n", RA)
    # ShowGRAHeatMap(RA)

    # 4.计算A0矩阵的关联相似度矩阵G
    G = get_G(RA)
    print("关联相似度矩阵G为：\n", G)
    # ShowGRAHeatMap(G)

    # 5. 计算欧式距离矩阵D
    D = get_distance(A_norm)

    # 6. 根据欧式距离特点，进行归一化
    D_norm = get_diatance_norm(D)
    # print("归一化后的欧式距离矩阵为：\n", D_norm)

    # 7. 根据灰色关联相似矩阵G 和欧式距离矩阵D ，权重系数为0.5，得到贴近度矩阵S
    S = get_S(G, D_norm)

    return S


def get_cluster_inner(road_section_list):
    SW_C = 0
    S = get_road_gray_dist(road_section_list)
    print("道路贴近度矩阵S为：\n", S)
    C_length = p
    for c in cluster_level_list:
        s = 0
        level_list = c.get_level()
        # 判断聚类里不为空
        if len(level_list) != 0:
            level_list = [i-1 for i in level_list]
            i = 0
            for i in range(len(level_list)):
                for j in range(i + 1, len(level_list)):
                    if level_list[i] != level_list[j]:
                        s = s + S.iat[level_list[i], level_list[j]]
            sw_c_i = 1 / (len(c.get_level()) * len(c.get_level())) * s
            SW_C += sw_c_i
        else:
            C_length = len(cluster_level_list) - 1

    SW_C = 1 / C_length * SW_C
    print("类内隶属度为：\n", SW_C)
    print("聚类数为：", C_length)
    
    return SW_C


def get_cluster_outer(road_section_list):
    SB_C_tmp = 0
    C_length2 = p
    for i in range(len(cluster_level_list)):
        c = cluster_level_list[i]
        level_list1 = c.get_level()
        if len(level_list1) != 0:
            level_list1 = [i-1 for i in level_list1]
            sb_tmp = 0
            sbc_tmp = 0
            for j in range(i+1, len(cluster_level_list)):
                level_list2 = cluster_level_list[j].get_level()
                if len(level_list2) != 0:
                    level_list2 = [i-1 for i in level_list2]
                    for c1 in level_list1:
                        for c2 in level_list2:
                            sb_tmp = sb_tmp + S.iat[c1, c2]
                    #
                    sbc_tmp = sbc_tmp + 1 / (len(level_list1) * len(level_list2)) * sb_tmp
            SB_C_tmp = SB_C_tmp + sbc_tmp
        else:
            C_length2 = len(cluster_level_list) - 1

    SB_C = 1 / (C_length2 * (C_length2 - 1)) * SB_C_tmp
    print("类间离散度为：\n", SB_C)
    print("聚类数为：", C_length2)
    
    return SB_C


if __name__ == '__main__':
    # 1. road_section_list为道路三个字段的信息
    # 路段数量为10
    n = 10
    # 假设p = 5
    p = 4
    # 获得初始数据
    road_section_list = get_road_info_data()
    # 获得自由流三个字段数据
    Y1 = [70, 16, 22]
    Y2 = [60, 18, 25]
    Y3 = [80, 10, 20]
    Y4 = [67, 19, 22]
    Y5 = [50, 20, 27]

    # 2.得到最优值列表X0
    X0 = get_X0(road_section_list)

    lambda_test = list(np.arange(0, 1, 0.05))
    for lam in lambda_test:
        # 3. 得到(p+1)个聚类结果
        A0_cluster = get_cluster_results(road_section_list, X0)
        A1_cluster = get_cluster_results(road_section_list, Y1)
        A2_cluster = get_cluster_results(road_section_list, Y2)
        A3_cluster = get_cluster_results(road_section_list, Y3)
        A4_cluster = get_cluster_results(road_section_list, Y4)
        A5_cluster = get_cluster_results(road_section_list, Y5)
    
        cluster_list = [A0_cluster, A1_cluster, A2_cluster, A3_cluster, A4_cluster, A5_cluster]
        # 4. 建立决策表系统，计算各个聚类成员的信息熵
        # set_decision()
    
        # 5.计算每个路段在cluster中出现概率
        w = 0.2
        # 5.1 初始化 road对象
        road_obj_list = get_road_obj(road_section_list)
        # 5.2 遍历road对象，得到road路段的等级level
        for road in road_obj_list:
            # 5.3 计算每个road隶属于C的概率
            for A_cluster in cluster_list:
                # 得到road属于第几聚类0-4
                cluster_i = get_cluster(road.index, A_cluster)
                # print("属于第", cluster_i, "聚类")
                # 将[P_C1,P_C2,P_C3,P_C4,P_C5]相应概率加0.2
                road.prob_list[cluster_i] += w
            print("第", road.index, "道路的各聚类概率为：", road.prob_list)
    
            # 5.4 得到概率最大的聚类等级level
            # 等级为0-4
            road_cluster_level = road.prob_list.index(max(road.prob_list))
            # 等级为1-5
            road.cluster = road_cluster_level + 1
    
        # 5.5 将相同的聚类结果合并在一起，得到各个聚类
        cluster_level_list = []
        for i in range(p):
            cluster_i = Cluster()
            cluster_i.set_i(i+1)
            cluster_level_list.append(cluster_i)
    
        # 5.6 遍历road
        for road in road_obj_list:
            for c in cluster_level_list:
                # 若路段的等级等于聚类的等级，则将此路段加入到这个聚类中ci
                if road.cluster == c.i:
                    c.set_level(road.index)
    
        # 5.7 输出等级聚类的结果
        for c in cluster_level_list:
            print("等级聚类的结果为：", c.get_level())
            print("等级聚类的相关度为：", c.get_G())
    
        # 5.8 计算类内隶属度
        SW_C = get_cluster_inner(road_section_list)
    #     5.9 计算类间分离度SB_C
        SB_C = get_cluster_outer(road_section_list)
        # 5.10 得到隶属度
        VS_C = SW_C + SB_C
        print("隶属度为：\n", VS_C)
