import datetime

from numpy import zeros, mean

import process_date
from two_bezier.Util.Cluster import Cluster
from two_bezier.choose_ID import read_csv
from two_bezier.clean_data import listdir
from two_bezier.index.FCM import FCM_Cluster
from two_bezier.index.GMRC import GRMC
from two_bezier.index.get_weight import get_entropy_weight, get_score
from two_bezier.index.gray import GRA, Gray, ShowGRAHeatMap, get_G
from two_bezier.index.guass import get_dist, get_hist
from two_bezier.transform_time import cal_time
from vector import choose_ID
from two_bezier import vector_extract
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import transform_distance


def get_date(date):
    l = []
    for t in date:
        # 转换成时间数组
        timeArray = time.strptime(t, "%Y-%m-%d %H:%M:%S")
        # 转换成时间戳
        timestamp = time.mktime(timeArray)
        l.append(timestamp)
    return l


def get_tra_graph(R):
    time_list = []
    x_list = []
    y_list = []

    for v in R:
        # print('v的最后一个点的坐标', v.get_f(v.t_end))
        unit_time = 15
        t = v.t_start

        while t < v.t_end:
            time_list.append(float(t))
            t += unit_time
        # middle_t = (v.t_start + v.t_end)/2
        # time_list.append(middle_t)
        time_list.append(float(v.t_end))
        # print('画图的开始时间为：', v.t_start, '结束时间为：', v.t_end)
    point_list = []
    for t in time_list:
        x, y = vector_extract.trajectory_point(t, R)
        point = (float(x), float(y))
        point_list.append(point)
        x_list.append(float(x))
        y_list.append(float(y))
    print(point_list)
    # new a figure and set it into 3d
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    #
    # # set figure information
    # ax.set_title("predict points")
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("time")

    # draw the figure, the color is r = read
    # figure = ax.plot(x_list, y_list, time_list, c='r')

    #     画图
    # print(x_list)

    plt.plot(x_list, y_list)
    x1, y1 = v.pa[0], v.pa[1]
    x2, y2 = v.pb[0], v.pb[1]
    print(x1, y1)
    plt.plot(x1, y1, 'o', c='g')
    plt.plot(x2, y2, 'o', c='g')
    data1 = pd.read_csv('clean.csv', header=None, sep=',')
    lines = data1[data1[0].isin(["京BN0386"])]
    x = lines[1]
    y = lines[2]
    date = lines[3]
    # print(x)
    # print('real x', x[0:1])
    # print('real y', y[0:1])
    new_date = get_date(date)

    plt.plot(x, y, c='pink')
    plt.show()


def get_cars(filename):
    # [[x,y],[x1,y1]]
    rows = read_csv(filename)
    car_list = []
    for row in rows:
        date = row[3]
        year, month, day, hour, minute, second = process_date.split_datetime(date)
        if year != '2013' and month != '5' and day != '31':
            continue
        row = [row[1], row[2], row[3]]
        car_list.append(row)
    return car_list


def get_area_cars(file_name):
    car_list = read_csv(file_name)
    data = pd.DataFrame(car_list)

    p_cols = ['name', 'long', 'lat', 't', 'speed']
    data.columns = p_cols
    # data = data.drop(['number'], axis=1)
    print(data)
    west = 116.4546
    south = 39.8833
    east = 116.4560
    north = 39.893655
    querySer = (data.loc[:, 'long'] > str(west)) & (data.loc[:, 'long'] < str(east)) & (
            data.loc[:, 'lat'] < str(north)) & (data.loc[:, 'lat'] > str(south))
    # 应用查询条件
    print('删除异常值前：', data.shape)
    data = data.loc[querySer, :]
    print('删除异常值后：', data.shape)
    area_cars = data['name'].values.tolist()
    area_cars_list = [i for i in area_cars if i != '']
    # print(area_cars_list)
    return area_cars_list


def choose_ID_data(ID, file_name):
    # [[x,y],[x1,y1]]
    rows = read_csv(file_name)
    car_list = []
    for row in rows:
        if row[1] == ID:
            date = row[4]
            # print(date)
            year, month, day, hour, minute, second = process_date.split_datetime(date)
            if year != '2013' and month != '5' and day != '31':
                continue
            row = [row[1], row[2], row[3], row[4], row[5]]
            car_list.append(row)
    return car_list


def get_bounds_car_vector(area_cars_name, file_name):
    new_cars_list = []
    for car_name in area_cars_name:
        car_list = choose_ID_data(car_name, file_name)
        # print("****************")
        # print(car_list)
        # 3. 处理时间格式数据
        car_list = choose_ID.new_date(car_list)
        # 预处理
        data1 = pd.DataFrame(car_list)
        p_cols1 = ['long', 'lat', 't']
        data1.columns = p_cols1
        data1.drop_duplicates('t', 'first', inplace=True)
        car_list = data1.values.tolist()
        new_car_list = [car_name, car_list]
        new_cars_list.append(new_car_list)
    # for car in new_cars_list:
        # print("选择的出租车" + car[0] + "轨迹为：")
        # print(car[1])



    # 3.矢量提取函数
    # 定义阈值大小
    dist = 0.01
    # 向量的集合：[['京B1',[v1,v2,v3]],]
    V_list = []
    for car in new_cars_list:
        # print("--------------- "+car[0] + "--------------- ")
        V = vector_extract.vector_extraction(car[1], dist)
        # 若只有一个轨迹点，不存在矢量，则不存取
        if len(V) != 0:
            car_vector = [car[0], V]
            V_list.append(car_vector)

    # print("存储的矢量集合如下：\n")
    # print(V_list)
    #
    # for i in V_list:
    #     print(i[0] + "的矢量个数为：")
    #     print(len(i[1]))

    # bounds
    west = 116.4546
    south = 39.8833
    east = 116.4560
    north = 39.893655
    # 筛选在bounds范围内的矢量
    bounds_car_list = []
    for car in V_list:
        # 获得矢量集合
        V = car[1]
        v_list = []
        # print("汽车：")
        # print(car[0])
        for v in V:
            point1 = v.get_f(v.t_start)
            point2 = v.get_f(v.t_end)

            if (west < float(point1[0]) < east and south < float(point1[1]) < north) or (
                    west < float(point2[0]) < east and south < float(point2[1]) < north):
                v_list.append(v)
        if len(v_list) != 0:
            car_name = car[0]
            bound_car = [car_name, v_list]
            bounds_car_list.append(bound_car)
    return bounds_car_list


def get_car_v_list(bounds_car_list):
    # 计算矢量速度序列：先计算两个矢量的
    # 1. 速度为起始点的距离除以时间t
    for car in bounds_car_list:
        # car[1]代表矢量的列表
        for vector in car[1]:
            # print("vector为：", vector)
            t_start = vector.t_start
            t_end = vector.t_end
            point_start = vector.get_real_f(t_start)
            point_end = vector.get_real_f(t_end)
            # print("起点为：", point_start)
            # print("终点为：", point_end)
            #     经纬度转为距离,单位为km   1.4862km
            distance = transform_distance.haversine(float(point_start[1]), float(point_start[0]),
                                                    float(point_end[1]), float(point_end[0]))
            # print("矢量起始点之间的距离为：", distance)

            #     计算矢量起始点的时间,以小时为单位
            time = cal_time(t_start, t_end)
            #  计算速度
            if distance == 0:
                v1 = 0
            else:
                v1 = distance / time
            v = round(v1, 2)
            # 设置vector的速度v
            vector.set_v(v)
            # print("矢量的速度为：", v)

    # 开始做相似度匹配,
    # 1. 得到速度序列
    car_v_list = []
    for car in bounds_car_list:
        # car[1]代表矢量的列表
        v_list = []
        for vector in car[1]:
            # print("矢量的速度为：", vector.v)
            v_list.append(vector.v)
        car_v = [car[0], car[1], v_list]
        car_v_list.append(car_v)

    # print
    # for car_v in car_v_list:
    #     print("出租车", car_v[0], "的速度序列为：", car_v[2])

    return car_v_list


def get_avg_list(car_v_list):
    v_ratio_list = []
    v_list = []
    avg_list = []
    # 1. 计算每辆车的平均速度，得到平均速度序列
    for car_v in car_v_list:
        print("出租车", car_v[0], "的速度序列为：", car_v[2])
        sum_v = 0
        v_ratio = mean(car_v[2]) / max(car_v[2])
        avg_v = mean(car_v[2])

        if v_ratio == 0:
            v_ratio1 = 0
        else:
            v_ratio1 = 1 / v_ratio
        v_ratio_list.append(v_ratio1)

        if avg_v == 0:
            avg_v1 = 0
        else:
            avg_v1 = 1 / avg_v
        v_list.append(avg_v1)

        #
        # 获得每辆车的平均速度
        avg_list.append(avg_v)
    return avg_list


def get_road_v(v_list):
    sum = 0
    for v in v_list:
        sum = sum + v
    avg_v = sum / len(v_list)
    return avg_v


def get_car_v(file_name):
    # 1. 读入文件
    # car_list = read_csv(file_name)
    # 预处理
    car_list = read_csv(file_name)
    data = pd.DataFrame(car_list)
    p_cols = ['number', 'name', 'long', 'lat', 't', 'speed']
    data.columns = p_cols
    data = data.drop(['number'], axis=1)
    area_cars_name1 = data['name']
    area_cars_name = set(area_cars_name1)
    # print("*******    提取区域的车辆名称   *******", area_cars_name)
    # area_cars_name1 = get_area_cars(file_name)
    # area_cars_name = set(area_cars_name1)
    # print("提取到的车辆名为：", area_cars_name)
    bounds_car_list = get_bounds_car_vector(area_cars_name, file_name)
    # 2. 获得在road范围内的矢量集合
    # print("******* bounds内的矢量集合如下 *********\n")
    # print(bounds_car_list)
    # 3.计算矢量速度序列
    car_v_list = get_car_v_list(bounds_car_list)

    return car_v_list


def get_road_avg_v(car_v_list):
    # 获得平均速度矢量序列
    v_list = get_avg_list(car_v_list)
    # print("平均速度序列为：", v_list)
    # 计算道路平均速度
    road_avg_v = get_road_v(v_list)
    # print("道路的平均速度为：", road_avg_v)
    return road_avg_v


def get_car_complex_list(car_v_list):
    car_complex_list = []
    car_var_list = []
    car_var_sum = 0
    car_avg_sum = 0
    for car_v in car_v_list:
        car_var = np.var(car_v[2])
        car_var_sum = car_var_sum + car_var
        car_avg = np.mean(car_v[2])
        car_avg_sum = car_avg_sum + car_avg
        # car_complex = car_var / car_avg
        # car_var_list.append(car_var)
        # car_complex_list.append(car_complex)
    car_avg_var = car_var_sum / len(car_v_list)
    car_avg_v = car_avg_sum / len(car_v_list)
    road_complex = car_avg_var / car_avg_v
    # print("各辆车的方差为：", car_var_list)
    # print("各辆车的复杂程度为：", car_complex_list)

    # 计算道路平均复杂程度
    # road_avg_complex = np.mean(car_complex_list)
    return road_complex


def get_car_complex_second_list(car_v_list):
    car_complex_list = []
    car_var_list = []
    car_nums_sum = 0
    car_avg_sum = 0
    for car_v in car_v_list:
        car_nums = len(car_v[2])
        car_nums_sum = car_nums_sum + car_nums
        car_avg = np.mean(car_v[2])
        car_avg_sum = car_avg_sum + car_avg
        # # 道路越畅通，复杂度越小
        # car_complex = car_nums / car_avg
        # # car_var_list.append(car_var)
        # car_complex_list.append(car_complex)
    # print("各辆车的方差为：", car_var_list)
    car_avg_nums = car_nums_sum / len(car_v_list)
    car_avg_v = car_avg_sum / len(car_v_list)
    car_complex = car_avg_nums / car_avg_v
    print("道路的复杂程度为：", car_complex)
    return car_complex


def transform_timestamp(file_name):
    pass


if __name__ == '__main__':
    # 1.获得清洗后的所有文件
    list_name = []
    # 文件夹路径
    path = "./data/"
    list_name = listdir(path, list_name)
    # 2. 获得每个文件，东三环道路每个时间段的平均速度、复杂度
    time_avg_v_ls = []
    road_complex_list = []
    road_vector_complex = []

    for file_name in list_name:
        print("此时处理的文件是：", file_name)
        # 预处理
        # 获取到速度列表
        car_v_list = get_car_v(file_name)
        # 获取到每辆车的  var/v
        road_avg_complex = get_car_complex_list(car_v_list)
        road_complex_list.append(road_avg_complex)
        # 获取道路的nums/v
        road_avg_vector_complex = get_car_complex_second_list(car_v_list)
        road_vector_complex.append(road_avg_vector_complex)
        # 获取到T时间的道路平均速度
        road_avg_v = get_road_avg_v(car_v_list)
        # 获取到1天内道路平均速度
        time_avg_v_ls.append(road_avg_v)

    print("各时间的道路平均速度为：", time_avg_v_ls)
    print("各时间的道路平均复杂程度为：", road_complex_list)
    print("各时间的道路矢量平均复杂程度为：", road_vector_complex)

    # file_name = "H:/trafficData/clean_0601/clean_YTAX_20130601000040_4016674_CH_DB.csv"
    # # 获取到速度列表
    # car_v_list = get_car_v_list(file_name)
    # # 获取到每辆车的速度方差/v
    # car_complex_list = get_car_complex_list(car_v_list)
    # # 获取到T时间的道路平均速度
    # road_avg_v = get_road_avg_v(car_v_list)


    # # 2. 得到矢量个数序列
    # vector_list = []
    # for car_v in car_v_list:
    #     vector_list.append(len(car_v[2]))
    #
    # # 3. 归一化
    # # 3.1 速度
    # v_max = max(v_list)
    # v_min = min(v_list)
    # v_norm_list = []
    # for v in v_list:
    #     v_norm = (v - v_min) / (v_max - v_min)
    #     v_norm_list.append(v_norm)
    #     print("标准化后的速度", v_norm)
    # # 3.2 矢量个数
    # vector_max = max(vector_list)
    # vector_min = min(vector_list)
    # vector_nums_norm = []
    # for vector in vector_list:
    #     vector_norm = (vector - vector_min) / (vector_max - vector_min)
    #     vector_nums_norm.append(vector_norm)
    #     print("标准化后的矢量个数", vector_norm)
    # # 3.3 ratio
    # ratio_max = max(v_ratio_list)
    # ratio_min = min(v_ratio_list)
    # ratio_norm_list = []
    # for ratio in v_ratio_list:
    #     ratio_norm = (ratio - ratio_min) / (ratio_max - ratio_min)
    #     ratio_norm_list.append(ratio_norm)
    #     print("标准化后的ratio", ratio_norm)
    # # 4. 按照速度比：v/v_max，计算隶属度
    # # v_ratio_list
    #
    # # 5. 计算权重
    # # 5.1 变为数据框
    # from pandas.core.frame import DataFrame
    #
    # c = {"vector": vector_nums_norm,
    #      "avg_v": v_list}
    # data = DataFrame(c)
    # w_list = get_entropy_weight(data)
    # print("权重为：", w_list)
    # score_list = get_score(data, w_list)
    # data['score'] = score_list
    # # 然后对数据框按得分从大到小排序
    # result = data.sort_values(by='score', axis=0, ascending=False)
    # result['rank'] = range(1, len(result) + 1)
    # print(result)
    # # 计算平均复杂程度
    # avg_complexity = result['score'].mean()
    # print("这条路段的平均复杂程度为：", avg_complexity)
    # m, n = result.shape
    #
    # # 6. 计算车流量
    # # 车流量 = 车辆 / 时间，时间设为30min
    # cars = m - 1
    # traffic_flow = cars / 0.5
    # print("这个路段的平均复杂程度为", avg_complexity, ",车流量为：", traffic_flow)
    # road_section1 = [avg_complexity, traffic_flow]
    #
    # # 7. 模拟数据
    # import random
    #
    # road_section_list = []
    # for c in range(100):
    #     road_section_list.append([random.random(), random.randint(10, 200)])
    #
    # road_section_list.append(road_section1)
    # i = 1
    # for road_section in road_section_list:
    #
    #     print("第", i, "条道路的信息为：", road_section)
    #     i = i + 1
    #
    # # 8. 归一化
    # import pandas as pd
    # from numpy import *
    #
    # road_df1 = DataFrame(road_section_list)
    # # 取出矩阵中最大值与最小值
    # max_loc = road_df1.max()
    # max_comp = max_loc[0]
    # max_flow = max_loc[1]
    # # print("max_comp、max_flow为", max_comp, max_flow)
    # road_section_list.insert(0, [max_comp, max_flow])
    # road_df2 = DataFrame(road_section_list)
    # # road_df = road_df.values
    # # print("未进行归一化：", road_df2)
    # road_df_norm = (road_df2 - road_df2.min()) / (road_df2.max() - road_df2.min())
    # # print("进行归一化：", road_df_norm)
    # print("标准化后的道路路段信息为：\n", road_df_norm)
    #
    # # 9.计算相关度,
    # RA = Gray(road_df_norm)
    # # print("灰色关联矩阵为：\n", RA)
    #
    # # ShowGRAHeatMap(RA)
    #
    # # 10.计算G 关联相似度矩阵
    # G = get_G(RA)
    # print(G)
    # # ShowGRAHeatMap(G)
    #
    # # 11. 聚类
    # cluster_list = GRMC(G)
    #
    # # 12. 画出分布函数图
    # # get_hist(road_df_norm, 0)
    # # get_dist(road_df_norm, 1)
    #
    # # 13. 判断lambda的值，选择0.8 和 0.7两个值，计算隶属度
    # lambda1 = 0.8
    # k = 5
    # road_df_norm = road_df_norm[1:]
    # FCM_Cluster(cluster_list, road_df_norm, k)





    # cluster_list = GRMC(G1)
    #
    # for cluster in cluster_list:
    #     print(cluster.get_X())


    ##
    # num_list = []
    # for car in bounds_car_list:
    #     # print(car)
    #     print(car[0] + "的矢量个数为：", len(car[1]))
    #     num_list.append(len(car[1]))
    # vectors_count = set(num_list)
    # print("矢量个数为：", vectors_count)

    # 相似度判断，进行聚类。
    # 1. 一开始每个car为一类，聚类的标准为：矢量个数的多少，如果数量相似，则聚为一类。
    # 对于同一类，再进行聚类，再计算（每个矢量的
    # list1 = []
    # list2 = []
    # for car in bounds_car_list:
    #     if len(car[1]) == 1:
    #         list1.append(car)
    #     elif len(car[1]) == 2:
    #         list2.append(car)
    #     else:
    #         print("矢量个数不是1或者2.")
    #
    # print("矢量个数为1的车为：", list1)
    # print("矢量个数为2的车为：", list2)



    # 2. 使用相关系数作为相似度度量
    from sklearn.metrics.pairwise import cosine_similarity

    # S = zeros((len(car_v_list), len(car_v_list)))
    # for i in range(len(car_v_list)):
    #     for j in range(len(car_v_list)):
    #         S[i][j] = cosine_similarity([car_v_list[i][2]], [car_v_list[j][2]])
    #         if S[i][j] > 0.999:
    #             print("这两个速度序列为：", car_v_list[i][2], car_v_list[j][2])
    #             # print("相似度为：", round(S[i][j], 2))
    #
    # #  遍历S 相似度矩阵
    # rows, cols = S.shape#读取X的行列数
    # for i in range(rows):
    #     for j in range(i+1, rows):
    #         if S[i][j] > 0.99:
    #             print("S", S[i][j])


    # 计算道路的平均速度
    # for v in car_v[2]:
    #     sum_v = sum_v + v
    # avg_v = sum_v / len(car_v[2])
    # # [京A123, 20]
    # print("出租车", car_v[0], "的平均速度为：", avg_v)
    # v_ratio = avg_v /
    # v_list.append(avg_v)






    # 3. 聚类
    # my = Hierarchical(4)
    # my.fit(S)
    # print(np.array(my.labels))
    # for node in my.nodes:
    #     print(node.distance)
    # print(my.nodes)

    # for i in range(len(S)):
    #     for j in range(i+1, len(S)):
    #         if S[i][j] > 0.99:
    #             print("存在可以匹配的矢量序列")

    # 对矢量进行评价
    # 1. 先暂时定义在这条道路2km范围内的矢量个数为1000，定义拥堵的阈值为0.01，定义速度20km/h为拥堵

    # 1.1 矢量拥堵定义
    # f1 = len(bounds_car_list) / 1000
    # 单位时间的平均矢量个数最大为4，最小为1
    # f1 = abs()
    # print("f1为：", f1)
    #
    # # 1.2 计算矢量的平均速度为30
    # f2 = 30
    #
    # # 1.3 归一化
    #
    # # 1.4 定义综合拥堵评价指标
    # congestion = f1 * 0.5 + f2 * 0.5
    #
    # # 当定义的指标的值大于某个阈值时，则为拥堵
    # if congestion > 0.5:
    #     print("crowd!")

    # print("********* 提取到的矢量集合为 ***********")
    # print(R)
    #     # 4. 输入时间time，输出坐标（x,y）
    #     # time = input("input:")
    #     # x, y = vector_extract.trajectory_point(time, R)
    #     # print('在时间为', time, '时，坐标为：', (x, y))
    #     # 5 .画出预测的轨迹图
    #     get_tra_graph(R)



#     文件备份
# for file_name in list_name:
#     print("此时处理的文件是：", file_name)
# 获取到采集的时间T
# time_str = file_name.split("20130601")[1]
# time = time_str[0:4]
# hour, min = time[0:2], time[2:4]
# t = hour + ":" + min

### ***************************
# file_name = "./test/GPS_Data_0601.csv"
# # 按照时间分段后存为csv文件。
# car_list = read_csv(file_name)
# data = pd.DataFrame(car_list)
# p_cols = ['name', 'long', 'lat', 't', 'speed']
# data.columns = p_cols
#
# index = 0
# while index <= 4005:
#     # 获取csv文件第一行
#     head_time = data.iloc[index][3]
#     print("分段的第一行时间为：", head_time)
#     time_str = head_time.split(" ")[1]
#     attr = time_str.split(":")
#     hour, min, sec = attr[0], attr[1], attr[2]
#     # print(hour, min, sec)
#     # hour, min = time[0:2], time[2:4]
#     unit_time = 15
#     min = int(min)
#     hour = int(hour)
#     if min + 15 > 60:
#         hour = hour + 1
#         after_min = 15 - (60 - min)
#     else:
#         after_min = min + unit_time
#     year_m_d = head_time.split(" ")[0] + " " + str(hour) + ":" + str(after_min) + ":" + sec
#     timeArray = time.strptime(year_m_d, "%Y-%m-%d %H:%M:%S")
#     # 转换成时间戳
#     timestamp1 = time.mktime(timeArray)
#     print("加上15分钟后的时间为：", year_m_d)
#     # print(timestamp1)
#     # s_date = datetime.datetime.strptime(year_m_d, "%Y-%m-%d %H:%M:%S").date()
#     # 将df某一列转为list
#     date = data['t'].tolist()
#     date = date[index:]
#     print("第一个日期为：", date[0])
#     new_date = []
#     index_tmp = index
#     # 将日期转为时间戳形式
#     for i in date:
#         # s_date = datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S").date()
#         timeArray = time.strptime(i, "%Y-%m-%d %H:%M:%S")
#         # 转换成时间戳
#         timestamp = time.mktime(timeArray)
#         if timestamp >= timestamp1:
#             break
#         index = index + 1
#     print("按照十五分钟进行分段，获得第一块分段数据的最后一个index索引值", index)
#     split_data = data.iloc[index_tmp:index]
#     split_data.to_csv("0601_" + str(index) + ".csv")
### ***************************

# print(data.iloc[0:index])
# data.iloc[0:index].to_csv("0601" + index + ".csv")
# 将时间转为时间戳进行比较
#
# car_list = transform_timestamp(file_name)
#
#
#
#
# data['t'] = pd.to_numeric+(data['t'])
# result = data[data['t'] >= s_date].index.tolist()
# print(result)
# 清洗数据，如果一辆车两个点时间相同，则去除一个
# for file_name in list_name:
#     print("此时处理的文件是：", file_name)
#     car_list = read_csv(file_name)
#     data = pd.DataFrame(car_list)
#     p_cols = ['number', 'name', 'long', 'lat', 't', 'speed']
#     data.columns = p_cols
#     data = data.drop(['number'], axis=1)
#     area_cars_name1 = data['name']
#     area_cars_name = set(area_cars_name1)
#     print("*******    提取区域的车辆名称   *******", area_cars_name)
#     for car_name in area_cars_name:
#         #  获取到一辆车的GPS点序列
#         car_list = choose_ID_data(car_name, file_name)
#         print("****************")
#         print(car_list)
#         # 3. 处理时间格式数据
#         car_list = choose_ID.new_date(car_list)
#         data1 = pd.DataFrame(car_list)
#         p_cols1 = ['long', 'lat', 't']
#         data1.columns = p_cols1
#         data1.drop_duplicates('t', 'first', inplace=True)
#         print(data1.values.tolist())
