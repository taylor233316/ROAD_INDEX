def get_road_v_complex(car_v_ls, a):
    # 1. 假设第一个的平均速度为最大平均速度
    max_v = mean(car_v_ls[0][2])
    # print(car_v_ls)
    for car in car_v_ls:
        # print(car[2])
        avg_v = mean(car[2])
        if avg_v >= max_v:
            max_v = avg_v
    # 2. 获得前面的数字
    front_v = int(max_v / 10)
    after_v = int(max_v % 10)
    v_list = []

    # 3. 得到标准速度划分区间 [0,10],[10,20],[20,30],...,[90,100]
    for i in range(front_v + 1):
        # if i == front_v + 1:
        #     stand_v = (i + 1) * 10
        #     if after_v >= 5:
        #         five_stand_v = stand_v - 5
        #         v_list.append(five_stand_v)
        #         v_list.append(stand_v)
        #     else:
        #         five_stand_v = stand_v - 5
        #         v_list.append(five_stand_v)
        # else:
        stand_v = (i + 1) * 10
        # five_stand_v = stand_v - 5
        # v_list.append(five_stand_v)
        v_list.append(stand_v)

    print("v_list=", v_list)
    car_count_list = [0] * len(v_list)
    # 4. 统计处于各个时速区间的车辆数
    avg_v_ls = []
    for car in car_v_ls:
        avg_v = mean(car[2])
        avg_v_ls.append(avg_v)
        for j in range(len(v_list)):
            # 如果平均时速在这个时速区间内，则count+1
            if avg_v <= v_list[j]:
                car_count_list[j] = car_count_list[j] + 1
                break

    # 5.输出各时速的车辆个数
    print("各时速的车辆个数：", car_count_list)
    # 获得信息熵，即复杂程度
    et = get_entropy(car_count_list)

    k = len(car_count_list)
    avg = a / k
    avg_sum = math.pow(avg) * k
    
    # 将列表中0元素加1
    
    speed_dt_square = reduce(lambda x, y: x * y, car_count_list)

    square_sum = car_count_list

    s = pd.Series(car_count_list)

    # 方差×速度
    var_v = np.var(car_count_list)
    print("速度列表的方差为：", var_v)

    # 峰度计算
    kurt = s.kurt()

    # 偏度
    skew = s.skew()
    print("速度列表的峰度为：", kurt, ",偏度为：", skew)
    # car_v_ls = [float(i) for i in car_v_ls]

    # 使用四分位数，返回一个依次包含所有四分位数的列表
    four_number = np.array(car_count_list)
    four_25 = np.percentile(four_number, 10)
    four_75 = np.percentile(four_number, 90)
    # print("上分位点为：", four_75, "下分位点为：", four_25)
    # # min_avg_v = min(car_avg_v_list)
    # # max_avg_v = max(car_avg_v_list)
    road_complex_diff = four_75 - four_25

    road_complex = max(car_count_list) - min(car_count_list)
    # road_complex = road_complex / sum(car_count_list)
    return road_complex, car_count_list, kurt, skew, var_v, road_complex_diff, et
