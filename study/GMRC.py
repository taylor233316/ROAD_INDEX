def GRMC(G):
    G1 = G.sort_values(by=0, ascending=False, axis=1)
    # 1.取出第一行G0
    G_0 = G1.loc[0]
    # print("G_0为：", G_0)
    print("排序的结果：", G_0)
    G_numpy = G.values
    # 2. m = G_0.shape[0]
    m = G_0.shape[0]
    print("G的第一行的维度为：\n", m)
    a = zeros([m - 1, 3])
    for i in range(m - 1):
        if i != 0:
            j = G_0.index[i]
            front_index = G_0.index[i]
            after_index = G_0.index[i + 1]

            a[i - 1, 0] = G_numpy[front_index, after_index]
            a[i - 1, 1] = front_index
            a[i - 1, 2] = after_index

    # 3.暂定lambda=0.8
    lambda1 = 0.6
    clusters = 4
    j = 0
    Cluster_list = []
    # cluster存放索引index
    while j <= clusters:
        if j != clusters:
            cluster_list1, a1 = cluster_mem(a, lambda1, Cluster_list)
            Cluster_list = cluster_list1
            if isinstance(a1, DataFrame):
                a = a1.values
            else:
                a = a1
        else:
            index_X = []
            G0_list = []
            m, n = a.shape
            for index in range(0, m):
                before_index = a[index, 1]
                G0_list.append(a[index, 0])
                index_X .append(before_index)
            c_1 = Cluster()
            c_1.set_index(index_X)
            c_1.set_G(G0_list)
            Cluster_list.append(c_1)
        j = j + 1

    for cluster in Cluster_list:
        print(cluster.get_index())

    return Cluster_list
