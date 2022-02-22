# 1. 获得归一化并保存两位小数的距离矩阵
# 下面函数是按照每一列进行归一化
def get_diatance_norm(D):
#   保留两位小数
    D_norm = D.apply(lambda x: ((x.max() - x)/(x.max() - x.min())).round(2))
    return D_norm

#  2. 若想要按照每一行进行归一化，可以转置之后，再按每一列进行归一化
#  可以使用T属性获得转置的pandas.DataFrame

# 3. 矩阵元素和一个常数相乘
# 如果其中一个是数值，那么这个数值会和DataFrame的每个位置上的数据进行相应的运算
def do_matrix():
  df = pd.DataFrame(val, columns = idx)
  print df
  print df * 2
  print df + 2
  


## 下面是GMRC聚类的相关代码  

# 3.计算相关度RA
RA = Gray(road_df)
print("灰色关联矩阵为：\n", RA)
# ShowGRAHeatMap(RA)

# 4.计算A0矩阵的关联相似度矩阵G
G = get_G(RA)
print(G)
# ShowGRAHeatMap(G)

# 5. 计算欧式距离矩阵D
D = get_distance(road_df)

# 6. 根据欧式距离特点，进行归一化
D_norm = get_diatance_norm(D)
print("归一化后的欧式距离矩阵为：\n", D_norm)

# 7. 根据灰色关联相似矩阵G 和欧式距离矩阵D ，权重系数为0.5，得到贴近度矩阵S
S = get_S(G, D_norm)
print("贴近度矩阵为：\n", S)

# 8. 聚类，定义的lambda=0.6
cluster_list = GRMC(S)
print("改进后的GMRC聚类结果为：\n", cluster_list)
