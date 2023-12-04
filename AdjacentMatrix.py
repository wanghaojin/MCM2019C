import numpy as np
import pandas as pd

def create_adjacency_matrix(adjacency_file, counties_list):
    # 读取邻接文件
    adjacency_data = pd.read_csv(adjacency_file)

    # 创建一个空的邻接矩阵
    adjacency_matrix = np.zeros((len(counties_list), len(counties_list)))

    # 建立县FIPS代码到矩阵索引的映射
    county_index = {county: index for index, county in enumerate(counties_list)}

    for _, row in adjacency_data.iterrows():
        county_fips = row['fipscounty']
        neighbor_fips = row['fipsneighbor']

        if county_fips in county_index and neighbor_fips in county_index:
            i, j = county_index[county_fips], county_index[neighbor_fips]
            adjacency_matrix[i][j] = 1  # 设置相邻县之间的连接

    return adjacency_matrix

def read_and_process_data_by_year(file_path):
    data = pd.read_excel(file_path, sheet_name='Data')

    # 获取所有县的FIPS代码
    all_counties = data['FIPS_Combined'].unique()

    return all_counties

# 从原始数据文件中获取所有县的FIPS代码
file_path = 'MCM_NFLIS_Data.xlsx'
all_counties = read_and_process_data_by_year(file_path)

# 邻接文件的路径
adjacency_file = 'county_adjacency2010.csv'

# 创建邻接矩阵
adj_matrix = create_adjacency_matrix(adjacency_file, all_counties)
# check = 0
# for i in range(adj_matrix.shape[0]):
#     for j in range(adj_matrix.shape[1]):
#         if(adj_matrix[i][j] == 1):
#             check += 1
#
# print(check)

save_path = 'adjacent_data'
np.save(save_path, adj_matrix)
print("Done!")