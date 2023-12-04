import numpy as np
import pandas as pd

def read_and_process_data_by_year(file_path):
    data = pd.read_excel(file_path, sheet_name='Data')
    results = {}

    # 获取所有年份中出现的毒品种类和地区（FIPS_Combined）
    all_drugs = data['SubstanceName'].unique()
    all_counties = data['FIPS_Combined'].unique()

    for year in data['YYYY'].unique():
        year_data = data[data['YYYY'] == year]
        # 确保每个年份的数据都包含所有毒品种类和地区
        grouped_data = year_data.groupby(['FIPS_Combined', 'SubstanceName']).size().unstack(fill_value=0)
        grouped_data = grouped_data.reindex(columns=all_drugs, fill_value=0)
        grouped_data = grouped_data.reindex(all_counties, fill_value=0)
        results[year] = grouped_data.to_numpy()

    return results

file_path = 'MCM_NFLIS_Data.xlsx'

# 处理数据并按年份分组
yearly_data = read_and_process_data_by_year(file_path)

# 为每个年份保存一个numpy数组
for year, drug_data_array in yearly_data.items():
    save_path = f'data_drug_{year}.npy'
    np.save(save_path, drug_data_array)
    # 可选：打印信息以确认保存
    print(f"Saved data for {year} in {save_path}")
