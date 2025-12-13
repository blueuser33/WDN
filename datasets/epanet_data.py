from epyt import epanet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
import random
import math
import os
import json
from scipy.ndimage import gaussian_filter1d

class EpytHelper:
    def __init__(self, epa_net_path, hrs=168):
        print(f"Initializing EPANET model from: {epa_net_path}")
        self.epa_net_path = epa_net_path
        self.hrs = hrs
        self.raw_data = None

        # 生成缓存文件路径
        cache_dir = os.path.join(os.path.dirname(epa_net_path), 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(epa_net_path))[0]
        self.cache_file = os.path.join(cache_dir, f"{base_name}_sim_{hrs}hrs.npz")

        # 检查是否存在缓存文件
        if os.path.exists(self.cache_file):
            self._load_from_cache()
        else:
            try:
                self.G = epanet(epa_net_path)
            except Exception as e:
                raise

            # --- 单位检测与转换系数设定 ---
            self._detect_and_set_conversion_factors()

            # 运行仿真
            self.G.setTimeSimulationDuration(hrs * 3600)
            self.R = self.G.getComputedHydraulicTimeSeries()
            self.time_ranges = self.R.Pressure.shape[0]
            print(f"EPANET simulation complete for {self.time_ranges} timesteps.")

            # 缓存仿真结果
            self._save_to_cache()

            # 释放EPANET模型资源
            self.G.unload()

    def _detect_and_set_conversion_factors(self):
        """
        检测 .inp 文件中的单位制，并设置相应的SI转换系数。
        目标单位制: 流量(m³/s), 压力(m), 长度/直径(m)。
        """
        # 获取流量单位字符串，并转换为大写以进行不区分大小写的比较
        flow_units = self.G.getFlowUnits().upper()
        print(f"Detected flow units: {flow_units}")

        # 美制单位 (US Customary)
        us_units = ['CFS', 'GPM', 'MGD', 'IMGD', 'AFD']

        self.conversion = {}
        if flow_units in us_units:
            print("System identified as US Customary. Applying US to SI conversion factors.")

            # 流量转换
            if flow_units == 'GPM':
                self.conversion['flow'] = 6.30902e-5  # GPM to m³/s
            elif flow_units == 'CFS':
                self.conversion['flow'] = 0.0283168  # CFS to m³/s
            # ...可以为 MGD, IMGD, AFD 添加更多转换 ...
            else:
                raise ValueError(f"Unsupported US Customary flow unit: {flow_units}")

            # 长度、压力等单位
            self.conversion['pressure_to_head'] = 0.70325  # psi to m H₂O
            self.conversion['length'] = 0.3048  # ft to m
            self.conversion['diameter'] = 0.0254  # in to m

        else:  # 假设其他都是SI单位或接近SI单位
            print("System identified as SI-based. Applying SI-based to standard SI conversion factors.")

            # 流量转换
            if flow_units in ['LPS']:
                self.conversion['flow'] = 0.001  # LPS to m³/s
            elif flow_units in ['LPM']:
                self.conversion['flow'] = 1.66667e-5  # LPM to m³/s
            elif flow_units in ['CMH', 'M3/HR']:
                self.conversion['flow'] = 1 / 3600  # m³/h to m³/s
            elif flow_units in ['CMD']:
                self.conversion['flow'] = 1 / (3600 * 24)  # m³/d to m³/s
            elif flow_units in ['MLD']:
                self.conversion['flow'] = 1000 / (3600 * 24)  # MLD to m³/s
            else:
                raise ValueError(f"Unsupported SI-based flow unit: {flow_units}")

            # 长度、压力等单位
            # 在SI制下，压力通常直接是米水头 (m)，但要检查[OPTIONS]中的Pressure设置
            # 我们假设 epynet 总是返回米水头
            self.conversion['pressure_to_head'] = 1.0
            self.conversion['length'] = 1.0  # m to m
            self.conversion['diameter'] = 0.001  # mm to m

        # Hazen-Williams C系数无量纲，无需转换
        self.conversion['hazen_williams_c'] = 1.0

        print("Conversion factors to standard SI (m, s, m³, m H₂O) set:", self.conversion)

    def get_raw_data(self):
        """
        返回所有原始数据的字典，所有值都已转换为国际单位制 (SI)。
        """
        if self.raw_data is not None:
            # 如果已经从缓存加载过数据，直接返回
            return self.raw_data

        node_count = self.G.getNodeCount()
        reservoir_indices = [index - 1 for index in self.G.getNodeReservoirIndex()]
        node_type = [1 if i in reservoir_indices else 0 for i in range(node_count)]

        edge_index_list = []
        link_indices = self.G.getLinkIndex()
        for link_id in link_indices:
            start_node, end_node = self.G.getLinkNodesIndex(link_id)
            edge_index_list.append([start_node - 1, end_node - 1])

        # --- 在提取数据后立即应用转换 ---
        pressures = self.R.Pressure * self.conversion['pressure_to_head']
        flows = self.R.Flow * self.conversion['flow']
        demands = self.R.Demand * self.conversion['flow']

        elevations = np.array(self.G.getNodeElevations()) * self.conversion['length']
        # getNodeBaseDemands() 返回一个元组，我们需要第二个元素
        base_demands = np.array(self.G.getNodeBaseDemands()[1]) * self.conversion['flow']

        diameters = np.array([self.G.getLinkDiameter(i) for i in link_indices]) * self.conversion['diameter']
        lengths = np.array([self.G.getLinkLength(i) for i in link_indices]) * self.conversion['length']
        roughnesses = np.array([self.G.getLinkRoughnessCoeff(i) for i in link_indices])  # 无需转换

        self.raw_data = {
            # 动态数据 (SI)
            'pressures': pressures,  # [T, N], 单位: m
            'flows': flows,  # [T, E], 单位: m³/s
            'demand_real': demands,  # [T, N], 单位: m³/s

            # 静态节点数据 (SI)
            'node_elevations': elevations,  # [N], 单位: m
            'node_demands': base_demands,  # [N], 单位: m³/s

            # 拓扑和元数据
            'node_type': np.array(node_type),
            'reservoir_indices': np.array(reservoir_indices),
            'edge_index_directed': np.array(edge_index_list).T,

            # 静态边数据 (SI)
            'link_diameters': diameters,  # [E], 单位: m
            'link_lengths': lengths,  # [E], 单位: m
            'link_roughnesses': roughnesses  # [E], 无量纲
        }
        return self.raw_data

    def _save_to_cache(self):
        """
        将仿真结果保存到缓存文件
        """
        # 确保已经计算了raw_data
        if self.raw_data is None:
            self.raw_data = self.get_raw_data()

        # 使用numpy的savez函数保存所有数组
        np.savez_compressed(self.cache_file,
                            pressures=self.raw_data['pressures'],
                            flows=self.raw_data['flows'],
                            demand_real=self.raw_data['demand_real'],
                            node_elevations=self.raw_data['node_elevations'],
                            node_demands=self.raw_data['node_demands'],
                            node_type=self.raw_data['node_type'],
                            reservoir_indices=self.raw_data['reservoir_indices'],
                            edge_index_directed=self.raw_data['edge_index_directed'],
                            link_diameters=self.raw_data['link_diameters'],
                            link_lengths=self.raw_data['link_lengths'],
                            link_roughnesses=self.raw_data['link_roughnesses'])
        print(f"Simulation results saved to cache: {self.cache_file}")

    def _load_from_cache(self):
        """
        从缓存文件加载仿真结果
        """
        try:
            data = np.load(self.cache_file)
            self.raw_data = {
                'pressures': data['pressures'],
                'flows': data['flows'],
                'demand_real': data['demand_real'],
                'node_elevations': data['node_elevations'],
                'node_demands': data['node_demands'],
                'node_type': data['node_type'],
                'reservoir_indices': data['reservoir_indices'],
                'edge_index_directed': data['edge_index_directed'],
                'link_diameters': data['link_diameters'],
                'link_lengths': data['link_lengths'],
                'link_roughnesses': data['link_roughnesses']
            }
            # 计算时间范围
            self.time_ranges = self.raw_data['pressures'].shape[0]
        except Exception as e:
            print(f"Error loading from cache: {e}")
            # 删除损坏的缓存文件
            os.remove(self.cache_file)
            raise

    def destroy(self):
        if hasattr(self, 'G') and self.G is not None:
            self.G.unload()

# ep=EpytHelper('/data/zsm/case01/data/d-town.inp').get_raw_data()
# pressures = ep['pressures']   # shape: [T, N]
# T, N = pressures.shape
# print(T,N)
# # 想要绘制的节点索引
# node_ids = [0, 5, 10, 20]  # 可改为任何你关心的节点 ID
#
# for nid in node_ids:
#     print(f"\n=== Node {nid} 前 100 个压力值 ===")
#     print(pressures[:100, nid])
# plt.figure(figsize=(12, 6))
#
# for nid in node_ids:
#     plt.plot(pressures[:, nid], label=f"Node {nid}")
#
# plt.title("Pressure Time Series of Selected Nodes")
# plt.xlabel("Timestep (5-minute steps or your EPANET timestep)")
# plt.ylabel("Pressure (m H2O)")
# plt.legend()
# plt.grid(True)
# plt.show()