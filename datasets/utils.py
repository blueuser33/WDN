from torch_geometric.utils import to_networkx, degree
from torch_geometric.data import Data
import networkx as nx
import torch
def preprocess_for_graph_transformer(graph_topology, max_spd_cutoff=5):
    """
    为图Transformer模型计算一次性的、与拓扑相关的结构信息。

    Args:
        graph_topology (torch_geometric.data.Data):
            一个包含图拓扑信息的Data对象 (至少需要 .edge_index 和 .num_nodes)。
        max_spd_cutoff (int): 计算最短路径时的最大路径长度截断，以防大图计算过慢。

    Returns:
        dict: 一个包含 'degree_encoding', 'spd_matrix', 'edge_map' 的字典。
    """
    print("Preprocessing for Graph Transformer...")

    num_nodes = graph_topology.num_nodes
    edge_index = graph_topology.edge_index
    num_edges = graph_topology.num_edges

    # 1. 计算节点度数中心性 (Degree Centrality)
    # 度的log变换可以稳定数值
    deg = degree(edge_index[0], num_nodes).float()
    degree_encoding = deg.view(-1, 1)
    print("  - Degree encoding calculated.")

    # 2. 计算最短路径距离 (Shortest Path Distance)
    # to_networkx需要一个包含num_nodes信息的Data对象
    temp_data_for_nx = Data(edge_index=edge_index, num_nodes=num_nodes)
    G = to_networkx(temp_data_for_nx, to_undirected=True)

    path_lengths = dict(nx.all_pairs_shortest_path_length(G, cutoff=max_spd_cutoff))

    spd_matrix = torch.full((num_nodes, num_nodes), float('inf'))
    for i, paths in path_lengths.items():
        for j, length in paths.items():
            spd_matrix[i, j] = length

    # 将inf替换为一个比cutoff大的整数，方便后续embedding
    spd_matrix[spd_matrix == float('inf')] = max_spd_cutoff + 1
    spd_matrix = spd_matrix.long()
    print("  - Shortest path distance matrix calculated.")

    # 3. 创建边索引到边特征的映射
    # 这步是为了在Transformer层中方便地通过(i, j)找到边的特征
    # 注意：你的图是双向的，这里我们只映射一个方向，或者需要更复杂的处理
    # 简单的处理方式是，假设edge_attr的顺序与edge_index对应
    edge_map = torch.full((num_nodes, num_nodes), -1, dtype=torch.long)
    edge_map[edge_index[0], edge_index[1]] = torch.arange(num_edges)
    edge_map[edge_index[1], edge_index[0]] = torch.arange(num_edges)
    print("  - Edge map created.")

    structural_data = {
        'degree_encoding': degree_encoding,
        'spd_matrix': spd_matrix,
        'edge_map': edge_map
    }

    print("Preprocessing finished.")
    return structural_data