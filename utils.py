import html
import json
import dgl
import torch
from collections import defaultdict
from tqdm import tqdm
import os
import networkx as nx
import gdown
import sys
import zipfile
# import matplotlib.pyplot as plt

# file_id = "1tUe3YVyA2K2Xh_GORWYaOGEKyYE5vnAp"
# url = f"https://drive.google.com/uc?id={file_id}"
url = "https://github.com/xfd997700/unibiomap_demo/releases/download/dev/unibiomap.zip"
def download_raw_kg(link_root):
    os.makedirs(link_root, exist_ok=True)
    # Download the file from GitHub and show the progress
    file_path = os.path.join(link_root, "unibiomap.zip")
    gdown.download(url, file_path, quiet=False)
    # use zipfile unzip the file and delete the zip file
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(file_path))
    os.remove(file_path)
    print(f"Downloaded and extracted files to {os.path.dirname(link_root)}")

def load_desc(desc_path_dict):
    """
    Load the description files and return a dictionary of descriptions.
    """
    desc_dict = {}
    for key, path in desc_path_dict.items():
        with open(path, 'r') as f:
            cur_dict = json.load(f)
        desc_dict[key] = {sys.intern(k): v for k, v in cur_dict.items()}
    desc_dict['pathway'] = repair_smpdb_name(desc_dict['pathway'])
    return desc_dict

def nodemap2idmap(node_map):
    return {k: {vv: kk for kk, vv in v.items()} for k, v in node_map.items()}

def process_knowledge_graph(file_path, simplify_edge=False):
    """
    Process the knowledge graph data and return a DGL graph object.
    """
    node_map = defaultdict(dict)        # 节点类型到名称-ID映射
    current_ids = defaultdict(int)
    edges_dict = defaultdict(lambda: ([], []))
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc='fetching data', unit=' entries'):
            row = line.strip().split('\t')
            h_type, t_type = row[0], row[1]
            h_name, rel, t_name = row[2], row[3], row[4]
            if simplify_edge:
                if rel == "HAS_METABOLITE":
                    rel = "protein_metabolite"
                else:
                    rel = h_type + "-" + t_type
            if h_name not in node_map[h_type]:
                node_map[h_type][h_name] = current_ids[h_type]
                current_ids[h_type] += 1
            h_id = node_map[h_type][h_name]

            if t_name not in node_map[t_type]:
                node_map[t_type][t_name] = current_ids[t_type]
                current_ids[t_type] += 1
            t_id = node_map[t_type][t_name]

            edge_type = (h_type, rel, t_type)
            edges_dict[edge_type][0].append(h_id)
            edges_dict[edge_type][1].append(t_id)

    hetero_data = {
        et: (torch.tensor(heads), torch.tensor(tails))
        for et, (heads, tails) in edges_dict.items()
    }
    g = dgl.heterograph(hetero_data)

    print("Node type counts:")
    for ntype in g.ntypes:
        print(f"{ntype}: {g.num_nodes(ntype)}")

    print("\nEdge type counts:")
    for etype in g.canonical_etypes:
        print(f"{etype}: {g.num_edges(etype)}")

    return g, node_map


def degree_search(graph, node_type, node_name, node_map):
    """
    Print out-degree and in-degree statistics of a node.
    """
    if node_name not in node_map[node_type]:
        print(f"Node {node_name} does not exist in type {node_type}.")
        return

    node_id = node_map[node_type][node_name]
    print(f"\nNode Statistics - {node_type}-{node_name} (ID {node_id}):")

    total_out = sum(
        graph.out_degrees(node_id, etype=etype)
        for etype in graph.canonical_etypes if etype[0] == node_type
    )
    total_in = sum(
        graph.in_degrees(node_id, etype=etype)
        for etype in graph.canonical_etypes if etype[2] == node_type
    )
    print("Total out-degree:", total_out)
    print("Total in-degree:", total_in)

    print("Out-degrees by edge type:")
    for etype in graph.canonical_etypes:
        if etype[0] == node_type:
            deg = graph.out_degrees(node_id, etype=etype)
            if deg > 0:
                print(f"  {etype}: {deg}")

    print("In-degrees by edge type:")
    for etype in graph.canonical_etypes:
        if etype[2] == node_type:
            deg = graph.in_degrees(node_id, etype=etype)
            if deg > 0:
                print(f"  {etype}: {deg}")


def analyze_connections(graph, sample_dict, id_map):
    connection_stats = {}
    for node_type, node_ids in sample_dict.items():
        for node_id in node_ids:
            node_stats = {"connected_nodes": {}, "connected_edges": {}}
            for etype in graph.canonical_etypes:
                # 统计出度
                if etype[0] == node_type:
                    neighbors = graph.successors(node_id, etype=etype).tolist()
                    node_stats["connected_nodes"].setdefault(etype[2], 0)
                    node_stats["connected_nodes"][etype[2]] += len(neighbors)
                    node_stats["connected_edges"].setdefault(etype, 0)
                    node_stats["connected_edges"][etype] += len(neighbors)
                # 统计入度
                if etype[2] == node_type:
                    neighbors = graph.predecessors(node_id, etype=etype).tolist()
                    node_stats["connected_nodes"].setdefault(etype[0], 0)
                    node_stats["connected_nodes"][etype[0]] += len(neighbors)
                    node_stats["connected_edges"].setdefault(etype, 0)
                    node_stats["connected_edges"][etype] += len(neighbors)
            connection_stats[(node_type, id_map[node_type][node_id])] = node_stats
    return connection_stats



def subgraph_by_node(graph, sample_dict, node_map, depth=1,
                     relabel_nodes=True):
    """
    Get a subgraph centered around a specific node.
    Parameters:
        - graph: The input DGL graph.
        - sample_dict: A dictionary of node names to sample. The keys are node types.
        - node_map: The node name to ID mapping.
        - depth: The depth of the subgraph
    Output:
        - full_g: The subgraph centered around the node.
    """
    cur_id_map = {}
    # print(f"Getting subgraph from: {sample_dict}")
    for node_type, node_names in sample_dict.items():
        for node_name in node_names:
            if node_name not in node_map[node_type]:
                print(f"Node {node_name} does not exist in type {node_type}.")
                return
        # convert node names to node IDs
        sample_dict[node_type] = [node_map[node_type][node_name] for node_name in node_names]
        cur_id_map[node_type] = {node_map[node_type][node_name]: node_name for node_name in node_names}

    connection_stats = analyze_connections(graph, sample_dict, cur_id_map)

    print("Getting out subgraph1...")
    out_g, _ = dgl.khop_out_subgraph(graph, sample_dict, k=depth,
                                        relabel_nodes=True, store_ids=True)
    print("Getting in subgraph...")
    in_g, _ = dgl.khop_in_subgraph(graph, sample_dict, k=depth,
                                    relabel_nodes=True, store_ids=True)
    
    # 收集所有节点类型的原始 ID（合并去重）
    print("收集所有节点类型的原始 ID（合并去重）")
    all_nodes = {}
    for ntype in graph.ntypes:
        out_nids = out_g.nodes[ntype].data[dgl.NID] if ntype in out_g.ntypes else torch.tensor([], dtype=torch.int64)
        in_nids = in_g.nodes[ntype].data[dgl.NID] if ntype in in_g.ntypes else torch.tensor([], dtype=torch.int64)
        combined = torch.cat([out_nids, in_nids]).unique()
        all_nodes[ntype] = combined
    
    # 直接从原始图提取包含这些节点的子图
    print("直接从原始图提取包含这些节点的子图")
    if not relabel_nodes:
        full_g = dgl.node_subgraph(graph, all_nodes, relabel_nodes=False)
        return full_g
    
    full_g = dgl.node_subgraph(graph, all_nodes, relabel_nodes=True, store_ids=True)

    # TODO: 此处暂时使用 relabel_nodes=True 和 ID 重映射的策略，AI 模型中可以去除，直接使用全节点
    # === 构建新 ID 到原始 ID 的映射 ===
    print("构建新 ID 到原始 ID 的映射")
    new2orig = defaultdict(dict)
    for ntype in full_g.ntypes:
        orig_ids = full_g.nodes[ntype].data[dgl.NID].tolist()
        for new_id, orig_id in enumerate(orig_ids):
            new2orig[ntype][new_id] = orig_id

    # === 构建 full_g 的新 node_map（名字 -> 新 ID） ===
    print("构建 full_g 的新 node_map（名字 -> 新 ID）")
    new_node_map = {}
    for ntype in full_g.ntypes:
        # 构建 id -> name 的反向映射
        relevant_ids = set(full_g.nodes[ntype].data[dgl.NID].tolist())
        id_to_name = {v: k for k, v in node_map[ntype].items() if v in relevant_ids}

        # id_to_name = {v: k for k, v in node_map[ntype].items()}

        orig_ids = full_g.nodes[ntype].data[dgl.NID].tolist()
        new_node_map[ntype] = {}

        for new_id, orig_id in enumerate(orig_ids):
            node_name = id_to_name.get(orig_id)
            if node_name is not None:
                new_node_map[ntype][node_name] = new_id

    # new_node_map = {}
    # for ntype in full_g.ntypes:
    #     orig_ids = full_g.nodes[ntype].data[dgl.NID].tolist()
    #     new_node_map[ntype] = {}
    #     for new_id, orig_id in enumerate(orig_ids):
    #         for name, id_val in node_map[ntype].items():
    #             if id_val == orig_id:
    #                 new_node_map[ntype][name] = new_id
    #                 break

    # print("Node type counts:")
    # for ntype in full_g.ntypes:
    #     print(f"{ntype}: {full_g.num_nodes(ntype)}")

    # print("Edge type counts:")
    # for etype in full_g.canonical_etypes:
    #     print(f"{etype}: {full_g.num_edges(etype)}")
    
    return full_g, new2orig, new_node_map, connection_stats

def report_subgraph(graph, id_map, save_root='static'):
    entities = defaultdict(list)
    for ntype in graph.ntypes:
        for nid in graph.nodes(ntype).tolist():
            entities[ntype].append(id_map[ntype][nid])
    
    triplets = []
    for etype in graph.canonical_etypes:
        src_type, relation, dst_type = etype
        src_ids, dst_ids = graph.edges(etype=etype)
        
        # 将每条边作为三元组加入列表
        for src, dst in zip(src_ids.tolist(), dst_ids.tolist()):
            src = id_map[src_type][src]
            dst = id_map[dst_type][dst]
            triplets.append((src, relation, dst))

    print(f"Total triplets: {len(triplets)}")

    if save_root:
        os.makedirs(save_root, exist_ok=True)
        with open(os.path.join(save_root, "entities.json"), "w") as f:
            json.dump(entities, f)
        with open(os.path.join(save_root, "triples.txt"), "w") as f:
            for triplet in triplets:
                f.write(f"{triplet[0]}\t{triplet[1]}\t{triplet[2]}\n")
    return entities, triplets
    


def convert_subgraph_to_networkx(sub_g, id_map,
                                 display_limits, must_show,
                                 remove_self_loop=True):
    # 筛选各类型中需要显示的节点
    displayed_nodes = {}
    for ntype in sub_g.ntypes:
        num_nodes = sub_g.number_of_nodes(ntype)
        all_node_ids = list(range(num_nodes))
        # 先筛选必须显示的节点（通过名称匹配）
        must_nodes = [nid for nid in all_node_ids 
                      if id_map.get(ntype, {}).get(nid, None) in must_show.get(ntype, [])]
        selected = set(must_nodes)
        limit = display_limits.get(ntype, -1)
        # 如果有数量限制，则在必须显示的基础上补足其它节点，直到达到限制
        if limit != -1:
            for nid in all_node_ids:
                if len(selected) >= limit:
                    break
                selected.add(nid)
        else:
            selected = set(all_node_ids)
        displayed_nodes[ntype] = selected

    # 构造 NetworkX 图（节点标识符采用 "类型_编号" 格式，确保唯一性）
    G = nx.Graph()

    # 添加节点：设置 label 和 title 为 id_map 对应的名称，同时保存节点所属类型
    for ntype, node_ids in displayed_nodes.items():
        for nid in node_ids:
            node_label = id_map.get(ntype, {}).get(nid, f"{ntype}_{nid}")
            node_id = f"{ntype}_{nid}"
            G.add_node(node_id, label=node_label, title=node_label, group=ntype)

    # 添加边：仅保留两端节点都在显示集合内的边
    for canonical_etype in sub_g.canonical_etypes:
        src_type, etype, dst_type = canonical_etype
        # 获取当前边类型的边列表
        src, dst = sub_g.edges(etype=etype)
        src = src.tolist()
        dst = dst.tolist()
        for u, v in zip(src, dst):
            if u in displayed_nodes[src_type] and v in displayed_nodes[dst_type]:
                src_node = f"{src_type}_{u}"
                dst_node = f"{dst_type}_{v}"
                G.add_edge(src_node, dst_node, title=etype)

    # 移除自环边
    if remove_self_loop:
        G.remove_edges_from(nx.selfloop_edges(G))

    return G

# TODO: SMPDB中存在一些（极少，3对儿）名称和描述完全一致但是 ID 不一致的通路，实际检索其通路图可能不同
# 暂时在其name后加上ID以进行区分，以避免绘制ECharts时的冲突
def repair_smpdb_name(pathway_dict):
    smp_dict = {
        k: v['name'] for k, v in pathway_dict.items() if k.startswith('SMP')
    }  
    reverse_map = defaultdict(list)
    
    for key, value in smp_dict.items():
        reverse_map[value].append(key)
    
    # 筛选出有多个key的value
    result = [values for values in reverse_map.values() if len(values) > 1]
    result = [item for sublist in result for item in sublist]

    for k in result:
        pathway_dict[k]['name'] = pathway_dict[k]['name'] + f" ({k})"
    return pathway_dict


# === Generate HTML with ECharts ===
def generate_echarts_html(echarts_data):
    data = json.loads(echarts_data)

    html_code = f"""
    <div id=\"main\" style=\"width:100%;height:100%;\"></div>
    <script src=\"https://cdn.jsdelivr.net/npm/echarts/dist/echarts.min.js\"></script>
    <script>
        var chartDom = document.getElementById('main');
        var myChart = echarts.init(chartDom);
        var option = {{
            tooltip: {{
                formatter: function (params) {{
                    if (params.data.type == "edge") {{
                        // 这是边
                        var source_show = params.data?.source_show;
                        var target_show = params.data?.target_show;
                        var source_group = params.data?.source_group;
                        var target_group = params.data?.target_group;
                        return `<div style="max-width:400px;">
                                    <div style="font-weight:bold;">
                                        <span style="color:#999; font-size:12px;">
                                            ${{source_group + ' -- ' + target_group}}
                                        </span>
                                        <br>
                                        ${{source_show + ' -- ' + target_show}}
                                    </div>
                                </div>`;
                    }}

                    var color = params.color || '#000';
                    var name = params.data.tooltip_name || params.data.name;
                    var desc = params.data.tooltip_desc || params.data.value;
                    return `<div>
                                <div style="display:flex;align-items:center;">
                                    <span style="display:inline-block;width:10px;height:10px;
                                                border-radius:5px;background:${{color}};margin-right:6px;"></span>
                                    <span style="font-weight:bold;
                                        max-width: 400px;
                                        white-space: nowrap;
                                        overflow: hidden;
                                        text-overflow: ellipsis;
                                        display: inline-block;">
                                        ${{name}}
                                    </span>
                                </div>
                                <div style="margin-left:16px; max-width:400px;
                                    white-space:normal; word-wrap:break-word;
                                    overflow:hidden; text-overflow:ellipsis;
                                    display:-webkit-box; -webkit-line-clamp:10; -webkit-box-orient:vertical;">
                                    ${{desc}}
                                </div>
                            </div>`;
                }}
            }},
            legend: [{{
                data: {json.dumps([cat['name'] for cat in data['categories']])},

            }}],
            series: [{{
                type: 'graph',
                layout: 'force',
                roam: true,
                label: {{ show: true, position: 'right' }},
                edgeSymbol: ['none', 'none'],
                data: {json.dumps(data['nodes'])},
                links: {json.dumps(data['links'])},
                categories: {json.dumps(data['categories'])},
                force: {{
                    repulsion: 800,
                    edgeLength: 120
                }},
                emphasis: {{
                    focus: 'adjacency',
                    label: {{ show: true }}
                }}
            }}]
        }};
        myChart.setOption(option);

        myChart.on('click', function (params) {{
            if (params.data && params.data.url) {{
                window.open(params.data.url, '_blank');
            }}
        }});
        // ✅ 添加响应式图表大小
        window.addEventListener('resize', function () {{
            myChart.resize();
            myChart.setOption(option);  // 强制触发重新布局
        }});
        
    </script>
    """
    return html_code

def get_url_by_id(id, group):
    base_url = "https://identifiers.org/"
    if group == "protein":
        id = 'uniprot:' + id
    if group == "compound":
        id = id.split(":")[-1]
        return f"https://www.ebi.ac.uk/unichem/compoundsources?type=uci&compound={id}"
    if group == "disease":
        id = id.split(':')[1]
        return f"https://uts.nlm.nih.gov/uts/umls/concept/{id}"
    if group == "pathway":
        if id.startswith("R-"):
            id = 'reactome:' + id
        elif id.startswith("SMP"):
            id = 'smpdb:' + id
        else:
            return f"https://www.kegg.jp/pathway/{id}"
    
    url = base_url + id
    return url

def fetch_desc_info(nid, ntype, desc_dict):
    url = get_url_by_id(nid, ntype) if ntype!='other' else None

    cur_desc = desc_dict.get(ntype, {}).get(sys.intern(nid), {})
    if ntype == 'protein':
        nname = cur_desc.get('entry_name', nid)
        if nname is None: nname = nid
        ndesc = cur_desc.get('protein_name', nid)
        if ndesc is None: ndesc = nid
        node_desc = {
            "name": nname,
            "category": ntype,
            "desc": ndesc,
            "url": url,
            "tooltip_name": nname + '<br>' + nid,
            "tooltip_desc": ndesc
        }
    elif ntype == 'compound':
        nname = cur_desc.get('name', nid)
        if nname is None: nname = cur_desc.get('inchikey', nid)
        ndesc = cur_desc.get('smiles', nid)
        node_desc = {
            "name": nid,
            "category": ntype,
            "desc": ndesc,
            "url": url,
            "tooltip_name": nname + '<br>' + nid,
            "tooltip_desc": ndesc
        }
    else:
        nname = cur_desc.get('name', nid)
        ndesc = cur_desc.get('definition', nname)
        node_desc = {
            "name": nname,
            "category": ntype,
            "desc": ndesc,
            "url": url,
            "tooltip_name": nname + '<br>' + nid,
            "tooltip_desc": html.escape(ndesc).replace('\n', '<br>')
        }

    return node_desc
    # elif ntype == 'compound':
    #     nname = 



# === Convert NX Graph to ECharts JSON ===
def nx_to_echarts_json(G, color_map, desc_dict,
                       base_size=12, max_ratio=2.5):
    nodes = []
    links = []

    degrees = dict(G.degree())
    min_deg = min(degrees.values()) or 1
    max_deg = max(degrees.values())

    def scale(deg):
        if max_deg == min_deg:
            return base_size
        norm = (deg - min_deg) / (max_deg - min_deg)
        return base_size + norm * base_size * (max_ratio - 1)
    
    for node in G.nodes(data=True):
        label = node[1].get("label", node[0])
        id = node[0]
        deg = degrees[node[0]]
        group = node[1].get("group", "other")

        node_desc = fetch_desc_info(label, group, desc_dict)
        node_desc["type"] = "node"
        node_desc['value'] = id
        node_desc['symbolSize'] = scale(deg)
        nodes.append(node_desc)

        node[1]['ident'] = node_desc['name']

        # url = get_url_by_id(label, group) if group!='other' else None
        # nodes.append({
        #     "name": label,
        #     "value": id,
        #     "category": group,
        #     "symbolSize": scale(deg),
        #     "url": url
        # })
    for edge in G.edges():
        source_label = G.nodes[edge[0]].get("ident", edge[0])
        source_show = G.nodes[edge[0]].get("label", False)
        source_group = G.nodes[edge[0]].get("group", "other")
        target_label = G.nodes[edge[1]].get("ident", edge[1])
        target_show = G.nodes[edge[1]].get("label", False)
        target_group = G.nodes[edge[1]].get("group", "other")
        links.append({
            "type": "edge",
            "source": source_label,
            "target": target_label,
            "source_show": source_show,
            "target_show": target_show,
            "source_group": source_group,
            "target_group": target_group,
        })
    categories = [{"name": cat, "itemStyle": {"color": color_map.get(cat, '#ccc')}} for cat in color_map]
    return json.dumps({"nodes": nodes, "links": links, "categories": categories}, ensure_ascii=False)

