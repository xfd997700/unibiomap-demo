import zipfile

import yaml
import gradio as gr
import os
import json
import dgl
from utils import *
from os.path import join
import base64

desc_path_dict = {
    "compound": "database/unibiomap/compound_desc.json",
    "protein": "database/unibiomap/protein_desc.json",
    "pathway": "database/unibiomap/pathway_desc.json",
    "go": "database/unibiomap/go_desc.json",
    "disease": "database/unibiomap/disease_desc.json",
    "phenotype": "database/unibiomap/phenotype_desc.json"
}
# TODO:尝试打开项目直接启动一次检索，避免加载静态

color_map = {
    # 'complex': '#FFA07A',
    'compound': '#98FB98',
    'disease': '#FFD700',
    # 'genetic_disorder': '#FF69B4',
    'go': '#87CEEB',
    'pathway': '#DDA0DD',
    'phenotype': '#808080',
    'protein': '#FF6347'
}
node_size = 500
font_size = 10
font_color = "black"
results_root = "results/"
os.makedirs(results_root, exist_ok=True)

def load_or_process_graph():
    link_root = "database/unibiomap"
    data_root = "database/processed"
    os.makedirs(data_root, exist_ok=True)
    node_map_path = join(data_root, "node_map.json")
    graph_path = join(data_root, "unibiomap_simp.dgl")
    link_path = join(link_root, "unibiomap.links.tsv")

    def nodemap2idmap(node_map):
        return {k: {vv: kk for kk, vv in v.items()} for k, v in node_map.items()}

    if os.path.exists(node_map_path) and os.path.exists(graph_path):
        with open(node_map_path, "r") as f:
            node_map = json.load(f)
        graph = dgl.load_graphs(graph_path)[0][0]
    else:
        if not os.path.exists(link_root):
            download_raw_kg(link_root)
        graph, node_map = process_knowledge_graph(link_path, simplify_edge=True)
        dgl.save_graphs(graph_path, [graph])
        with open(node_map_path, "w") as f:
            json.dump(node_map, f)

    id_map = nodemap2idmap(node_map)
    return graph, node_map, id_map

graph, node_map, id_map = load_or_process_graph()
desc_dict = load_desc(desc_path_dict)

def fetch_input_id(input_string):
    if not input_string:
        return []
    return [item.strip() for item in input_string.split(",")]

def get_limit(mode, limit_val):
    return -1 if mode == "No Limit" else limit_val

def save_subgraph_and_metadata(sub_g, id_map_sub, must_show, save_dir="static"):
    os.makedirs(save_dir, exist_ok=True)
    # 保存 DGL 子图
    dgl.save_graphs(os.path.join(save_dir, "subgraph.dgl"), [sub_g])
    # 保存 id_map_sub
    with open(os.path.join(save_dir, "id_map_sub.json"), "w", encoding="utf-8") as f:
        json.dump(id_map_sub, f)
    # 保存 must_show
    with open(os.path.join(save_dir, "must_show.json"), "w", encoding="utf-8") as f:
        json.dump(must_show, f)

# 新增：公共函数，用于根据子图和限制条件生成 iframe HTML 以及生成的 html_code
def generate_iframe(sub_g, id_map_sub, must_show, display_limits):
    remove_self_loop = True
    G = convert_subgraph_to_networkx(sub_g, id_map_sub, display_limits, must_show, remove_self_loop)
    echarts_data = nx_to_echarts_json(G, color_map, desc_dict)
    html_code = generate_echarts_html(echarts_data)
    html_base64 = base64.b64encode(html_code.encode('utf-8')).decode('utf-8')
    data_uri = f"data:text/html;base64,{html_base64}"
    iframe_html = f"<iframe src='{data_uri}' width='100%' height='850px' style='border:none;'></iframe>"
    return iframe_html, html_code

def run_query(protein, compound, disease, pathway, go, depth,
            #   complex_mode, complex_limit,
              compound_mode, compound_limit,
              disease_mode, disease_limit,
            #   genetic_mode, genetic_limit,
              go_mode, go_limit,
              pathway_mode, pathway_limit,
              phenotype_mode, phenotype_limit,
              protein_mode, protein_limit):
    # 构造查询字典和显示限制
    sample_dict = {
        "protein": fetch_input_id(protein),
        "compound": fetch_input_id(compound),
        "disease": fetch_input_id(disease),
        "pathway": fetch_input_id(pathway),
        "go": fetch_input_id(go)
    }
    display_limits = {
        # 'complex': get_limit(complex_mode, complex_limit),
        'compound': get_limit(compound_mode, compound_limit),
        'disease': get_limit(disease_mode, disease_limit),
        # 'genetic_disorder': get_limit(genetic_mode, genetic_limit),
        'go': get_limit(go_mode, go_limit),
        'pathway': get_limit(pathway_mode, pathway_limit),
        'phenotype': get_limit(phenotype_mode, phenotype_limit),
        'protein': get_limit(protein_mode, protein_limit),
    }

    must_show = sample_dict.copy()
    try:
        print('start sampling')
        sub_g, new2orig, node_map_sub, statistics = subgraph_by_node(graph, sample_dict, node_map, depth=depth)
        id_map_sub = {k: {vv: kk for kk, vv in v.items()} for k, v in node_map_sub.items()}
        # save statistics as json
        with open(join(results_root, "statistics.yaml"), "w") as f:
            yaml.dump(statistics, f)

        statistics_text = "\n".join([f"{k}: {v}" for k, v in statistics.items()])
        
        # 储存subgraph
        # save_subgraph_and_metadata(sub_g, id_map_sub, must_show)
        # 调用公共函数生成展示 HTML
        iframe_html, html_code = generate_iframe(sub_g, id_map_sub, must_show, display_limits)

        return iframe_html, 'success', statistics_text, sub_g, id_map_sub, must_show
    except Exception as e:
        return f"Error: {str(e)}", f"Error: {str(e)}", sample_dict, None, None, None

def refresh_display(sub_g, id_map_sub, must_show,
                    # complex_mode, complex_limit,
                    compound_mode, compound_limit,
                    disease_mode, disease_limit,
                    # genetic_mode, genetic_limit,
                    go_mode, go_limit,
                    pathway_mode, pathway_limit,
                    phenotype_mode, phenotype_limit,
                    protein_mode, protein_limit):
    # 检查状态数据
    if sub_g is None or id_map_sub is None or must_show is None:
        return gr.update(value="<b>Please run query first.</b>")

    display_limits = {
        # 'complex': get_limit(complex_mode, complex_limit),
        'compound': get_limit(compound_mode, compound_limit),
        'disease': get_limit(disease_mode, disease_limit),
        # 'genetic_disorder': get_limit(genetic_mode, genetic_limit),
        'go': get_limit(go_mode, go_limit),
        'pathway': get_limit(pathway_mode, pathway_limit),
        'phenotype': get_limit(phenotype_mode, phenotype_limit),
        'protein': get_limit(protein_mode, protein_limit),
    }
    try:
        iframe_html, _ = generate_iframe(sub_g, id_map_sub, must_show, display_limits)
        return iframe_html
    except Exception as e:
        return f"Error updating display: {str(e)}"

# 删除 get_default_content 函数，不再需要
def get_default_content(take_empty=True):
    if take_empty:
        with open("static/gr_empty.html", "r", encoding="utf-8") as f:
            return f.read()
    with open("static/default.html", "r", encoding="utf-8") as f:
        default_html = f.read()
        html_base64 = base64.b64encode(default_html.encode('utf-8')).decode('utf-8')
        data_uri = f"data:text/html;base64,{html_base64}"
        iframe_html = f"<iframe src='{data_uri}' width='100%' height='600px' style='border:none;'></iframe>"
        return iframe_html
    
def get_text_content(file_path="static/gr_head.md"):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# def download_entity(sub_g, id_map_sub):
#     try:
#         report_subgraph(sub_g, id_map_sub, save_root=results_root)
#         path = join(results_root, 'triples.txt')
#         return gr.update(value=path, visible=True)
#     except Exception as e:
#         return gr.update(value=f"Error: {e}", visible=True)

def download_entity(sub_g, id_map_sub):
    try:
        # 生成统计数据和三元组文件
        report_subgraph(sub_g, id_map_sub, save_root=results_root)
        triples_path = join(results_root, 'triples.txt')
        statistics_path = join(results_root, 'statistics.yaml')

        # 创建一个压缩文件
        zip_path = join(results_root, 'results.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(triples_path, arcname='triples.txt')
            zipf.write(statistics_path, arcname='statistics.yaml')

        # 返回压缩文件路径
        return gr.update(value=zip_path, visible=True)
    except Exception as e:
        return gr.update(value=f"Error: {e}", visible=True)
    
# 新增：加载静态文件的函数，如果存在则加载保存的子图数据
def load_static_files():
    subgraph_file = "static/subgraph.dgl"
    must_show_file = "static/must_show.json"
    id_map_sub_file = "static/id_map_sub.json"
    if os.path.exists(subgraph_file) and os.path.exists(must_show_file) and os.path.exists(id_map_sub_file):
        sub_g = dgl.load_graphs(subgraph_file)[0][0]
        with open(must_show_file, "r", encoding="utf-8") as f:
            must_show = json.load(f)
        with open(id_map_sub_file, "r", encoding="utf-8") as f:
            id_map_sub = json.load(f)
        # 将 id_map_sub 中的键转换为整数
        id_map_sub = {ntype: {int(k): v for k, v in mapping.items()} for ntype, mapping in id_map_sub.items()}
        return sub_g, id_map_sub, must_show
    return None, None, None

def get_default_content(get_empty=True):
    if get_empty:
        with open("static/gr_empty.html", "r", encoding="utf-8") as f:
            return f.read()
    
    with open("static/default.html", "r", encoding="utf-8") as f:
        default_html = f.read()
        html_base64 = base64.b64encode(default_html.encode('utf-8')).decode('utf-8')
        data_uri = f"data:text/html;base64,{html_base64}"
        iframe_html = f"<iframe src='{data_uri}' width='100%' height='850px' style='border:none;'></iframe>"
        return iframe_html

# 尝试加载预存数据
sub_g_static, id_map_sub_static, must_show_static = load_static_files()

empty_display = get_default_content()
# 如果加载成功，则使用默认的 slider 默认值来生成初始展示内容
if sub_g_static is not None:
    initial_display = refresh_display(
        sub_g_static, id_map_sub_static, must_show_static,
        # "Set Limit", 10,  # complex
        "Set Limit", 10,  # compound
        "Set Limit", 10,  # disease
        # "Set Limit", 10,  # genetic_disorder
        "Set Limit", 10,  # go
        "Set Limit", 10,  # pathway
        "Set Limit", 10,  # phenotype
        "Set Limit", 10   # protein
    )
else:
    initial_display = "<b>Press Run Query button to start exploring!</b>"

with gr.Blocks() as demo:
    gr.HTML(get_text_content("static/gr_head.html"))
    gr.Markdown(get_text_content())

    # 使用预加载的展示内容作为初始值
    html_output = gr.HTML(value=get_default_content(get_empty=True))
    # 如果有预加载数据，则设置初始状态，否则为空
    subgraph_state = gr.State(value=sub_g_static)
    idmap_state = gr.State(value=id_map_sub_static)
    mustshow_state = gr.State(value=must_show_static)

    with gr.Row():
        with gr.Column():
            run_btn = gr.Button("▶ Run Query")
        with gr.Row():
            with gr.Column():
                down_btn = gr.Button("⬇️ Get All Queried Entities")
            with gr.Column():
                download_file = gr.File(label="Query triples file", interactive=False, visible=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Query Content")
            gr.Markdown("You can enter multiple entity IDs to query, separated by commas, for example: P50416, P05091.")
            protein_input = gr.Textbox("P05091", label="Protein ID")
            compound_input = gr.Textbox(label="Compound ID")
            disease_input = gr.Textbox(label="Disease ID")
            pathway_input = gr.Textbox(label="Pathway ID")
            go_input = gr.Textbox(label="GO ID")
            inputs_1 = [protein_input, compound_input, disease_input, pathway_input, go_input]
        with gr.Column():
            gr.Markdown("### Sample Limit")
            gr.Markdown("Setup the sampling restriction parameters below.")
            depth_slider = gr.Slider(0, 4, step=1, label="Subgraph Sampling Depth", value=1)
            with gr.Accordion("Display Limit", open=False):
                def slider_with_mode(label):
                    with gr.Row():
                        mode = gr.Radio(["Set Limit", "No Limit"], value="Set Limit", label=f"{label} Mode", interactive=True)
                        slider = gr.Slider(1, 20, step=1, value=10, label=label, visible=True)
                        def toggle_slider(mode_val):
                            return gr.update(visible=(mode_val == "Set Limit"))
                        mode.change(fn=toggle_slider, inputs=mode, outputs=slider)
                    return mode, slider

                # complex_mode, complex_limit = slider_with_mode("Complex")
                compound_mode, compound_limit = slider_with_mode("Compound")
                disease_mode, disease_limit = slider_with_mode("Disease")
                # genetic_mode, genetic_limit = slider_with_mode("Genetic Disorder")
                go_mode, go_limit = slider_with_mode("GO")
                pathway_mode, pathway_limit = slider_with_mode("Pathway")
                phenotype_mode, phenotype_limit = slider_with_mode("Phenotype")
                protein_mode, protein_limit = slider_with_mode("Protein")
                limit_inputs = [
                    # complex_mode, complex_limit,
                                compound_mode, compound_limit,
                                disease_mode, disease_limit,
                                # genetic_mode, genetic_limit,
                                go_mode, go_limit, pathway_mode, pathway_limit,
                                phenotype_mode, phenotype_limit, protein_mode, protein_limit]

    msg = gr.Textbox("OUTPUT INFO", label="INFO")
    debug = gr.Textbox("DEBUG", label="debug")

    run_btn.click(
        fn=run_query,
        inputs=inputs_1 + [depth_slider] + limit_inputs,
        outputs=[html_output, msg, debug, subgraph_state, idmap_state, mustshow_state]
    )

    for inp in limit_inputs:
        inp.change(
            fn=refresh_display,
            inputs=[subgraph_state, idmap_state, mustshow_state] + limit_inputs,
            outputs=html_output
        )

    down_btn.click(
        fn=download_entity,
        inputs=[subgraph_state, idmap_state],
        outputs=[download_file]
    )
    demo.launch()
