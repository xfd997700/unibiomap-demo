import gradio as gr
import os
import json
import dgl
import torch
import networkx as nx
import matplotlib.pyplot as plt
from utils import *
from os.path import join
import base64

color_map = {
    'complex': '#FFA07A',
    'compound': '#98FB98',
    'disease': '#FFD700',
    'genetic_disorder': '#FF69B4',
    'go': '#87CEEB',
    'pathway': '#DDA0DD',
    'phenotype': '#808080',
    'protein': '#FF6347'
}
node_size = 500
font_size = 10
font_color = "black"
edge_color = "gray"
results_root = "results/"

# === Step 1: Load or Process Graph ===
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
        if not os.path.exists(link_path):
            download_raw_kg(link_path)
        graph, node_map = process_knowledge_graph(link_path, simplify_edge=True)
        dgl.save_graphs(graph_path, [graph])
        with open(node_map_path, "w") as f:
            json.dump(node_map, f)

    id_map = nodemap2idmap(node_map)
    return graph, node_map, id_map

graph, node_map, id_map = load_or_process_graph()


def fetch_input_id(input_string):
    if not input_string:
        return []
    return [item.strip() for item in input_string.split(",")]

def get_limit(limit, inf):
    if inf:
        return -1
    return limit

# === Step 4: Main Gradio Function ===
def run_query(protein, compound, disease, pathway, go, depth,
              complex_limit, complex_inf,
            compound_limit, compound_inf,
            disease_limit, disease_inf,
            genetic_limit, genetic_inf,
            go_limit, go_inf,
            pathway_limit, pathway_inf,
            phenotype_limit, phenotype_inf,
            protein_limit, protein_inf):
    
    sample_dict = {
        "protein": fetch_input_id(protein),
        "compound": fetch_input_id(compound),
        "disease": fetch_input_id(disease),
        "pathway": fetch_input_id(pathway),
        "go": fetch_input_id(go)
        }
    display_limits = {
    'complex': get_limit(complex_limit, complex_inf),
    'compound': get_limit(compound_limit, compound_inf),
    'disease': get_limit(disease_limit, disease_inf),
    'genetic_disorder': get_limit(genetic_limit, genetic_inf),
    'go': get_limit(go_limit, go_inf),
    'pathway': get_limit(pathway_limit, pathway_inf),
    'phenotype': get_limit(phenotype_limit, phenotype_inf),
    'protein': get_limit(protein_limit, protein_inf),
    }

    must_show = sample_dict.copy()
    remove_self_loop = True
    try:
        sub_g, new2orig, node_map_sub = subgraph_by_node(graph, sample_dict, node_map, depth=depth)
        id_map_sub = nodemap2idmap(node_map_sub)
        report_subgraph(sub_g, id_map_sub, save_root=results_root)

        G = convert_subgraph_to_networkx(sub_g, id_map_sub, display_limits, must_show, remove_self_loop)
        echarts_data = nx_to_echarts_json(G, color_map)
        html_code = generate_echarts_html(echarts_data)
        # Encode to base64
        html_base64 = base64.b64encode(html_code.encode('utf-8')).decode('utf-8')
        data_uri = f"data:text/html;base64,{html_base64}"
        iframe_html = f"<iframe src='{data_uri}' width='100%' height='850px' style='border:none;'></iframe>"
        return iframe_html, 'success', html_code
    except Exception as e:
        return f"Error: {str(e)}", f"Error: {str(e)}", sample_dict, html_code

def get_default_content():
    with open("static/default.html", "r", encoding="utf-8") as f:
      default_html = f.read()
      html_base64 = base64.b64encode(default_html.encode('utf-8')).decode('utf-8')
      data_uri = f"data:text/html;base64,{html_base64}"
      iframe_html = f"<iframe src='{data_uri}' width='100%' height='850px' style='border:none;'></iframe>"
      return iframe_html

def get_text_content(file_path = "static/gr_head.md"):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def download_entity():
    path = os.path.join(results_root, 'triples.txt')
    return gr.update(value=path, visible=True)



with gr.Blocks() as demo:
    gr.HTML(get_text_content("static/gr_head.html"))
    gr.Markdown(get_text_content())

    html_output = gr.HTML(value=get_default_content())

    with gr.Row():
        with gr.Column():
            run_btn = gr.Button("▶ Run Query")
        with gr.Row():
            with gr.Column():
                down_btn = gr.Button("⬇️ Get All Queried Entities")
            with gr.Column():
                download_file = gr.File(label="Query triples file",
                                        interactive=False, visible=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Query Content")
            gr.Markdown("You can enter multiple entity IDs to query, separated by commas, for example: P50416, P05091.")
            protein_input = gr.Textbox("P05091", label="Protein ID")
            compound_input = gr.Textbox(label="Compound ID")
            disease_input = gr.Textbox(label="Disease ID")
            pathway_input = gr.Textbox(label="Pathway ID")
            go_input = gr.Textbox(label="GO ID")
            
            inputs_1=[protein_input, compound_input, disease_input, pathway_input, go_input]
        
        with gr.Column():
            gr.Markdown("### Sample Limit")
            gr.Markdown("Setup the sampling restriction parameters below.")
            depth_slider = gr.Slider(0, 4, step=1, label="Subgraph Sampling Depth", value=1)
            with gr.Accordion("Display Limit", open=False):
                def slider_with_inf(label):
                    with gr.Row():
                        slider = gr.Slider(1, 20, step=1, value=10, label=label)
                        checkbox = gr.Checkbox(label="Inf", value=False)
                    return slider, checkbox

                complex_limit, complex_inf = slider_with_inf("Complex")
                compound_limit, compound_inf = slider_with_inf("Compound")
                disease_limit, disease_inf = slider_with_inf("Disease")
                genetic_limit, genetic_inf = slider_with_inf("Genetic Disorder")
                go_limit, go_inf = slider_with_inf("GO")
                pathway_limit, pathway_inf = slider_with_inf("Pathway")
                phenotype_limit, phenotype_inf = slider_with_inf("Phenotype")
                protein_limit, protein_inf = slider_with_inf("Protein")

                inputs_2 = [depth_slider,
                            complex_limit, complex_inf,
                            compound_limit, compound_inf,
                            disease_limit, disease_inf,
                            genetic_limit, genetic_inf,
                            go_limit, go_inf,
                            pathway_limit, pathway_inf,
                            phenotype_limit, phenotype_inf,
                            protein_limit, protein_inf]
    
    msg = gr.Textbox("OUTPUT INFO", label="INFO")
    debug = gr.Textbox("DEBUG", label="debug")


    inputs = inputs_1 + inputs_2
    run_btn.click(
        fn=run_query,
        inputs=inputs,
        outputs=[html_output, msg, debug]
    )


    down_btn.click(
        fn=download_entity,
        outputs=[download_file]
    )
    demo.launch()