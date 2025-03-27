---
title: UniBioMap Explorer
emoji: 📊
colorFrom: blue
colorTo: pink
sdk: gradio
sdk_version: 5.23.1
app_file: app.py
pinned: false
---


# UniBioMap Demo
[![Open in Hugging Face](https://img.shields.io/badge/Open%20in-HuggingFace-orange?logo=huggingface)](https://huggingface.co/spaces/EZ4Fanta/unibiomap_demo)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xfd997700/unibiomap-demo/blob/main/demo.ipynb)
[![Open In GitHub](https://img.shields.io/badge/Open%20in-GitHub-black?logo=github)](https://github.com/xfd997700/unibiomap_demo)


📊 This demo demonstrates sampling and visualizing subgraphs from current compiled `UniBioMap` based on specific entities (sets).

## Usage
- We have released the <a href="https://www.gradio.app/">
  <img src="https://www.gradio.app/_app/immutable/assets/gradio.CHB5adID.svg" alt="Gradio" height="30">
</a> based demo on [Hugging Face](https://huggingface.co/spaces/EZ4Fanta/unibiomap_demo), please click the button to use the WebUI online.
- If you want to use the WebUI locally, please run:
    ```bash
    python app.py
    ```

- If you just want to compile the networkx graph image and entity .txt file, Simply following the steps in `demo.ipynb`. Or you can run the demo on [Google Colab]((https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xfd997700/unibiomap-demo/blob/main/demo.ipynb)).

## Requirements

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/unibiomap-demo.git
    cd unibiomap-demo
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```


