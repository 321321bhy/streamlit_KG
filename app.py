import streamlit as st
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import spacy
import random
import string
import os

# -------------------------
# Setup spaCy and load model
# -------------------------
nlp = spacy.load("en_core_web_sm")

# -------------------------
# Helper Functions for Proposition Extraction and Graph Building
# -------------------------
def tokenize_text(text):
    """Remove punctuation, lowercase, and split text into tokens."""
    translator = str.maketrans('', '', string.punctuation)
    return set(text.translate(translator).lower().split())

def extract_proposition_from_sentence(sentence, source_type, sentence_index):
    """
    Extract a simple proposition from a sentence.
    Returns a dict with:
      - full_sentence: the original sentence,
      - main_predicate: the ROOT verb (if found),
      - main_arguments: list of tokens with dep in (nsubj, dobj, iobj),
      - sub_propositions: list of dicts (each with keys "type" and "content"),
      - source_type and sentence_index.
    """
    doc = nlp(sentence)
    result = {
        "full_sentence": sentence,
        "main_predicate": "",
        "main_arguments": [],
        "sub_propositions": [],
        "source_type": source_type,
        "sentence_index": sentence_index
    }
    # Find the ROOT verb as the main predicate.
    root = None
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            root = token
            break
    if root:
        result["main_predicate"] = root.text
    # Main arguments: tokens with dependency nsubj, dobj, iobj.
    for child in root.children if root else []:
        if child.dep_ in ("nsubj", "dobj", "iobj"):
            result["main_arguments"].append(child.text)
    # Sub-propositions:
    # Time: advmod tokens like "yesterday", "today", "tomorrow".
    for token in doc:
        if token.dep_ == "advmod" and token.text.lower() in {"yesterday", "today", "tomorrow"}:
            result["sub_propositions"].append({"type": "time", "content": token.text})
    # Place: pobj of prepositions "in", "at", "on".
    for token in doc:
        if token.dep_ == "pobj" and token.head.lemma_ in {"in", "at", "on"}:
            result["sub_propositions"].append({"type": "place", "content": token.text})
    # Attribute: amod modifiers on main arguments.
    for token in doc:
        if token.dep_ == "amod":
            if root and token.head.text.lower() in [arg.lower() for arg in result["main_arguments"]]:
                result["sub_propositions"].append({"type": "attribute", "content": token.text})
    return result

def build_graph_from_propositions(propositions_list):
    """
    Build a NetworkX DiGraph from a list of proposition dicts.
    For each proposition (sentence):
      - Main predicate node: ID = "{source}_S{index}_pred" (red)
      - Main argument nodes: ID = "{source}_S{index}_arg_{i}" (blue)
      - Sub-proposition nodes: ID = "{source}_S{index}_sub_{i}" 
          (colored by subtype: time/orange, place/green, attribute/purple, other/gray)
    Directed edges:
      - From each main argument node to the main predicate node.
      - From each sub-proposition node to the main predicate node.
    Each node is assigned a group equal to its source type and a cluster key (source+sentence_index).
    Returns:
      - G: the NetworkX DiGraph,
      - node_cluster: dict mapping node id -> cluster key,
      - color_map: dict mapping node id -> color.
    """
    G = nx.DiGraph()
    color_map = {}
    node_cluster = {}
    main_pred_color = "red"
    main_arg_color = "blue"
    sub_color_map = {"time": "orange", "place": "green", "attribute": "purple"}
    default_sub_color = "gray"
    
    for record in propositions_list:
        source = record.get("source_type", "Unknown")
        sent_index = record.get("sentence_index", "0")
        main_pred = record.get("main_predicate", "")
        main_args = record.get("main_arguments", [])
        sub_props = record.get("sub_propositions", [])
        cluster_key = f"{source}_S{sent_index}"
        
        # Main predicate node.
        pred_node = f"{cluster_key}_pred"
        G.add_node(pred_node, label=main_pred, node_type="main_predicate", source=source, cluster=cluster_key)
        color_map[pred_node] = main_pred_color
        node_cluster[pred_node] = cluster_key
        
        # Main argument nodes.
        for i, arg in enumerate(main_args):
            arg_node = f"{cluster_key}_arg_{i}"
            G.add_node(arg_node, label=arg, node_type="main_argument", source=source, cluster=cluster_key)
            color_map[arg_node] = main_arg_color
            node_cluster[arg_node] = cluster_key
            G.add_edge(arg_node, pred_node, relation="argument")
        
        # Sub-proposition nodes.
        for i, sub in enumerate(sub_props):
            sub_type = sub.get("type", "").lower()
            sub_content = sub.get("content", "")
            sub_node = f"{cluster_key}_sub_{i}"
            label = f"{sub_type}: {sub_content}"
            G.add_node(sub_node, label=label, node_type="sub_proposition", subtype=sub_type, source=source, cluster=cluster_key)
            color_map[sub_node] = sub_color_map.get(sub_type, default_sub_color)
            node_cluster[sub_node] = cluster_key
            G.add_edge(sub_node, pred_node, relation="sub_proposition")
    return G, node_cluster, color_map

def build_pyvis_network(G, color_map):
    """
    Create and return a Pyvis Network from a NetworkX graph.
    Each node is assigned a group based on its source type.
    """
    net = Network(height="800px", width="100%", directed=True)
    net.barnes_hut()
    for node, data in G.nodes(data=True):
        net.add_node(n_id=node,
                     label=data["label"],
                     color=color_map[node],
                     group=data.get("source", "default"))
    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, label=data.get("relation", ""))
    return net

# -------------------------
# Session State Setup (start with empty propositions)
# -------------------------
if "props" not in st.session_state:
    st.session_state.props = []  # list of proposition dicts
if "sentence_counters" not in st.session_state:
    st.session_state.sentence_counters = {"Source": 0, "Think-Aloud": 0, "Final Essay": 0}

def clear_graph():
    st.session_state.props = []
    st.session_state.sentence_counters = {"Source": 0, "Think-Aloud": 0, "Final Essay": 0}
    st.success("Graph cleared. You can start adding sentences again.")

# -------------------------
# Streamlit GUI
# -------------------------
st.title("Interactive Propositional KG Generator (User Input Only)")

st.markdown("""
### Legend
**Nodes:**
- **Red:** Main Predicate  
- **Blue:** Main Argument  
- **Orange:** Sub-Proposition (Time)  
- **Green:** Sub-Proposition (Place)  
- **Purple:** Sub-Proposition (Attribute)  
- **Gray:** Sub-Proposition (Other)

**Edges:**
- **'argument':** from Main Argument → Main Predicate  
- **'sub_proposition':** from Sub-Proposition → Main Predicate

Nodes are grouped by Source Type (each source's nodes are encompassed by a wide, transparent border).
""")

if st.button("Clear Graph and Start Over"):
    clear_graph()

st.markdown("---")
st.subheader("Add a New Sentence")
col1, col2 = st.columns([2,1])
with col1:
    input_sentence = st.text_area("Enter sentence here:", "")
with col2:
    source_type = st.selectbox("Select Source Type:", ["Source", "Think-Aloud", "Final Essay"])

if st.button("Add Sentence"):
    if input_sentence.strip() != "":
        current_index = st.session_state.sentence_counters[source_type]
        prop = extract_proposition_from_sentence(input_sentence, source_type, current_index)
        st.session_state.props.append(prop)
        st.session_state.sentence_counters[source_type] += 1
        st.success("Sentence added!")
    else:
        st.error("Please enter a sentence.")

st.markdown("---")
st.subheader("Generated Knowledge Graph")

# Build graph from propositions.
G, node_cluster, color_map = build_graph_from_propositions(st.session_state.props)
net = build_pyvis_network(G, color_map)

# -------------------------
# Custom JavaScript: Draw wide transparent borders around nodes of each source group.
# -------------------------
custom_js = """
network.on("afterDrawing", function(ctx) {
    var groups = {};
    // Group nodes by their 'group' (source type)
    for (var nodeId in network.body.nodes) {
        var node = network.body.nodes[nodeId];
        var group = node.options.group || "default";
        if (!groups[group]) groups[group] = [];
        groups[group].push({x: node.x, y: node.y});
    }
    // For each source group, compute bounding box and draw a wide transparent rounded rectangle.
    for (var group in groups) {
        var points = groups[group];
        var xs = points.map(function(p) { return p.x; });
        var ys = points.map(function(p) { return p.y; });
        var margin = 0.1 * network.canvas.getWidth();  // Wide margin
        var min_x = Math.min.apply(null, xs) - margin;
        var max_x = Math.max.apply(null, xs) + margin;
        var min_y = Math.min.apply(null, ys) - margin;
        var max_y = Math.max.apply(null, ys) + margin;
        var width = max_x - min_x;
        var height = max_y - min_y;
        ctx.beginPath();
        ctx.lineWidth = 4;
        ctx.strokeStyle = "black";
        ctx.globalAlpha = 0.2;
        if (ctx.roundRect) {
            ctx.roundRect(min_x, min_y, width, height, 20);
        } else {
            ctx.rect(min_x, min_y, width, height);
        }
        ctx.stroke();
        ctx.globalAlpha = 1.0;
        ctx.font = "20px Arial";
        ctx.fillStyle = "black";
        ctx.fillText(group, min_x, min_y - 10);
    }
});
"""

html_data = net.generate_html()
html_data = html_data.replace("</script>", custom_js + "\n</script>")

# -------------------------
# Display the Interactive Graph in Streamlit.
# -------------------------
st.markdown("### Interactive Knowledge Graph")
components.html(html_data, height=900, width=1200)
