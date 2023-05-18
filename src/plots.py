# plots.py
#     neuroscope plots
# by: Noah Syrkis

# imports
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
import igraph as ig
import jraph
from nilearn import plotting, datasets

# functions
def connectome_to_nx_graph(connectome):
    # given a jraph graph, return a nx graph
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(connectome.n_node))
    edges = [(int(connectome.senders[i]), int(connectome.receivers[i])) for i in range(connectome.n_edge)]
    nx_graph.add_edges_from(edges)
    return nx_graph

def plot_graph(graph: jraph.GraphsTuple) -> None:
    # visualize the connectome
    G = ig.Graph.from_networkx(connectome_to_nx_graph(graph))
    pos = G.layout("kk", dim=3)
    Xn, Yn, Zn = zip(*pos)
    Xe, Ye, Ze = [], [], []
    for e in G.get_edgelist():
        Xe += [pos[e[0]][0], pos[e[1]][0], None]
        Ye += [pos[e[0]][1], pos[e[1]][1], None]
        Ze += [pos[e[0]][2], pos[e[1]][2], None]
    
    edge_trace = go.Scatter3d(
        x=Xe,
        y=Ye,
        z=Ze,
        mode="lines",
        line=dict(color="rgb(125,125,125)", width=1),
        hoverinfo="none",
    )

    node_trace = go.Scatter3d(
        x=Xn,
        y=Yn,
        z=Zn,
        mode="markers",
        name="",
        marker=dict(
            symbol="circle",
            size=6,
            color="rgb(200,200,200)",
            line=dict(color="rgb(50,50,50)", width=0.5),
        ),
        text=list(range(len(Xn))),
        hoverinfo="text",
    )
    axis = dict(
        showbackground=False,
        showline=False,
        zeroline=False,
        showgrid=False,
        showticklabels=False,
        title="",
    )
    layout = go.Layout(
        title="",
        width=1500,
        height=1000,
        # black background
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        scene=dict(xaxis=dict(axis), yaxis=dict(axis), zaxis=dict(axis)),
        margin=dict(t=100),
        hovermode="closest",
        annotations=[
            dict(
                showarrow=False,
                text="",
                xref="paper",
                yref="paper",
                x=0,
                y=0.1,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=14),
            )
        ],
    )
    data = [edge_trace, node_trace]
    fig = go.Figure(data=data, layout=layout)
    fig.show()


def plot_regions(rois, hem, img=None):
    # if variable called view exists, close it
    if img is None:
        surface = np.zeros(fsaverage_roi_response_to_image(rois[0], img, hem).shape[0])
        for roi in rois:
            surface += fsaverage_roi(roi, hem)
    else:
        surface = np.zeros(fsaverage_roi_response_to_image(rois[0], img, hem).shape[0])
        for roi in rois:
            surface += fsaverage_roi_response_to_image(roi, img, hem)
    view = plotting.view_surf(
        surf_mesh=atlas["pial_" + hem],
        surf_map=surface,
        darkness=1.0,
        bg_map=atlas["sulc_" + hem],
        threshold=1e-14,
        cmap="twilight_shifted",
        colorbar=True,
        title=hem + " hemisphere " + ", ".join(rois),
        black_bg=True,
    )
    return view.resize(height=900, width=1400)


# plot_regions(rois, "left", img=2)

def plot_region(roi, hem, img=None):
    # if variable called view exists, close it
    if img is None:
        surface = fsaverage_roi(roi, hem)
    else:
        surface = fsaverage_roi_response_to_image(roi, img, hem)
    view = plotting.view_surf(
        surf_mesh=atlas["pial_" + hem],
        surf_map=surface,
        bg_map=atlas["sulc_" + hem],
        threshold=1e-14,
        cmap="twilight_r",
        colorbar=True,
        title=roi + ", " + hem + " hemisphere",
    )
    view.open_in_browser()
