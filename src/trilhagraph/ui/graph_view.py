"""Transformacao do subgrafo em visualizacao Plotly."""

from __future__ import annotations

import math
from dataclasses import dataclass

from trilhagraph.retrieval.models import GraphEdge, GraphNode

LABEL_COLORS = {
    "Area": "#0F766E",
    "Curso": "#EA580C",
    "Competencia": "#2563EB",
    "Perfil": "#7C3AED",
    "Nivel": "#CA8A04",
    "Trilha": "#DC2626",
}


@dataclass(frozen=True, slots=True)
class GraphViewData:
    positions: dict[str, tuple[float, float]]
    nodes: list[GraphNode]
    edges: list[GraphEdge]


def build_graph_view_data(
    nodes: list[GraphNode], edges: list[GraphEdge]
) -> GraphViewData:
    ordered_nodes = sorted(nodes, key=lambda node: (node.label, node.name, node.id))
    count = len(ordered_nodes)
    positions = (
        {
            node.id: (
                math.cos(2 * math.pi * index / count),
                math.sin(2 * math.pi * index / count),
            )
            for index, node in enumerate(ordered_nodes)
        }
        if count
        else {}
    )
    visible_edges = [
        edge for edge in edges if edge.source in positions and edge.target in positions
    ]
    return GraphViewData(positions, ordered_nodes, visible_edges)


def create_graph_figure(nodes: list[GraphNode], edges: list[GraphEdge]):
    import plotly.graph_objects as go

    data = build_graph_view_data(nodes, edges)
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_text: list[str | None] = []
    for edge in data.edges:
        source_x, source_y = data.positions[edge.source]
        target_x, target_y = data.positions[edge.target]
        edge_x.extend([source_x, target_x, None])
        edge_y.extend([source_y, target_y, None])
        edge_text.extend([edge.relationship_type, edge.relationship_type, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        text=edge_text,
        mode="lines",
        hoverinfo="text",
        line={"width": 1.5, "color": "#94A3B8"},
        name="Relacionamentos",
    )
    node_trace = go.Scatter(
        x=[data.positions[node.id][0] for node in data.nodes],
        y=[data.positions[node.id][1] for node in data.nodes],
        text=[node.name for node in data.nodes],
        hovertext=[f"{node.name}<br>{node.label}<br>{node.id}" for node in data.nodes],
        hoverinfo="text",
        mode="markers+text",
        textposition="top center",
        marker={
            "size": 22,
            "color": [LABEL_COLORS.get(node.label, "#475569") for node in data.nodes],
            "line": {"width": 1.5, "color": "#F8FAFC"},
        },
        name="Entidades",
    )
    figure = go.Figure(data=[edge_trace, node_trace])
    figure.update_layout(
        showlegend=False,
        hovermode="closest",
        height=520,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        xaxis={"visible": False},
        yaxis={"visible": False},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        dragmode="pan",
    )
    return figure
