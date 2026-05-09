import pandas as pd
import networkx as nx
from itertools import combinations


def build_entity_graph(df):
    """
    Builds a weighted undirected graph connecting entities that co-appear in the same transaction.

    Nodes represent unique values from card1, card2, P_emaildomain, R_emaildomain,
    addr1, addr2, and DeviceInfo. An edge is added between every pair of non-null entity
    values that appear in the same row. Edge weights accumulate TransactionAmt across all
    transactions that share a given entity pair.

    Parameters:
    - df: DataFrame containing the entity columns and TransactionAmt

    Returns:
    - G: networkx Graph with weighted edges (weight = cumulative TransactionAmt)
    """
    entity_cols = [
        'card1', 'card2',
        'P_emaildomain', 'R_emaildomain',
        'addr1', 'addr2',
        'DeviceInfo',
    ]
    G = nx.Graph()

    for col1, col2 in combinations(entity_cols, 2):
        subset = df[[col1, col2, 'TransactionAmt']].dropna()
        grouped = subset.groupby([col1, col2], sort=False)['TransactionAmt'].sum()
        for (a, b), w in grouped.items():
            if G.has_edge(a, b):
                G[a][b]['weight'] += w
            else:
                G.add_edge(a, b, weight=w)

    return G


def get_graph_features(G, df, entity_cols):
    """
    Computes four graph metrics per node and joins them back to df as new columns per entity.

    Metrics:
    - pagerank: unweighted structural importance
    - pagerank_weighted: importance scaled by cumulative TransactionAmt edge weights
    - degree: unweighted degree centrality
    - component_size: number of nodes in the containing connected component

    Column naming: {entity}_pagerank, {entity}_pagerank_weighted,
                   {entity}_degree, {entity}_component_size

    For the default 7-entity configuration this produces 28 graph feature columns
    (7 entities x 4 metrics).

    Parameters:
    - G: networkx Graph built by build_entity_graph
    - df: DataFrame containing the entity columns
    - entity_cols: list of entity column names whose values are graph nodes

    Returns:
    - df with new graph feature columns added
    """
    pagerank = nx.pagerank(G)
    pagerank_weighted = nx.pagerank(G, weight='weight')
    degree = nx.degree_centrality(G)

    component_size = {}
    for comp in nx.connected_components(G):
        size = len(comp)
        for node in comp:
            component_size[node] = size

    df = df.copy()
    for col in entity_cols:
        df[f'{col}_pagerank'] = df[col].map(pagerank)
        df[f'{col}_pagerank_weighted'] = df[col].map(pagerank_weighted)
        df[f'{col}_degree'] = df[col].map(degree)
        df[f'{col}_component_size'] = df[col].map(component_size)

    return df


def get_fraud_dense_subgraph(G, df, entity_cols, top_n=1):
    """
    finds the most fraud-dense connected components in the graph.
    scores each component by the mean fraud rate of its member nodes
    and returns the top_n components as a single subgraph.

    parameters:
    - G: networkx graph from build_entity_graph()
    - df: dataframe containing entity cols and isFraud
    - entity_cols: list of entity columns used to build the graph
    - top_n: number of top fraud-dense components to return

    returns:
    - subgraph containing the most fraud-dense nodes
    """
    import numpy as np

    # compute fraud rate per node across all entity cols
    node_fraud = {}
    for col in entity_cols:
        rates = df.groupby(col)['isFraud'].mean().to_dict()
        for node, rate in rates.items():
            if node not in node_fraud:
                node_fraud[node] = []
            node_fraud[node].append(rate)

    # average fraud rate per node across entity types
    node_fraud_mean = {
        node: np.mean(rates)
        for node, rates in node_fraud.items()
        if node in G.nodes
    }

    # score each component by mean fraud rate of its nodes
    components = list(nx.connected_components(G))
    component_scores = []
    for comp in components:
        scores = [node_fraud_mean.get(n, 0) for n in comp]
        component_scores.append((comp, np.mean(scores)))

    # sort by fraud score descending
    component_scores.sort(key=lambda x: x[1], reverse=True)

    # take top_n components
    top_nodes = set()
    for comp, score in component_scores[:top_n]:
        top_nodes.update(comp)

    return G.subgraph(top_nodes)


def plot_fraud_subgraph(subgraph, node_fraud_rates, output_path):
    """
    plots a fraud-dense subgraph with nodes colored by fraud rate
    and sized by degree. edge thickness reflects edge weight.

    parameters:
    - subgraph: networkx subgraph from get_fraud_dense_subgraph()
    - node_fraud_rates: dict of node -> fraud rate for coloring
    - output_path: path to save the plot

    returns:
    - None, saves plot to output_path
    """
    from pathlib import Path
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # layout
    pos = nx.spring_layout(subgraph, seed=42, k=0.5)

    # node colors by fraud rate
    fraud_rates = [node_fraud_rates.get(n, 0) for n in subgraph.nodes]
    norm = mcolors.Normalize(vmin=0, vmax=max(fraud_rates) if fraud_rates else 1)
    cmap = cm.RdYlBu_r
    node_colors = [cmap(norm(r)) for r in fraud_rates]

    # node sizes by degree
    degrees = dict(subgraph.degree())
    node_sizes = [300 + degrees[n] * 100 for n in subgraph.nodes]

    # edge weights for thickness
    edge_weights = [
        subgraph[u][v].get('weight', 1) for u, v in subgraph.edges
    ]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [0.5 + 3 * (w / max_weight) for w in edge_weights]

    fig, ax = plt.subplots(figsize=(14, 10))

    nx.draw_networkx_edges(
        subgraph, pos,
        width=edge_widths,
        alpha=0.4,
        edge_color='gray',
        ax=ax
    )

    nx.draw_networkx_nodes(
        subgraph, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        ax=ax
    )
    # only label high degree nodes
    labels = {
        n: str(n) for n in subgraph.nodes
        if subgraph.degree(n) > 5
    }

    nx.draw_networkx_labels(
        subgraph, pos,
        labels=labels,
        font_size=6,
        font_color='black',
        ax=ax
    )

    # colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='fraud rate')

    ax.set_title('fraud-dense subgraph — node color = fraud rate, size = degree, edge thickness = transaction amount')
    ax.axis('off')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
