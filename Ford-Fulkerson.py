# implementation of the ford fulkerson algorithm by SEK171
# inspired by https://engri-1101.github.io/textbook/chapters/max_flow/algorithms_python_web.html

# will need the following packages: numpy, networkx, matplotlib


import numpy as np
# for this one i am using networkx as pure graphviz is a hassle to do (i will regret that later)
import networkx as nx
from matplotlib import pyplot as plt


# a function to extract arcs from a graph dictionary
def EdgeList(X, U):
    edges = []
    for i in range(len(X)):
        for j in range(len(X)):
            if U[i, j] != 0:
                edges.append((X[i], X[j], U[i, j]))

    return edges


def positions(X, grouping):
    pos = [[0, 0] for _ in range(len(X))]
    current = 1
    for i in X:
        for level in grouping:
            if i in grouping[level]:
                pos[i][0] = 2 * level
                n = 10
                if len(grouping[level]) == 1:
                    pos[i][1] = n / 2
                else:
                    pos[i][1] = n / 2 + (current - (len(grouping[level]) + 1) / 2) * (
                                (n - 1) / (len(grouping[level]) - 1))

                if current == len(grouping[level]):
                    current = 0
                current += 1
    pos = [(i, j) for i, j in pos]
    return pos


def plot_graph():
    edge_labels = {}
    for i, j in G.edges:
        edge_labels[(i, j)] = G.edges[i, j]['cap']

    plt.figure(figsize=(12, 6))

    nx.draw_networkx(G, pos, labels=X, node_size=400, node_color='darkorange')  # draw graph
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, verticalalignment="bottom",
                                 bbox=dict(boxstyle='square', ec=(0.0, 0.0, 0.0), fill=False))

    plt.show()


def plot_residual_graph():
    global residual_graph

    residual_graph = nx.DiGraph()

    for i, j in G.edges:
        capacity = G.edges[i, j]['cap']
        flux = G.edges[i, j]['flux']

        if capacity > flux:
            residual_graph.add_edge(i, j)
            residual_graph.edges[i, j]['residual_cap'] = capacity - flux
            residual_graph.edges[i, j]['forward'] = True

        if flux > 0:
            residual_graph.add_edge(j, i)
            residual_graph.edges[j, i]['residual_cap'] = flux
            residual_graph.edges[j, i]['forward'] = False

    plt.figure(figsize=(12, 6))

    F_edges = [(i, j) for i, j in residual_graph.edges if residual_graph.edges[i, j]['forward'] == True]
    F = residual_graph.edge_subgraph(F_edges)

    nx.draw_networkx(F, pos, labels=X, node_size=400, node_color='limegreen')  # draw forward graph

    residual_cap = nx.get_edge_attributes(F, 'residual_cap')
    nx.draw_networkx_edge_labels(F, pos, edge_labels=residual_cap, verticalalignment="bottom",
                                 bbox=dict(boxstyle='square', ec=(0.0, 0.0, 0.0), fill=False),
                                 label_pos=0.6)
    # -------------------------------------------- now for flux arrows
    B_edges = [(i, j) for i, j in residual_graph.edges if residual_graph.edges[i, j]['forward'] == False]
    B = residual_graph.edge_subgraph(B_edges)

    nx.draw_networkx_edges(B, pos, node_size=400, edge_color='blue',
                           connectionstyle='arc3, rad=0.3')  # draw flux graph on top

    residual_cap = nx.get_edge_attributes(B, 'residual_cap')
    nx.draw_networkx_edge_labels(B, pos, edge_labels=residual_cap, verticalalignment="bottom",
                                 bbox=dict(boxstyle='square', alpha=0),
                                 font_color='blue', label_pos=0.8)
    plt.show()


def update():
    for i, j in path:
        if residual_graph.edges[i, j]['forward']:
            G.edges[i, j]['flux'] += delta
        else:
            G.edges[j, i]['flux'] -= delta


def checking():
    for i in G.nodes:
        G.nodes[i]['check'] = False
    G.nodes[0]['check'] = True

    checking_list = [0]

    while (len(checking_list) > 0):

        i = checking_list.pop()

        for j in residual_graph.neighbors(i):

            if not G.nodes[j]['check']:
                G.nodes[j]['check'] = True
                G.nodes[j]['prev'] = i
                checking_list.append(j)


def path_find():
    global path, delta, u0

    j = list(X)[-1]  # last element
    path = []

    while j != 0:
        i = G.nodes[j]['prev']
        path.insert(0, (i, j))

        j = i

    path_letters = [(X[i], X[j]) for i, j in path]
    print("le chemin de S a T est:")
    print(path_letters)

    path_caps = [residual_graph.edges[i, j]['residual_cap'] for i, j in path]
    delta = min(path_caps)

    print("la valeur minimum de capacite sur le chemin est:")
    print(delta)

    u0 += delta
    print("la valeur de u0 est:")
    print(u0)


def ford_fulkerson():
    global G

    for i, j in G.edges:
        G.edges[i, j]['flux'] = 0

    plot_residual_graph()
    checking()

    while G.nodes[list(X)[-1]]['check']:
        path_find()
        update()
        plot_residual_graph()
        checking()


# graph definition
X = ["S", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "T"]
X = {i: x for i, x in enumerate(X)}
U = [
    [0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0],
    [5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 10, 7, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 5, 0, 4, 15, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 5, 0, 15, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 30, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

U = np.array(U)
# concatenate the -----S into the matrix (row and column)
srow = np.array([[0 for _ in range(U.shape[1])]])
srow[0, 0] = 15;
srow[0, 1] = 10;
srow[0, 2] = 15;
srow[0, 3] = 15
U = np.concatenate((srow, U), axis=0)  # row
U = np.concatenate((np.array([[0 for _ in range(U.shape[0])]]).T, U), axis=1)  # column
# concatenate the -----T into the matrix (row and column)
U = np.concatenate((U, np.array([[0 for _ in range(U.shape[1])]])), axis=0)  # row
tcol = np.array([[0 for _ in range(U.shape[0])]])
tcol[0, -2] = 15;
tcol[0, -3] = 20;
tcol[0, -4] = 15;
U = np.concatenate((U, tcol.T), axis=1)  # column

# definition initial du graph graphique
G = nx.DiGraph()
# definition des arcs
edges = EdgeList(list(X.keys()), U)
G.add_weighted_edges_from(edges, 'cap')

# les niveau des points / change accordingly to make it look pretty
grouping = {
    0: [0],
    1: [1, 2, 3, 4],
    2: [5, 6, 7],
    3: [8, 9],
    4: [10, 11, 12],
    5: [13]
}

# les positions des points
pos = positions(X, grouping)

# construction du graph residual
residual_graph = nx.DiGraph()

# le chemin courant
path = []
# la valeur minimal de cette chemin
delta = 0
# la valeur de u0
u0 = 0

# main
ford_fulkerson()
