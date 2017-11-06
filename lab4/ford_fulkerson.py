import numpy as np
from math import inf
from copy import deepcopy


def load_data(file_name):
    graph = {}
    with open(file_name, "r", encoding="utf-8") as input_file:
        for line in input_file:
            parts = line.split(sep="\t")
            source = int(parts[0])
            destination = int(parts[1])
            flow = float(parts[2])
            if source not in graph:
                graph[source] = {}
            graph[source][destination] = flow
    return graph


def transform_graph_into_matrix(graph):
    vertexes = get_vertexes(graph)
    matrix_size = max(vertexes) + 1
    graph_matrix = np.zeros((matrix_size, matrix_size))
    for v1, v2s in graph.items():
        for v2, flow in v2s.items():
            graph_matrix[v1][v2] = flow
    return graph_matrix


def get_vertexes(graph):
    vertexes = set()
    for v1 in graph.keys():
        vertexes.add(v1)
        for v2 in graph[v1].keys():
            vertexes.add(v2)
    return vertexes


def find_all_paths(graph, source, destination, c, f):
    to_visit = []
    for next_node in graph[source].keys():
        to_visit.append((next_node, [source]))

    while len(to_visit) > 0:
        current, path = to_visit.pop(0)

        new_path = list(path)
        new_path.append(current)

        if current == destination:
            yield new_path
            to_visit = list(filter(lambda x: is_p(x[1], c, f), to_visit))
        elif current in graph:
            to_delete = []
            for next_node in graph[current].keys():
                cf = c[current][next_node] - f[current][next_node]
                if cf > 0:
                    if next_node not in new_path:
                        to_visit.append((next_node, new_path))
                else:
                    to_delete.append(next_node)
            for redundant in to_delete:
                del graph[current][redundant]
    raise StopIteration()


def is_p(path, c, f):
    for index in range(len(path) - 1):
        u, v = path[index], path[index + 1]
        cf = c[u][v] - f[u][v]
        if cf <= 0:
            return False
    return True


def compute_path_flow(path, c, f):
    cf = inf
    for index in range(len(path) - 1):
        u, v = path[index], path[index + 1]
        cf_candidate = c[u][v] - f[u][v]
        if cf > cf_candidate:
            cf = cf_candidate
    return cf


def ford_fulkerson(graph, source, destination):

    graph = deepcopy(graph)
    c = transform_graph_into_matrix(graph)
    f = np.zeros_like(c)

    vertexes = get_vertexes(graph)
    for v1 in vertexes:
        for v2 in vertexes:
            f[v1][v2] = 0

    paths_generator = find_all_paths(graph, source, destination, c, f)

    path = next(paths_generator, None)
    while path is not None:
        cf = compute_path_flow(path, c, f)
        for index in range(len(path) - 1):
            u, v = path[index], path[index + 1]
            f[u][v] = f[u][v] + cf
            f[v][u] = f[v][u] - cf
        path = next(paths_generator, None)

    return f


if __name__ == "__main__":
    graph = load_data("graph1.txt")
    flows = ford_fulkerson(graph, 43, 180)
    print("Flow 43 -> 180 =", -np.sum(flows[180]))

    nodes = get_vertexes(graph)
    test_source = 43
    max_node = -1
    max_value = -inf
    for node in sorted(nodes, reverse=True):
        if node != test_source:
            flows = ford_fulkerson(graph, test_source, node)
            flow = -np.sum(flows[node])
            print(test_source, " -> ", node, " = ", flow)
            if flow > max_value:
                max_node = node

    print("Max flow: ", test_source, " -> ", max_node, " = ", max_value)


