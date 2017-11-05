import numpy as np
from math import inf


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


def find_paths(graph, source, destination, c, f):
    to_visit = []
    for next_node in graph[source].keys():
        to_visit.append((next_node, [source]))

    while len(to_visit) > 0:
        current, path = to_visit.pop(0)

        new_path = list(path)
        new_path.append(current)

        if current == destination and is_p(new_path, c, f):
            yield new_path
        elif current in graph:
            for next_node in graph[current].keys():
                cf = c[current][next_node] - f[current][next_node]
                if cf > 0 and next_node not in new_path and is_p(new_path, c, f):
                    to_visit.append((next_node, new_path))
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
    c = transform_graph_into_matrix(graph)
    f = np.zeros_like(c)

    vertexes = get_vertexes(graph)
    for v1 in vertexes:
        for v2 in vertexes:
            f[v1][v2] = 0

    paths_generator = find_paths(graph, source, destination, c, f)

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
    print("Flow 43->180 =", flows[43][180])

    nodes = get_vertexes(graph)
    test_source = 43
    max_node = -1
    max_value = -inf
    for node in nodes:
        if node != test_source:
            flows = ford_fulkerson(graph, test_source, node)
            flow = np.sum(np.array([x if x > 0 else 0 for x in flows[node]]))
            if flow > max_value:
                max_node = node

    print("max flow = ", max_value, " achieved for node ", max_node)


