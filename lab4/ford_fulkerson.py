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

    vertexes = get_vertexes(graph)
    matrix_size = max(vertexes) + 1
    graph_matrix = np.zeros((matrix_size, matrix_size))
    for v1, v2s in graph.items():
        for v2, flow in v2s.items():
            graph_matrix[v1][v2] = flow
    return graph, graph_matrix


def get_vertexes(graph):
    vertexes = set()
    for v1 in graph.keys():
        vertexes.add(v1)
        for v2 in graph[v1].keys():
            vertexes.add(v2)
    return vertexes


def find_paths(graph, source, destination):
    to_visit = []
    for next_node in graph[source].keys():
        to_visit.append((next_node, [source]))

    while len(to_visit) > 0:
        current, path = to_visit.pop(0)

        new_path = list(path)
        new_path.append(current)

        if current == destination:
            yield new_path
        elif current in graph:
            for next_node in graph[current].keys():
                if next_node not in new_path:
                    to_visit.insert(0, (next_node, new_path))
    raise StopIteration()


def is_p(path, c, f):
    for index in range(len(path) - 1):
        u, v = path[index], path[index + 1]
        cf = c[u][v] - f[u][v]
        if cf <= 0:
            return False
    return True


def compute_next_cf(path, c, f):
    cf = inf
    for index in range(len(path) - 1):
        u, v = path[index], path[index + 1]
        cf_candidate = c[u][v] - f[u][v]
        if cf > cf_candidate:
            cf = cf_candidate
    return cf


def ford_fulkerson(graph, c, source, destination):
    f = np.zeros_like(c)

    vertexes = get_vertexes(graph)
    for v1 in vertexes:
        for v2 in vertexes:
            f[v1][v2] = 0

    paths_generator = find_paths(graph, source, destination)

    p = next(filter(lambda x: is_p(x, c, f), paths_generator), None)
    while p is not None:
        cf = compute_next_cf(p, c, f)
        for index in range(len(p) - 1):
            u, v = p[index], p[index + 1]
            f[u][v] = f[u][v] + cf
            f[v][u] = f[v][u] - cf
        p = next(filter(lambda x: is_p(x, c, f), paths_generator), None)

    return f


if __name__ == "__main__":
    graph, matrix = load_data("graph1.txt")
    result = ford_fulkerson(graph, matrix, 43, 180)