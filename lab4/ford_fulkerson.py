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


def get_vertexes(graph):
    vertexes = set()
    for v1 in graph.keys():
        vertexes.add(v1)
        for v2 in graph[v1].keys():
            vertexes.add(v2)
    return vertexes


def find_paths(graph, source, destination):
    paths = []
    to_visit = []
    for v in graph[source]:
        to_visit.append((source, v, [source]))

    while len(to_visit) > 0:
        source, current, path = to_visit.pop(0)
        paths.append((source, current))
        for v in graph[current]:
            new_path = list(path)
            new_path.append(current)
            to_visit.append((source, v, new_path))
            to_visit.append((current, v, [current]))


    return [(2, 45)]

def compute_residual_flow(residual, flows, u, v):
    return residual[u][v] - flows[u][v]

def ford_fulkerson(graph, source, destination):
    residual = {}
    flow = {}
    vertexes = get_vertexes(graph)
    for v1 in vertexes:
        flow[v1] = {}
        residual[v1] = {}
        for v2 in vertexes:
            flow[v1][v2] = 0
            residual[v1][v2] = graph[v1][v2] if v1 in graph and v2 in graph[v1] else 0

    paths = find_paths(graph, source, destination)
    while residual[source][destination] > 0:
        cf = min(map(lambda pair: compute_residual_flow(residual, flow, pair[0], pair[1]), paths))
        for u, v in paths:
            pass




if __name__ == "__main__":
    data = load_data("graph1.txt")
    ford_fulkerson(data, 2, 45)