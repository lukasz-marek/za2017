import math

def load_data(filepath):
    graph = {}
    with open(filepath, "r", encoding="utf-8") as input_file:
        for line in input_file:
            parts = line.strip().split(";")
            from_node = int(parts[0])
            to_node = int(parts[1])
            weight = int(parts[2])
            graph[(from_node, to_node)] = weight
            graph[(to_node, from_node)] = weight
    return graph


def floyd_warshall(graph):
    edges = set(graph.keys())

    distances = {}
    precedessors = {}
    for x, y in edges:
        if x not in distances:
            distances[x] = {}
        distances[x][y] = graph[(x, y)]
        distances[x][x] = 0
        if x not in precedessors:
            precedessors[x] = {}
        precedessors[x][y] = x

    nodes = list(map(lambda edge: edge[0], edges))

    for middle in nodes:
        for start in nodes:
            if middle not in distances[start]:
                distances[start][middle] = math.inf
            for end in nodes:
                if end not in distances[start]:
                    distances[start][end] = math.inf

                if end not in distances[middle]:
                    distances[middle][end] = math.inf

                if distances[start][end] > distances[start][middle] + distances[middle][end]:
                    distances[start][end] = distances[start][middle] + distances[middle][end]
                    precedessors[start][end] = precedessors[middle][end]

    return distances, precedessors


if __name__ == "__main__":
    file_name = "data.txt"
    data = load_data(file_name)
    d, p = floyd_warshall(data)
    print("Distance(1,20) =", d[1][20])
    x = 1
    print("Path:")
    while x != p[x][20]:
        next_x = p[x][20]
        print(x, " -> ", next_x)
        x = next_x
