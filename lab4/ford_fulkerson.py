def load_data(file_name):
    graph = {}
    with open(file_name, "r", encoding="utf-8") as input_file:
        for line in input_file:
            parts = line.split(sep="\t")
            source = parts[0]
            destination = parts[1]
            flow = parts[2]
            if source not in graph:
                graph[source] = {}
            if destination not in graph:
                graph[destination] = {}
            graph[source][destination] = flow
            if source not in graph[destination]:
                graph[destination][source] = 0
    return graph

if __name__ == "__main__":
    data = load_data("graph1.txt")
    print(data)