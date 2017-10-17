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



if __name__ == "__main__":
    file_name = "data.txt"
    data = load_data(file_name)
    print(data)
