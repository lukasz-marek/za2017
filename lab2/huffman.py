import binarytree
from queue import PriorityQueue


class HuffmanNode(binarytree.Node):
    def __init__(self, symbol, count):
        super().__init__(count)
        self.count = count
        self.left = None
        self.right = None
        self.symbol = symbol

    def __eq__(self, other):
        return self.symbol == other.symbol

    def __lt__(self, other):
        return self.count < other.count

    def find_code(self, symbol):
        if symbol == self.symbol:
            return []
        elif self.symbol:
            return None
        elif self.left or self.right:
            if self.left:
                search_result = self.left.find_code(symbol)
                if search_result:
                    search_result.insert(0, 0)
                    return search_result
            if self.right:
                search_result = self.right.find_code(symbol)
                if search_result:
                    search_result.insert(0, 1)
                    return search_result

    def __str__(self):
        return "HuffmanNode(" + str(self.symbol) + ", " + str(self.count) + ")"


def compute_symbol_counts(filepath):
    results = {}
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            for character in line:
                if character in results:
                    results[character] += 1
                else:
                    results[character] = 1
    return results


def huffman(symbol_counts):
    nodes = PriorityQueue()

    for x, y in symbol_counts.items():
        nodes.put(HuffmanNode(x, y))

    while nodes.qsize() > 1:
        left = nodes.get()
        right = nodes.get()
        new_node_count = left.count + right.count
        new_node = HuffmanNode(None, new_node_count)
        new_node.left = left
        new_node.right = right
        nodes.put(new_node)

    return nodes.get()

if __name__ == "__main__":
    filename = "test.txt"
    symbols = compute_symbol_counts(filename)
    result = huffman(symbols)
    print(result.find_code("h"))



