import binarytree
from queue import PriorityQueue
from bitstring import Bits, BitString


BUFFER_SIZE = 8
SYMBOL_LENGTH = 1


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
                if search_result is not None:
                    search_result.insert(0, '0')
                    return search_result
            if self.right:
                search_result = self.right.find_code(symbol)
                if search_result is not None:
                    search_result.insert(0, '1')
                    return search_result

    def to_dict(self):
        to_visit = [self]
        nodes = []
        while len(to_visit) > 0:
            current = to_visit.pop(0)
            if current.symbol:
                nodes.append(current)
            if current.left:
                to_visit.insert(0, current.left)
            if current.right:
                to_visit.insert(0, current.right)

        results = {}
        for node in nodes:
            results[node.symbol] = "".join(self.find_code(node.symbol))
        return results

    def find_by_path(self, bin_path):
        current = self
        path = [char for char in bin_path]
        used = 0

        while not current.symbol and len(path) > 0:
            to_go = path.pop(0)
            if to_go == '0':
                current = current.left
            else:
                current = current.right
            used += 1

        return current.symbol, used


def compute_symbol_counts(filepath, symbol_length=1):
    results = {}
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            for start_index in range(0, len(line), symbol_length):
                symbol = line[start_index:start_index + symbol_length]
                if symbol in results:
                    results[symbol] += 1
                else:
                    results[symbol] = 1
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


def encode(filename_in, filename_out, encoding_dict, symbol_length=1):
    buffer = ""
    with open(filename_in, "r", encoding="utf-8") as file, open(filename_out, "wb+") as output_file:
        for line in file:
            for start_index in range(0, len(line), symbol_length):
                symbol = line[start_index:start_index + symbol_length]
                buffer += encoding_dict[symbol]
                if len(buffer) >= BUFFER_SIZE:
                    to_store = buffer[:BUFFER_SIZE]
                    buffer = buffer[BUFFER_SIZE:]
                    b = Bits(bin=to_store)
                    output_file.write(b.tobytes())
        if len(buffer) > 0:
            buffer = buffer + "0" * (BUFFER_SIZE - len(buffer))
            b = Bits(bin=buffer)
            output_file.write(b.tobytes())


def decode(filename_in, filename_out, huffman_tree, symbol_length=1):
    with open(filename_in, "rb") as file_in, open(filename_out, "w+", encoding="utf-8") as file_out:
        b = file_in.read(1)
        buffer = ""
        while b:
            for byte in b:
                key = bin(byte)

                key = key.replace("0b", "")
                key = "0" * (BUFFER_SIZE - len(key)) + key

                buffer += key

                symbol, used_bits = huffman_tree.find_by_path(buffer)
                while symbol is not None:
                    buffer = buffer[used_bits:]
                    file_out.write(symbol)
                    symbol, used_bits = huffman_tree.find_by_path(buffer)

            b = file_in.read(1)

if __name__ == "__main__":
    filename_in = "test.txt"
    filename_encoded = "encoded.bin"
    filename_decoded = "decoded.txt"
    symbols = compute_symbol_counts(filename_in, symbol_length=SYMBOL_LENGTH)
    result = huffman(symbols)
    mapping = result.to_dict()
    encode(filename_in, filename_encoded, mapping, symbol_length=SYMBOL_LENGTH)
    decode(filename_encoded, filename_decoded, result, symbol_length=SYMBOL_LENGTH)



