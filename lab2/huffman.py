from queue import PriorityQueue
from bitstring import Bits
import ast
import sys
import os


class HuffmanNode:
    def __init__(self, symbol, count):
        self.count = count
        self.left = None
        self.right = None
        self.symbol = symbol
        self.code_part = None
        self.parent = None

    def __eq__(self, other):
        return self.symbol == other.symbol

    def __lt__(self, other):
        return self.count < other.count

    def _get_code(self):
        current = self
        path = [current.code_part]
        while current.parent.code_part:
            path.insert(0, current.parent.code_part)
            current = current.parent
        return "".join(path)

    def find_code(self, symbol):
        stack = [self]

        while len(stack) > 0:
            current = stack.pop(0)
            if current.symbol is not None:
                if current.symbol == symbol:
                    return current._get_code()
            else:
                if current.left:
                    stack.insert(0, current.left)
                if current.right:
                    stack.insert(0, current.right)

    def to_dict(self):
        to_visit = [self]
        results = {}
        while len(to_visit) > 0:
            current = to_visit.pop(0)
            if current.symbol:
                results[current.symbol] = current._get_code()
            if current.left:
                to_visit.insert(0, current.left)
            if current.right:
                to_visit.insert(0, current.right)

        return results

    def find_by_path(self, bin_path):
        current = self
        path = bin_path
        used = 0

        while current.symbol is None and len(path) > 0:
            to_go = path[0]
            path = path[1:]
            if to_go == '0':
                current = current.left
            else:
                current = current.right
            used += 1

        return current.symbol, used

    @staticmethod
    def from_dict(dict_of_mappings):
        root = HuffmanNode(None, None)
        for symbol, code in dict_of_mappings.items():
            current = root
            code_path = [part for part in code]
            while len(code_path) > 0:
                destination = code_path.pop(0)
                if destination == '0':
                    current.left = HuffmanNode(None, None) if current.left is None else current.left
                    current.left.code_part = '0'
                    current.left.parent = current
                    current = current.left
                else:
                    current.right = HuffmanNode(None, None) if current.right is None else current.right
                    current.right.code_part = '1'
                    current.right.parent = current
                    current = current.right

            current.symbol = symbol

        return root


def compute_symbol_counts(filepath, symbol_length=1):
    results = {}
    buffer = ""
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            buffer += line
            while len(buffer) >= symbol_length:
                symbol = buffer[:symbol_length]
                buffer = buffer[symbol_length:]
                if symbol in results:
                    results[symbol] += 1
                else:
                    results[symbol] = 1

    while len(buffer) > 0:
        symbol = buffer[:symbol_length]
        buffer = buffer[symbol_length:]
        if buffer in results:
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
        left.code_part = '0'
        right = nodes.get()
        right.code_part = '1'
        new_node_count = left.count + right.count
        new_node = HuffmanNode(None, new_node_count)
        new_node.left = left
        new_node.right = right
        left.parent = new_node
        right.parent = new_node
        nodes.put(new_node)

    return nodes.get()


def encode(filename_in, filename_out, encoding_dict, symbol_length=1):
    with open(filename_out, "w+", encoding="utf-8") as output_file:
        output_file.write(str(compress_dict(encoding_dict)) + "\n")
    output_buffer = ""
    with open(filename_in, "r", encoding="utf-8") as file, open(filename_out, "ab") as output_file:
        input_buffer = ""
        for line in file:
            input_buffer += line
            while len(input_buffer) >= symbol_length:
                symbol = input_buffer[:symbol_length]
                input_buffer = input_buffer[symbol_length:]
                output_buffer += encoding_dict[symbol]
            while len(output_buffer) >= 8:
                to_store = output_buffer[:8]
                output_buffer = output_buffer[8:]
                b = Bits(bin=to_store)
                output_file.write(b.tobytes())

        if len(input_buffer) > 0:
            symbol = encoding_dict[input_buffer]
            output_buffer += symbol

        if len(output_buffer) > 0:
            output_buffer = output_buffer + "0" * (8 - len(output_buffer) % 8)
            b = Bits(bin=output_buffer)
            output_file.write(b.tobytes())


def decode(filename_in, filename_out):
    with open(filename_in, "rb") as file_in, open(filename_out, "w+", encoding="utf-8") as file_out:

        codes_dict = ast.literal_eval(str(file_in.readline(), "utf-8", errors="strict").strip())
        codes_dict = decompress_dict(codes_dict)
        huffman_tree = HuffmanNode.from_dict(codes_dict)

        b = file_in.read(1)
        buffer = ""
        while b:
            for byte in b:
                key = bin(byte)

                key = key.replace("0b", "")
                key = "0" * (8 - len(key)) + key

                buffer += key

                symbol, used_bits = huffman_tree.find_by_path(buffer)
                while symbol is not None:
                    buffer = buffer[used_bits:]
                    file_out.write(symbol)
                    symbol, used_bits = huffman_tree.find_by_path(buffer)

            b = file_in.read(1)


def find_prefix(buffer, prefix_dict):
    prefixes = set(filter(lambda prefix: buffer.startswith(prefix), prefix_dict.keys()))
    return max(prefixes, key=lambda prefix: len(prefix)) if len(prefixes) > 0 else None


def compress_dict(prefix_dict):
    def compress_prefix(prefix):
        leading_zeros = 0
        while len(prefix) > leading_zeros and prefix[leading_zeros] == '0':
            leading_zeros += 1
        if leading_zeros > 0:
            return hex(leading_zeros).replace("0x", "") + ":" + hex(int(prefix, 2)).replace("0x", "")
        else:
            return hex(int(prefix, 2)).replace("0x", "")

    return {k: compress_prefix(v) for k, v in prefix_dict.items()}


def decompress_dict(prefix_dict):
    def decompress_prefix(prefix):
        if ":" in prefix:
            leading_zeros, decimal_value = prefix.split(":")
            decimal_value = int(decimal_value, 16)
            leading_zeros = int(leading_zeros, 16)
            binary_value = bin(int(decimal_value)).replace("0b", "") if decimal_value > 0 else ''
            decompressed_value = leading_zeros * '0' + binary_value
            return decompressed_value
        else:
            return bin(int(prefix, 16)).replace("0b", "")

    return {k: decompress_prefix(v) for k, v in prefix_dict.items()}


if __name__ == "__main__":
    print(sys.argv)
    MODE = sys.argv[1]
    if MODE == "e":

        SYMBOL_LENGTH = int(sys.argv[2])
        FILENAME_IN = os.path.normpath(sys.argv[3])
        FILENAME_OUT = os.path.normpath(sys.argv[4])

        symbols = compute_symbol_counts(FILENAME_IN, symbol_length=SYMBOL_LENGTH)
        result = huffman(symbols)
        mapping = result.to_dict()
        encode(FILENAME_IN, FILENAME_OUT, mapping, symbol_length=SYMBOL_LENGTH)
        input_size = os.path.getsize(FILENAME_IN)
        output_size = os.path.getsize(FILENAME_OUT)
        L = (input_size - output_size) / input_size
        print("L = ", L)
    elif MODE == "d":

        FILENAME_IN = os.path.normpath(sys.argv[2])
        FILENAME_OUT = os.path.normpath(sys.argv[3])

        decode(FILENAME_IN, FILENAME_OUT)
    elif MODE == "f":

        FILENAME_OUT = "temp.bin"
        FILENAME_IN = os.path.normpath(sys.argv[2])
        FILE_SIZE = 0

        with open(FILENAME_IN, "r", encoding="utf-8") as test_file:
            FILE_SIZE = len(test_file.read())

        for symbol_length in range(1, FILE_SIZE):
            symbols = compute_symbol_counts(FILENAME_IN, symbol_length=symbol_length)
            result = huffman(symbols)
            mapping = result.to_dict()
            encode(FILENAME_IN, FILENAME_OUT, mapping, symbol_length=symbol_length)
            input_size = os.path.getsize(FILENAME_IN)
            output_size = os.path.getsize(FILENAME_OUT)
            L = (input_size - output_size) / input_size
            print("Symbol length = ", symbol_length, ", ", "L = ", L)