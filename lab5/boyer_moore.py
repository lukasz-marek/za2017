def compute_last_occurrence(pattern, alphabet):
    _lambda = {}
    for symbol in alphabet:
        _lambda[symbol] = 0
    for j in range(len(pattern)):
        _lambda[pattern[j]] = j
    return _lambda


def boyer_moore(text, pattern):
    def _boyer_moore(text, pattern):
        alphabet = [x for x in text]
        alphabet.extend([x for x in pattern])
        alphabet = set(alphabet)
        last = compute_last_occurrence(pattern, alphabet)

        i = len(text) - 1
        j = len(pattern) - 1
        while len(text) > i > 0:
            if pattern[j] == text[i]:
                if j == 0:
                    return i
                else:
                    j -= 1
                    i -= 1
            else:
                i = i - (len(pattern) - max(j, 1 + last[text[i]]))
                j = len(pattern) - 1
        return -1
    occurrence = _boyer_moore(text, pattern)
    occurrences = []

    def identify_word(inner_index):
        word = ""
        index = inner_index
        while index > 0 and text[index] != " ":
            word = text[index] + word
            index -= 1
        index = inner_index + 1
        while index < len(text) and text[index] != " ":
            word = word + text[index]
            index += 1
        return word

    while occurrence > -1:
        occurrences.append((occurrence, identify_word(occurrence)))
        occurrence = _boyer_moore(text[:occurrence], pattern)
    return occurrences


def get_text(filename):
    output = ""
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            output += line
    return output

if __name__ == "__main__":
    pattern = "omne"
    text = get_text("text.txt")
    results = boyer_moore(text, pattern)
    print(results)

