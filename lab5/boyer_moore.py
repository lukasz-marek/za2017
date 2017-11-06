def compute_last_occurrence(pattern, alphabet):
    _lambda = {}
    for symbol in alphabet:
        _lambda[symbol] = 0
    for j in range(len(pattern)):
        _lambda[pattern[j]] = j
    return _lambda


def boyer_moore(text, pattern):
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
            i = i -1 #+ len(pattern) - min(j, 1 + last[text[i]])
            j = len(pattern) - 1
    return -1


def get_text(filename):
    output = ""
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            output += line
    return output

if __name__ == "__main__":
    pattern = "omne"
    text = get_text("text.txt")
    occurrence = boyer_moore(text, pattern)
    while occurrence > -1:
        print(occurrence)
        occurrence = boyer_moore(text[:occurrence], pattern)
