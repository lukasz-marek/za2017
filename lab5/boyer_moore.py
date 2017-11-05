def compute_last_occurrence(pattern, alphabet):
    _lambda = {}
    for symbol in alphabet:
        _lambda[symbol] = 0
    for j in range(len(pattern)):
        _lambda[pattern[j]] = j
    return _lambda


def compute_good_suffix(pattern):
    prefix = compute_prefix(pattern)
    inverse_prefix = compute_prefix(pattern[::-1])
    good_suffix = {}

    for j in range(len(pattern)):
        good_suffix[j] = len(pattern) - prefix[len(pattern)]

    for l in range(1, len(pattern)):
        j = len(pattern) - inverse_prefix[l]
        if good_suffix[j] > l - inverse_prefix[l]:
            good_suffix[j] = l - inverse_prefix[l]

    return good_suffix


def compute_prefix(pattern):
    prefix = {}
    for k in range(len(pattern)):
        for q in range(len(pattern)):
            if k < q and pattern[:q].endswith(pattern[:k]):
                if q in prefix:
                    prefix[q] = k if k > prefix[q] else prefix[q]
                else:
                    prefix[q] = k
    return prefix


def boyer_moore(text, pattern):
    alphabet = [x for x in text]
    alphabet.extend([x for x in pattern])
    alphabet = set(alphabet)
    last = compute_last_occurrence(pattern, alphabet)
    suffix = compute_good_suffix(pattern)

    s = 0
    while s <= len(text) - len(pattern):
        j = len(pattern)
        while j > 0 and pattern[j] == text[s + j]:
            j = j - 1
        if j == 0:
            print(s)
        else:
            s += max(suffix[j], j - last[text[s + j]])


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
