import re

def compute_last_occurrence(pattern, alphabet):
    _lambda = {}
    for symbol in alphabet:
        _lambda[symbol] = 0
    for j in range(len(pattern)):
        _lambda[pattern[j]] = j
    return _lambda


def boyer_moore(text, pattern):
    whitespace_regex = "\s"
    text_to_check = text.lower()
    pattern_to_check = pattern.lower()

    def _boyer_moore(text, pattern):
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
                i -= len(pattern) - max(j, last[text[i]])
                j = len(pattern) - 1
        return -1

    def identify_word(inner_index):
        word = ""
        index = inner_index
        while index > 0 and not re.match(whitespace_regex, text[index]):
            word = text[index] + word
            index -= 1
        index = inner_index + 1
        while index < len(text) and not re.match(whitespace_regex, text[index]):
            word = word + text[index]
            index += 1
        return word

    alphabet = [x for x in text_to_check]
    alphabet.extend([x for x in pattern_to_check])
    alphabet = set(alphabet)
    last = compute_last_occurrence(pattern_to_check, alphabet)
    occurrence = _boyer_moore(text_to_check, pattern_to_check)
    occurrences = []

    while occurrence > -1:
        occurrences.append((occurrence, identify_word(occurrence)))
        occurrence = _boyer_moore(text_to_check[:occurrence], pattern_to_check)
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
    count = 0
    for s, word in boyer_moore(text, pattern):
        print(s, " ", word)
        count += 1
    print("---------------------")
    print(count, " matches found")
