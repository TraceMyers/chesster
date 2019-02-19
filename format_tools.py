def alignOnPattern(pattern, strings, rev=False):
    """
    - Aligns a list of strings by lining up the first or last pattern in each string with the pattern
    farthest from index 0.
    - Does so by adding whitespace before the non-whitespace characters preceding the patterns.
    - Doesn't work if there isn't any white space in the string preceding the pattern or missing a pattern in any string.
    """
    if not rev:
        dot_indices = [string.find(pattern) for string in strings]
    else:
        dot_indices = [string.rfind(pattern) for string in strings]
    max_index = max(dot_indices)
    whitespace_indices = [strings[i].rfind(" ", 0, dot_indices[i]) for i in range(len(strings))]
    add_whitespace = [max_index - dot_index for dot_index in dot_indices]
    formatted_strings = [
        strings[i][:whitespace_indices[i]] + 
        (" " * add_whitespace[i]) + 
        strings[i][whitespace_indices[i]:] 
        for i in range(len(strings))
    ]
    return formatted_strings

