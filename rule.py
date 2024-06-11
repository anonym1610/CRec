import distance
import re
import unicodedata

from antlr4 import *


def full_to_half(text: str) -> str:
    """
    Convert full-width characters to half-width characters in a given text.

    Args:
        text: The input string that may contain full-width characters.

    Returns:
        str: The converted string where all full-width characters have been replaced with their half-width counterparts.
    """
    text = unicodedata.normalize('NFKC', text)
    table = {ord(f): ord(h) for f, h in zip(u'‘’“”【】、', '\'\'""[]\\')}
    text = text.translate(table)
    return text


def repair_multi_line_comment(lines: list) -> None:
    """
    Fix multi-line comments.

    We assume that the multi-line comments adhere to the following style:
    /*
     * ...
     * ...
     */

    Args:
        lines: A list of code lines.
    """
    in_comment = False  # if start == True, it means the current line is in a multi-line comment
    for i, line in enumerate(lines):
        lstripped_line = line.lstrip()
        # if the current line is not in a multi-line comment and starts with "*" or "/*"
        if in_comment == False and re.match('(\*|/\*)', lstripped_line) is not None:
            # we consider the current line as the beginning of a multi-line comment
            in_comment = True
            # if the current line starts with '*', it signifies an invalid start to a multi-line comment
            if lstripped_line.startswith('*'):
                # if the current line is the first line of the code
                if i == 0:
                    # we add "/*" to the beginning of the current line to initiate a multi-line comment
                    lines[i] = '/*' + lines[i]
                else:
                    # if the previous line does not start with "/*", we add "/*" to the beginning of it
                    if re.match('/\*', lines[i - 1].lstrip()) is None:
                        lines[i - 1] = '/*' + lines[i - 1]
        # if the current line is in a multi-line comment
        if in_comment == True:
            # if the current line is the last line of the ocred code
            # remember, the last line is a '}' that we previously added
            if i == len(lines) - 1:
                # if the current line does not start with "*" or "*/", it signifies an invalid end to a multi-line comment
                if re.match('(\*|\*/)', lstripped_line) is None:
                    # we consider the current line as the end of a multi-line comment
                    in_comment = False
                    # if the previous line does not end with "*/", we add "*/" to the beginning of the current line
                    if not lines[i - 1].rstrip().endswith('*/'):
                        lines[i] = '*/' + lines[i]
            else:
                # if the next line does not start with "*" or "*/", it signifies an invalid end to a multi-line comment
                if re.match('(\*|\*/)', lines[i + 1].lstrip()) is None:
                    # we consider the current line as the end of a multi-line comment
                    in_comment = False
                    # if the current line does not end with "*/", we add "*/" to the end of it
                    if not lstripped_line.rstrip().endswith('*/'):
                        lines[i] = lines[i] + '*/'


def simplified_brace_complement_gstyle(lines: list) -> list:
    """
    Add potentially missing right braces.

    Parameters:
    lines : A list of code lines.

    Returns:
    list: A list of code lines with potentially missing right braces added.
    """
    # remember, the code has been formatted by clang-format in google style
    insert_times = 0  # the number of times a new line is inserted
    scope = []  # a stack to track encountered block scopes
    add_class = False  # if add_class == True, it signifies that a class definition wrapped the entire code is added

    # three types of blocks
    type_block = 0
    method_block = 1
    normal_block = 2

    re_type_declaration = '(class|interface|enum)\s+\w+(\s+extends\s+\w+)?(\s+implements\s+\w+(\s*,\s*\w+)*\s*)?\s*'
    re_method_declaration = '(public|protected|private|static) +[\w\<\>\[\]\,\s]*\s*(\w+) *\([^\)]*\) *(\{?|[^;])'
    re_ignore = '(package|import|open|module|COMMENT|extends|implements|permits|throw|throws|@)'

    text = ''.join(lines)

    def merge_into_one_line(match):
        return match.group().replace('\n', ' ')

    # maintain each declaration on a single line
    text = re.sub(re_type_declaration, merge_into_one_line,
                  text, flags=re.DOTALL)
    text = re.sub(re_method_declaration,
                  merge_into_one_line, text, flags=re.DOTALL)

    def break_short_line(match, text):
        # exclude the case where the content is in a string
        if re.search('".*%s.*"' % re.escape(match.group()), text):
            return match.group()
        else:
            content = match.group()[1:-1].strip()
            return "{\n" + content + "\n}"

    # break the short line in braces
    re_braces = '\{[^}\n]*?\}'
    text = re.sub(re_braces, lambda m: break_short_line(
        m, text), text, flags=re.DOTALL)

    lines = text.split('\n')
    for i, line in enumerate(lines):
        lines[i] = line.strip() + '\n'

    # A copy of the input lines where modifications are applied.
    lines_copy = lines.copy()
    last_rbrace_line_index = 0  # the index of the line containing the last right brace

    def contains_match_outside_strings(regex, text):
        strings = re.findall('"[^"]*"', text)
        for s in strings:
            text = text.replace(s, '')
        return re.search(regex, text) is not None

    for i, line in enumerate(lines):
        if line.strip() != '' and re.match(re_ignore, line.lstrip()) is None:
            # if the line contains a 'type declaration'
            if contains_match_outside_strings(re_type_declaration, line):
                if scope:
                    # because 'scope' is not empty, the current line is definitely not the first line
                    suspect_line = re.sub(
                        r'COMMENT\d+', "", lines[i - 1]).strip()
                    # if the previous line has only 1 or 2 characters except 'COMMENT's
                    # it may be that OCR misrecognized '}'
                    # replace the previous line with '}'
                    if suspect_line not in ['{', '}', '};', ';'] and 0 < len(suspect_line) < 3:
                        lines_copy[i + insert_times - 1] = '}\n'
                        last_rbrace_line_index = i + insert_times - 1
                        # when a '}' is inserted, it signifies the closure of the block at the top of the stack.
                        # so, the block is popped from the stack
                        scope.pop()
                else:
                    # the code already contains a 'type declaration', so there is no need to add a new one
                    add_class = True
                scope.append(type_block)
            else:
                if not scope and re.search('\w', line) is not None:
                    if not add_class:
                        # since 'compilationUnit' is the start symbol of the grammar
                        # we add a necessary class definition to support parsing
                        class_header = 'class ParsingSupport {\n'
                        lines_copy.insert(i + insert_times, class_header)
                        insert_times = insert_times + 1
                        scope.append(type_block)
                        add_class = True
                    else:
                        # we have added a class definition wrapped the entire code to support parsing
                        # if 'scope' is empty, and the current line contains a word, it signifies the end of the previous block is incorrect
                        # so, we delete the last '}' and add the type declaration back to the top of the stack
                        idx = lines_copy[last_rbrace_line_index].rfind('}')
                        if idx != -1:
                            lines_copy[last_rbrace_line_index] = lines_copy[last_rbrace_line_index][:idx] + \
                                lines_copy[last_rbrace_line_index][idx + 1:]
                            scope.append(type_block)
                if contains_match_outside_strings(re_method_declaration, line):
                    suspect_line = re.sub(
                        r'COMMENT\d+', "", lines[i - 1]).strip()
                    if suspect_line not in ['{', '}'] and 0 < len(suspect_line) < 3:
                        lines_copy[i + insert_times - 1] = '}\n'
                        last_rbrace_line_index = i + insert_times - 1
                        scope.pop()
                    # we assume that a 'method declaration' cannot be nested within another 'method declaration'
                    # so, we close the block at the top of the stack until we reach a non-'method declaration' or the stack is empty
                    while scope and scope[-1] == method_block:
                        lines_copy.insert(i + insert_times, '}\n')
                        last_rbrace_line_index = i + insert_times
                        insert_times = insert_times + 1
                        scope.pop()
                    scope.append(method_block)
                # if the line is not a 'type declaration' or a 'method declaration', check if it is a 'normal block'
                else:
                    # given the code is formatted, if the current line contains '{', it must be located at the end of the line
                    # so, we find right braces in the line first
                    right_braces = re.finditer('\}', line)
                    for right_brace in right_braces:
                        in_string = False
                        for match in re.finditer('"[^"]*"', line):
                            if right_brace.start() >= match.start() and right_brace.end() <= match.end():
                                in_string = True
                                break
                        if not in_string:
                            if scope:
                                scope.pop()
                            else:
                                lines_copy[i + insert_times] = line[:right_brace.span()[0]] + \
                                    '' + line[right_brace.span()[0] + 1:]
                    if contains_match_outside_strings('\}', lines_copy[i + insert_times]):
                        last_rbrace_line_index = i + insert_times
                    # if the line contains '{', it signifies a 'normal block'
                    if contains_match_outside_strings('\{', line):
                        scope.append(normal_block)

    # iterate over the lines in reverse order, and fix any suspected '}'
    for i in range(len(lines_copy) - 1, -1, -1):
        if scope:
            if lines_copy[i].strip():
                suspect_line = re.sub(r'COMMENT\d+', "", lines_copy[i]).strip()
                if suspect_line not in ['{', '}', '};', ';'] and 0 < len(suspect_line) < 3:
                    lines_copy[i] = '}\n'
                    scope.pop()
        else:
            break
    # close any remaining blocks
    while scope:
        scope.pop()
        lines_copy.append('}\n')

    return lines_copy


def is_legal_digit(s: str) -> bool:
    """
    Check if a given string is a legal digit.

    Args:
        s: The input string.

    Returns:
        bool: True if the input string is a legal digit; False otherwise.
    """
    try:
        if re.match('0b', s):
            int(s, base=2)
        elif re.match('0o', s):
            int(s, base=8)
        elif re.match('0x', s):
            int(s, base=16)
        else:
            int(s)
    except ValueError:
        return False
    else:
        return True


def preprocess_tokens(tokens: list, reserved_words: list) -> tuple:
    """
    Preprocess the tokens.

    Args:
        tokens: A list of tokens.
        reserved_words: A set of reserved words.
    
    Returns:
        A tuple containing the following elements:
        - token_lines: A list of tokenized code lines.
        - split_ids_set: A set of identifiers that may be recognized as multiple identifiers.
        - linked_id_set: A set of identifiers that may be a combination of a reserved word and a token.
        - id_occurrence_map: A map with keys as identifiers and values as lists of indices where identifiers are present.
    """
    token_lines = []
    # the key represents the identifier, and the value corresponds to a list of indices where the identifier is present
    id_occurrence_map = {}
    # a set of identifiers that may be a combination of a reserved word and a token
    linked_id_set = []
    # a set of identifiers that may be recognized as multiple identifiers
    split_ids_set = []

    # the previous token in the same line, set to None when a new line is encountered
    pre_token = None
    current_line_index = 1
    current_line_tokens = []
    token_index = -1  # the index of the current token in all tokens

    # iterate over the tokens
    for token in tokens:
        # if the token is not '<EOF>' or whitespace
        if token.type not in [-1, 125]:
            token_index = token_index + 1
            if token.line != current_line_index:
                token_lines.append(current_line_tokens.copy())
                increment = token.line - current_line_index
                for i in range(0, increment - 1):
                    token_lines.append([])
                current_line_index = token.line
                current_line_tokens.clear()
                pre_token = None
            # if the token is an identifier
            if token.type == 128:
                if pre_token is not None and pre_token.type == 128:
                    # In cases where a identifier is recognized as multiple identifiers
                    if not split_ids_set:
                        # split_ids = [id_count, first_id, line position of first_id, token_index of first_id]
                        split_ids = [2, pre_token, len(
                            current_line_tokens) - 1, token_index - 1]
                        split_ids_set.append(split_ids)
                    else:
                        last_split_ids = split_ids_set[-1]
                        # the number of consecutive identifiers in 'last_split_ids'
                        id_count = last_split_ids[0]
                        # the first identifier in 'last_split_ids'
                        first_id = last_split_ids[1]
                        # the position of the first identifier in the line
                        first_id_line_pos = last_split_ids[2]

                        if first_id.line == current_line_index:
                            # the last token in the last 'split_ids'
                            last_token = current_line_tokens[first_id_line_pos + id_count - 1]
                            # if the previous token is the last token in the last 'split_ids'
                            # it signifies that the current token belongs to the last 'split_ids'
                            if last_token.tokenIndex == pre_token.tokenIndex:
                                last_split_ids[0] += 1
                            else:
                                # if not, it signifies that the previous token and the current token form a new 'split_ids'
                                split_ids = [2, pre_token, len(
                                    current_line_tokens) - 1, token_index - 1]
                                split_ids_set.append(split_ids)
                        else:
                            # the identifier in 'split_ids' must be in the same line
                            split_ids = [2, pre_token, len(
                                current_line_tokens) - 1, token_index - 1]
                            split_ids_set.append(split_ids)
                else:
                    if pre_token is not None:
                        # if the previous token is a 'DECIMAL_LITERAL' or a 'FLOAT_LITERAL' (e.g., .1)
                        # in general, 'DECIMAL/FLOAT_LITERAL IDENTIFIER' is less likely to appear in the code
                        # so, it may be an OCR recognition error that causes an identifier to start with a number
                        if pre_token.type == 67:  # if the previous token is a 'DECIMAL_LITERAL'
                            current_line_tokens[-1].text = ''
                            # In cases where 'l' is recognized as '1'
                            if pre_token.text[:1] == '1':
                                token.text = pre_token.text.replace(
                                    '1', 'l') + token.text
                        elif pre_token.type == 71:  # if the previous token is a 'FLOAT_LITERAL'
                            current_line_tokens[-1].text = ''
                            if pre_token.text[:1] == '.':
                                current_line_tokens[-1].text = '.'
                                # In cases where 'l' is recognized as '1'
                                if pre_token.text[1: 2] == '1':
                                    token.text = pre_token.text[1:].replace(
                                        '1', 'l') + token.text
                    if token.text not in reserved_words:
                        for reserved_word in reserved_words:
                            # if the token is similar to a reserved word, it may be an OCR recognition error
                            # we replace the token with the reserved word
                            if len(reserved_word) == len(token.text) and len(reserved_word) > 3:
                                if distance.hamming(reserved_word, token.text) == 1:
                                    token.text = reserved_word
                                    break
                            # if the beginning of the token matches a reserved word, and the remaining part is a legal identifier
                            # we consider it may be a combination of a reserved word and a token
                            # add it to the 'linked_id_set'
                            re_result = re.match(reserved_word, token.text)
                            if re_result is not None:
                                # the end position of the reserved word
                                stop = re_result.span()[1]
                                # linked_id = [stop, token, line position of token, token_index]
                                linked_id = [stop, token, len(
                                    current_line_tokens), token_index]
                                linked_id_set.append(linked_id)
                id_occurrence_map.setdefault(
                    token.text, []).append(token_index)
            current_line_tokens.append(token.clone())
            pre_token = token
    # add the last line of tokens
    if current_line_tokens:
        token_lines.append(current_line_tokens)

    return token_lines, split_ids_set, linked_id_set, id_occurrence_map


def splitting_one_into_many_repair(split_ids_set: list, token_lines: list, id_occurrence_map: dict, reserved_words: list) -> None:
    """
    Repair the identifiers that may be recognized as multiple identifiers.

    Args:
        split_ids_set: A set of identifiers that may be recognized as multiple identifiers.
        token_lines: A list of tokenized code lines.
        id_occurrence_map: A map with keys as identifiers and values as lists of indices where identifiers are present.
        reserved_words: A set of reserved words.
    """
    for split_ids in split_ids_set:
        id_count = split_ids[0]  # the number of consecutive identifiers in 'split_ids'
        first_id = split_ids[1]  # the first identifier in 'split_ids'
        # the position of the first identifier in the line
        first_id_line_pos = split_ids[2]
        # the index of the first identifier in all tokens
        first_id_index = split_ids[3]
        combined_text = ''  # the combined text of 'split_ids' with spaces
        underline_combined_text = ''  # the combined text of 'split_ids' with underscores
        # the list of indices where the identifier in 'split_ids' appears after the first occurrence
        index_list = []
        for i in range(id_count):
            id_token = token_lines[first_id.line - 1][first_id_line_pos + i]
            combined_text += id_token.text
            underline_combined_text += f'{id_token.text}_'
            if id_token.text in id_occurrence_map.keys():
                value = id_occurrence_map[id_token.text]
                for record in value:
                    if first_id_index + i < record:
                        index_list.append(record)
        underline_combined_text = underline_combined_text[0: -1]

        condition_1 = True
        condition_2 = True
        combined_text_exists = False
        underline_combined_text_exists = False

        # condition1: the combined text of 'split_ids' (with spaces or underscores) has appeared in the entire code
        # condition2: if the identifiers in 'split_ids' do not appear again, or each time they appear, they appear together in order
        # if both conditions are met, it is very likely that an identifier is recognized as multiple identifiers by OCR
        if combined_text in id_occurrence_map.keys():
            combined_text_exists = True
        elif underline_combined_text in id_occurrence_map.keys():
            underline_combined_text_exists = True
        else:
            condition_1 = False

        if len(index_list) != 0:
            index_list.sort()
            if len(index_list) % id_count == 0:
                group = len(index_list) // id_count
                for i in range(group):
                    for j in range(id_count - 1):
                        if index_list[i * id_count + j] != index_list[i * id_count + j + 1]:
                            condition_2 = False
                            break
                    if condition_2 == False:
                        break
            else:
                condition_2 = False

        if combined_text in reserved_words:
            if condition_2:
                for i in range(id_count):
                    if i == 0:
                        token_lines[first_id.line -
                                    1][first_id_line_pos + i].text = combined_text
                    else:
                        token_lines[first_id.line -
                                    1][first_id_line_pos + i].text = ''
        else:
            if condition_1 and condition_2:
                for i in range(id_count):
                    if i == 0:
                        if combined_text_exists:
                            token_lines[first_id.line -
                                        1][first_id_line_pos + i].text = combined_text
                        else:
                            token_lines[first_id.line - 1][first_id_line_pos +
                                                           i].text = underline_combined_text
                    else:
                        token_lines[first_id.line -
                                    1][first_id_line_pos + i].text = ''


def combining_two_into_one_repair(linked_id_set: list, id_occurrence_map: dict, reserved_words: list, token_lines: list) -> None:
    """
    Repair the identifiers that may be a combination of a reserved word and a token.

    Args:
        linked_id_set: A set of identifiers that may be a combination of a reserved word and a token.
        id_occurrence_map: A map with keys as identifiers and values as lists of indices where identifiers are present.
        reserved_words: A set of reserved words.
        token_lines: A list of tokenized code lines.
    """
    for linked_id in linked_id_set:
        # the identifier that may be a combination of a reserved word and a token
        id_token = linked_id[1]
        id_line_pos = linked_id[2]  # the position of 'id_token' in the line

        # if 'id_token' does not appear again in the code, it may be a real 'linked_id'
        # otherwise, it can't be determined as a real 'linked_id', and no processing is performed
        if len(id_occurrence_map[id_token.text]) == 1:
            stop = linked_id[0]  # the end position of the reserved word
            # the remaining part of the identifier
            remainder = id_token.text[stop:]
            if remainder == '':
                continue
            elif remainder in reserved_words:
                token_lines[id_token.line - 1][id_line_pos].text = id_token.text[:stop] + \
                    ' ' + id_token.text[stop:]
            else:
                # a legal identifier cannot start with a digit
                # so, if the first character of 'remainder' is a digit, it is not a legal identifier
                if remainder[0].isdigit():
                    # if 'remainder' is a legal digit, 'id_token' may be a real 'linked_id'
                    # we split 'id_token' into two parts
                    if is_legal_digit(remainder):
                        token_lines[id_token.line - 1][id_line_pos].text = id_token.text[:stop] + \
                            ' ' + id_token.text[stop:]
                    # if 'remainder' is not a legal identifier or a legal digit
                    # 'id_token' can't be determined as a real 'linked_id', and no processing is performed
                # if the first character of 'remainder' is not a digit, we consider it as a legal identifier
                else:
                    # if 'id_token' does not appear again in the code, and 'remainder' appears in the subsequent code, it may be a real 'linked_id'
                    # if the matched reserved word is 'return', 'package', or 'import', it suffices to satisfy that 'id_token' doesn't reoccur in the code.
                    if id_token.text[:stop] in ['return', 'package', 'import']:
                        token_lines[id_token.line - 1][id_line_pos].text = id_token.text[:stop] + \
                            ' ' + id_token.text[stop:]
                    else:
                        if remainder in id_occurrence_map.keys():
                            # the index of 'id_token' in all tokens
                            index = linked_id[3]
                            value = id_occurrence_map[remainder]
                            for record in value:
                                if index < record:
                                    token_lines[id_token.line - 1][id_line_pos].text = id_token.text[:stop] + \
                                        ' ' + id_token.text[stop:]
                                    break
