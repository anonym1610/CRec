import Levenshtein
import os
import random
import re
import string
import time
import unicodedata
from antlr4 import *
from antlr4.error.ErrorListener import *
from multiprocessing import Process
from threading import Thread

from HandleErrorListener import HandleErrorListener
from JavaLexer import JavaLexer
from JavaParser import JavaParser
from RepairTool import RepairTool


def full_to_half(input: str):
    input = unicodedata.normalize('NFKC', input)
    table = {ord(f): ord(h) for f, h in zip(u'‘’“”【】、', '\'\'""[]\\')}
    input = input.translate(table)
    return input


def is_legal_digit(s: str):
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


def comment_complement(lines: list) -> list:
    text = ''.join(lines)
    text = re.sub('\r', '\n', text)
    text = re.sub('\n+', '\n', text)
    lines = text.split('\n')
    for i, line in enumerate(lines):
        lines[i] = line + '\n'

    start = False
    for i, line in enumerate(lines):
        line = line.lstrip()
        if start == False and re.match('(\*|/\*)', line) is not None:
            start = True
            if line[0] == '*':
                if i == 0:
                    lines[i] = '/*' + lines[i]
                else:
                    if re.match('/\*', lines[i - 1].lstrip()) is None:
                        lines[i - 1] = '/*' + lines[i - 1]
        if start == True:
            if i == len(lines) - 1:
                if re.match('(\*|\*/)', line) is None:
                    start = False
                    if not lines[i - 1].rstrip().endswith('*/'):
                        lines[i] = '*/' + lines[i]
            else:
                if re.match('(\*|\*/)', lines[i + 1]) is None:
                    start = False
                    if not line.rstrip().endswith('*/'):
                        lines[i] = lines[i] + '*/'
    return lines


def simplified_brace_complement(lines: list) -> list:
    text = ''.join(lines)
    text = re.sub('(\n|\r){', ' {\n', text)
    lines = text.split('\n')
    for i, line in enumerate(lines):
        lines[i] = line + '\n'
    lines_copy = lines.copy()

    insert_times = 0
    scope = []
    type_block = 0
    method_block = 1
    normal_block = 2
    re_ignore = '(package|import|open|module|COMMENT|extends|implements|permits|throw|throws|@)'
    re_typeDeclaration = '(class|interface|enum)\s+\w+(\s+extends\s+\w+)?(\s+implements\s+\w+(\s*,\s*\w+)*\s*)?\s*'  # simplified
    re_str = '"[^"]*"'
    re_methodDeclaration = '(public|protected|private|static|\s) +[\w\<\>\[\]\,\s]*\s*(\w+) *\([^\)]*\) *(\{?|[^;])'

    for i, line in enumerate(lines):
        if line.isspace() == False and line != '' and re.match(re_ignore, line.lstrip()) is None:
            typeDeclaration_results = re.finditer(re_typeDeclaration, line)
            str_results = re.finditer(re_str, line)
            is_typeDeclaration = False
            # determines if the line contains a "typeDeclaration"
            if any(typeDeclaration_results):
                if any(str_results):
                    for t_result in typeDeclaration_results:
                        t_span = t_result.span()
                        for s_result in str_results:
                            s_span = s_result.span()
                            # Existing "class", "interface", "record" but not in the string, that is a "typeDeclaration"
                            if not (t_span[0] > s_span[0] and t_span[1] < s_span[1]):
                                is_typeDeclaration = True
                                break
                        if is_typeDeclaration:
                            break
                else:
                    is_typeDeclaration = True
            # if the line contains a "typeDeclaration"
            if is_typeDeclaration:
                if len(scope) != 0:
                    candidate_line = re.compile(r'COMMENT\d+').sub("", lines[i - 1]).strip()
                    '''
                    } should be inserted on the preceding line
                    but if the preceding line has only 1 or 2 characters other than comments
                    OCR probably misised the }
                    so replace the preceding line with }
                    '''
                    if candidate_line != '}' and (0 < len(candidate_line) < 3):
                        lines_copy[i + insert_times - 1] = '}\n'
                        scope.pop()
                scope.append(type_block)
            else:
                if len(scope) == 0 and re.search('\w', line) is not None:
                    random_str = ''.join(random.choice(string.ascii_letters) for _ in range(7))
                    class_header = 'class ' + random_str + ' {\n'
                    lines_copy.insert(i + insert_times, class_header)
                    insert_times = insert_times + 1
                    scope.append(type_block)
                # if the line contains a "methodDeclaration"
                if re.search(re_methodDeclaration, line.lstrip()) is not None:
                    candidate_line = re.compile(r'COMMENT\d+').sub("", lines[i - 1]).strip()
                    if candidate_line != '}' and (0 < len(candidate_line) < 3):
                        lines_copy[i + insert_times - 1] = '}\n'
                        scope.pop()
                    while len(scope) != 0 and (scope[-1] == normal_block or scope[-1] == method_block):
                        lines_copy.insert(i + insert_times, '}\n')
                        insert_times = insert_times + 1
                        scope.pop()
                    scope.append(method_block)
                # if the line contains a "normal_block"
                elif re.search('{', line) is not None:
                    scope.append(normal_block)
            right_braces = re.finditer('}', line)
            for right_brace in right_braces:
                if len(scope) != 0:
                    scope.pop()
                else:
                    lines_copy[i] = line[: right_brace.span()[0]] + ' ' + line[right_brace.span()[0] + 1:]
    for i in range(len(lines_copy) - 1, -1, -1):
        if len(scope) != 0:
            if lines_copy[i].isspace() == False and lines_copy[i] != '':
                candidate_line = re.compile(r'COMMENT\d+').sub("", lines_copy[i]).strip()
                if 0 < len(candidate_line) < 2:
                    if candidate_line != '}':
                        lines_copy[i] = '}\n'
                        scope.pop()
                else:
                    break
        else:
            break
    while len(scope) != 0:
        scope.pop()
        lines_copy.append('}')
    return lines_copy


def preprocess(tokens: list, output_file_path):
    keywords = ['abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'Class', 'class', 'const',
                'continue', 'default', 'do', 'double', 'else', 'enum', 'extends', 'final', 'finally', 'float',
                'for', 'if', 'goto', 'implements', 'import', 'instanceof', 'int', 'interface', 'lang', 'long', 'native',
                'new', 'package', 'private', 'protected', 'public', 'return', 'short', 'static', 'strictfp',
                'super', 'switch', 'synchronized', 'this', 'throw', 'throws', 'transient', 'try', 'void', 'volatile',
                'while', 'module', 'open', 'requires', 'exports', 'opens', 'to', 'uses', 'provides',
                'with', 'transitive', 'var', 'yield', 'record', 'sealed', 'permits', 'non-sealed', 'null']

    if len(tokens) != 0:
        pre_token = None  # Only record the previous token of the current token in the same line, and set None when wrapping a line
        current_line_index = 1
        token_lines = []
        tokens_of_current_line = []
        id_map = {}  
        order_no = -1  
        linked_id_set = []  
        parted_ids_set = []  
        for token in tokens:
            if token.type != 125 and token.text != '<EOF>':
                order_no = order_no + 1
                if token.line != current_line_index:
                    token_lines.append(tokens_of_current_line.copy())
                    increment = token.line - current_line_index
                    for i in range(0, increment - 1):
                        token_lines.append([])
                    current_line_index = token.line
                    tokens_of_current_line.clear()
                    pre_token = None
                
                if token.type == 128:
                    
                    if pre_token is not None:
                        if pre_token.type == 67:
                            tokens_of_current_line[-1].text = ''
                            if pre_token.text[: 1] == '1':
                                token.text = pre_token.text.replace('1', 'l') + token.text
                                if token.text not in keywords:
                                    for keyword in keywords:
                                        if len(keyword) == len(token.text) and len(keyword) > 3:
                                            if Levenshtein.hamming(keyword, token.text) == 1:
                                                token.text = keyword
                                                break

                                        re_result = re.match(keyword, token.text)
                                        if re_result is not None:
                                            stop = re_result.span()[1]
                                            linked_id_set.append([stop, token, len(tokens_of_current_line), order_no])
                        elif pre_token.type == 71:
                            tokens_of_current_line[-1].text = ''
                            if pre_token.text[: 1] == '.':
                                tokens_of_current_line[-1].text = '.'
                                if pre_token.text[1: 2] == '1':
                                    token.text = pre_token.text[1:].replace('1', 'l') + token.text
                                    if token.text not in keywords:
                                        for keyword in keywords:
                                            if len(keyword) == len(token.text) and len(keyword) > 3:
                                                if Levenshtein.hamming(keyword, token.text) == 1:
                                                    token.text = keyword
                                                    break

                                            re_result = re.match(keyword, token.text)
                                            if re_result is not None:
                                                stop = re_result.span()[1]
                                                linked_id_set.append(
                                                    [stop, token, len(tokens_of_current_line), order_no])
                        elif pre_token.type == 128:
                            if len(parted_ids_set) == 0:
                                parted_ids = [2, pre_token, len(tokens_of_current_line) - 1, order_no - 1]
                                parted_ids_set.append(parted_ids)
                            else:
                                last_parted_ids = parted_ids_set[-1]  
                                num_of_tokens = last_parted_ids[0]  
                                first_token = last_parted_ids[1]  
                                first_token_location = last_parted_ids[2]  
                                if first_token.line == current_line_index:
                                    last_token = tokens_of_current_line[
                                        first_token_location + num_of_tokens - 1]  
                                else:
                                    last_token = token_lines[first_token.line - 1][
                                        first_token_location + num_of_tokens - 1]  
                                if last_token.tokenIndex == pre_token.tokenIndex:
                                    last_parted_ids[0] = last_parted_ids[0] + 1
                                else:
                                    parted_ids = [2, pre_token, len(tokens_of_current_line) - 1, order_no - 1]
                                    parted_ids_set.append(parted_ids)
                        else:
                            if token.text not in keywords:
                                for keyword in keywords:
                                    if len(keyword) == len(token.text) and len(keyword) > 3:
                                        if Levenshtein.hamming(keyword, token.text) == 1:
                                            token.text = keyword
                                            break

                                    re_result = re.match(keyword, token.text)
                                    if re_result is not None:
                                        stop = re_result.span()[1]
                                        linked_id_set.append([stop, token, len(tokens_of_current_line), order_no])
                    else:
                        if token.text not in keywords:
                            for keyword in keywords:
                                if len(keyword) == len(token.text) and len(keyword) > 3:
                                    if Levenshtein.hamming(keyword, token.text) == 1:
                                        token.text = keyword
                                        break

                                re_result = re.match(keyword, token.text)
                                if re_result is not None:
                                    stop = re_result.span()[1]
                                    linked_id_set.append([stop, token, len(tokens_of_current_line), order_no])

                    if token.text in id_map.keys():
                        value = id_map[token.text]
                        value.append(order_no)
                        id_map[token.text] = value
                    else:
                        id_map[token.text] = [order_no]
                tokens_of_current_line.append(token.clone())
                pre_token = token

        if len(tokens_of_current_line) != 0:
            token_lines.append(tokens_of_current_line)
        for linked_id in linked_id_set:
            id = linked_id[1]  
            location = linked_id[2]  

            if len(id_map[id.text]) == 1:
                stop = linked_id[0]  
                remainder: str = id.text[stop:]  
                if remainder == '':
                    continue
                elif remainder in keywords:
                    token_lines[id.line - 1][location].text = id.text[: stop] + ' ' + id.text[stop:]
                else:
                    
                    if remainder[0].isdigit():
                        if is_legal_digit(remainder):
                            token_lines[id.line - 1][location].text = id.text[: stop] + ' ' + id.text[stop:]

                    else:
                        if remainder in keywords:
                            token_lines[id.line - 1][location].text = id.text[: stop] + ' ' + id.text[stop:]
                        else:
                            if id.text[: stop] in ['return', 'package', 'import']:
                                token_lines[id.line - 1][location].text = id.text[: stop] + ' ' + id.text[stop:]
                            else:
                                if remainder in id_map.keys():
                                    index = linked_id[3]  
                                    value = id_map[remainder] 
                                    for record in value:
                                        if index < record:
                                            
                                            token_lines[id.line - 1][location].text = id.text[: stop] + ' ' + id.text[
                                                                                                              stop:]
                                            break

        for parted_ids in parted_ids_set:
            id_num = parted_ids[0]  
            first_id = parted_ids[1]  
            first_id_location = parted_ids[2]  
            first_id_index = parted_ids[3]  
            combined_text = '' 
            combined_text_with_underline = ''  
            order_no_list = []  
            for i in range(id_num):
                id = token_lines[first_id.line - 1][first_id_location + i]
                combined_text = combined_text + id.text
                combined_text_with_underline = combined_text_with_underline + id.text + '_'
                if id.text in id_map.keys():
                    value = id_map[id.text]  
                    for record in value:
                        if (first_id_index + i) < record:
                            order_no_list.append(record)
            combined_text_with_underline = combined_text_with_underline[0: -1]

            condition_1 = True
            condition_2 = True

            combined_text_exists = False
            combined_text_with_underline_exists = False
            
            if combined_text in id_map.keys():
                combined_text_exists = True
            elif combined_text_with_underline in id_map.keys():
                combined_text_with_underline_exists = True
            else:
                condition_1 = False

            if len(order_no_list) != 0:
                order_no_list.sort()
                
                if len(order_no_list) % id_num == 0:
                    group = len(order_no_list) // id_num
                    for i in range(group):
                        for j in range(id_num - 1):
                            if order_no_list[i * id_num + j] != order_no_list[i * id_num + j + 1]:
                                condition_2 = False
                                break
                        if condition_2 == False:
                            break
                else:
                    condition_2 = False
            if combined_text in keywords:
                if condition_2:
                    for i in range(id_num):
                        if i == 0:
                            if combined_text_exists:
                                token_lines[first_id.line - 1][first_id_location + i].text = combined_text
                            else:
                                token_lines[first_id.line - 1][
                                    first_id_location + i].text = combined_text_with_underline
                        else:
                            token_lines[first_id.line - 1][first_id_location + i].text = ''
            else:
                if condition_1 and condition_2:
                    for i in range(id_num):
                        if i == 0:
                            if combined_text_exists:
                                token_lines[first_id.line - 1][first_id_location + i].text = combined_text
                            else:
                                token_lines[first_id.line - 1][
                                    first_id_location + i].text = combined_text_with_underline
                        else:
                            token_lines[first_id.line - 1][first_id_location + i].text = ''

        lines = []
        for token_line in token_lines:
            if len(token_line) != 0:
                current_line = ''
                for token in token_line:
                    current_line = current_line + token.text + ' '
                current_line = re.sub('(\r|\n)(\r|\n)+', '\n', current_line)
                lines.append(current_line.strip() + '\n')

        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)


def repair(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as input:
        lines = input.readlines()
        for i, line in enumerate(lines):
            lines[i] = full_to_half(line)

    random_str = ''.join(random.choice(string.ascii_letters) for _ in range(7))
    class_header = 'class ' + random_str + ' {\n'
    lines.insert(0, class_header)
    lines.append('}')

    lines = comment_complement(lines.copy())

    with open(output_file_path, 'w', encoding='utf-8') as output:
        output.writelines(lines)

    tokens_file_path = './JavaParser.tokens'
    token_model_file_path = './6_gram.bin'
    repair_tool = RepairTool(tokens_file_path, token_model_file_path, output_file_path)

    istream = FileStream(output_file_path, encoding='utf-8')
    lexer = JavaLexer(istream)
    stream = CommonTokenStream(lexer)
    parser = JavaParser(stream)
    parser.removeErrorListeners()
    tree = parser.compilationUnit()
    preprocess(parser.getInputStream().tokens, output_file_path)

    istream = FileStream(output_file_path, encoding='utf-8')
    lexer = JavaLexer(istream)
    stream = CommonTokenStream(lexer)
    parser = JavaParser(stream)
    parser.removeErrorListeners()
    tree = parser.compilationUnit()
    all_tokens = parser.getInputStream().tokens
    comments = []
    text = ''
    line_index = 1
    for token in all_tokens:
        if token.line != line_index:
            text = text + '\n'
            line_index = token.line
        if token.type == 126 or token.type == 127:
            comments.append(token.text)
            token.text = 'COMMENT' + str(len(comments) - 1)
        if token.text != '<EOF>':
            text = text + token.text
    text = re.sub('(\r|\n)(\r|\n)+', '\n', text)
    lines = text.split('\n')
    for i, line in enumerate(lines):
        lines[i] = line.strip() + '\n'

    lines.pop(0)
    for i in range(len(lines) - 1, -1, -1):
        index = lines[i].rfind('}')
        if index != -1:
            lines[i] = lines[i][: index] + lines[i][index + 1:]
            break

    lines = simplified_brace_complement(lines)

    text = ''.join(lines)
    for i, comment in enumerate(comments):
        text = re.sub('COMMENT' + str(i), comment, text)
    text = re.sub('\r', '\n', text)
    text = re.sub('\n+', '\n', text)
    lines = text.split('\n')

    for i, line in enumerate(lines):
        lines[i] = line + '\n'

    with open(output_file_path, 'w', encoding='utf-8') as output:
        output.writelines(lines)

    istream = FileStream(output_file_path, encoding='utf-8')
    lexer = JavaLexer(istream)
    stream = CommonTokenStream(lexer)
    parser = JavaParser(stream)
    parser.removeErrorListeners()
    handle_error_listener = HandleErrorListener(repair_tool)
    parser.addErrorListener(handle_error_listener)
    tree = parser.compilationUnit()
    while parser._syntaxErrors != repair_tool.error_tolerance:
        istream = FileStream(output_file_path, encoding='utf-8')
        lexer = JavaLexer(istream)
        stream = CommonTokenStream(lexer)
        parser = JavaParser(stream)
        parser.removeErrorListeners()
        parser.addErrorListener(handle_error_listener)
        tree = parser.compilationUnit()

    with open(output_file_path, 'r', encoding='utf-8') as output:
        lines = output.readlines()

    istream = FileStream(output_file_path, encoding='utf-8')
    lexer = JavaLexer(istream)
    stream = CommonTokenStream(lexer)
    parser = JavaParser(stream)
    tree = parser.compilationUnit()
    all_tokens = parser.getInputStream().tokens
    comments = []
    text = ''
    line_index = 1
    for token in all_tokens:
        if token.line != line_index:
            text = text + '\n'
            line_index = token.line
        if token.type == 126 or token.type == 127:
            comments.append(token.text)
            token.text = 'COMMENT' + str(len(comments) - 1)
        if token.text != '<EOF>':
            text = text + token.text
    text = re.sub('(\r|\n)(\r|\n)+', '\n', text)
    lines = text.split('\n')
    for i, line in enumerate(lines):
        lines[i] = line.strip() + '\n'

    lines = simplified_brace_complement(lines)
    text = ''.join(lines)
    for i, comment in enumerate(comments):
        text = re.sub('COMMENT' + str(i), comment, text)
    text = re.sub('\r', '\n', text)
    text = re.sub('\n+', '\n', text)
    lines = text.split('\n')
    for i, line in enumerate(lines):
        lines[i] = line + '\n'

    with open(output_file_path, 'w', encoding='utf-8') as output:
        output.writelines(lines)


def monitor(proc: Process):
    time.sleep(20)
    if proc.is_alive():
        proc.terminate()


if __name__ == '__main__':

    input_dir = "./input"
    output_dir = "./output"
    failures_dir = "./output/failures"

    input_files = os.listdir(input_dir)

    for input_file in input_files:
        input_file_path = os.path.join(input_dir, input_file)
        output_file_path = os.path.join(output_dir, input_file)

        try:
            proc = Process(target=repair, args=(input_file_path, output_file_path,))
            thread = Thread(target=monitor, args=(proc,))
            proc.start()
            print(input_file + ' is being repaired...')
            thread.start()
            proc.join()
        except Exception as e:
            failure_record = os.path.join(failures_dir, input_file)
            with open(failure_record, 'w') as f:
                f.write(repr(e))
