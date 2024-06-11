import os
import re
import shutil
import subprocess
import time
import torch
import traceback

from antlr4 import *
from antlr4.error.ErrorListener import *
from multiprocessing import Process
from threading import Thread

from error_listener import HandleErrorListener
from JavaLexer import JavaLexer
from JavaParser import JavaParser
from repair_tool import RepairTool
from rule import *
from unixcoder import UniXcoder


def repair(model_path: str, tmp_dir: str, input_file_path: str, output_file_path: str, error_file_path: str) -> None:
    """
    The implementation of CRec's repair process.

    Args:
        model_path: The path to the UniXcoder model.
        tmp_dir: The path to the temporary directory.
        input_file_path: The path to the input file.
        output_file_path: The path to the output file.
        error_file_path: The path to the error file.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UniXcoder(model_path)
        model.to(device)

        rule_based_repair(tmp_dir, input_file_path, output_file_path)

        cmd = ' '.join(['javac', output_file_path, '-d',
                       tmp_dir, '-encoding', 'utf-8'])
        if os.system(cmd) == 0:
            return

        parser_guided_repair(model, tmp_dir, output_file_path)

        # parser_guided_repair might disrupt the balance of braces again
        # Replace the comments with COMMENT+number
        istream = FileStream(output_file_path, encoding='utf-8')
        lexer = JavaLexer(istream)
        stream = CommonTokenStream(lexer)
        stream.fill()  # Force the lexer to tokenize the input
        all_tokens = stream.tokens
        comments = []
        text = ''
        for token in all_tokens:
            if token.type in [126, 127]:
                comments.append(token.text)
                token.text = f'COMMENT{str(len(comments) - 1)}'
            if token.type != -1:
                text = text + token.text
        lines = text.split('\n')
        for i, line in enumerate(lines):
            lines[i] = line.strip() + '\n'
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        output_file_copy_path = os.path.join(tmp_dir, 'output_copy.java')
        shutil.copyfile(output_file_path, output_file_copy_path)
        clang_format_path = 'clang-format.exe'
        cmd = ' '.join([clang_format_path, '-i', output_file_copy_path])
        # cmd = ' '.join(['clang-format', '-i', output_file_copy_path])
        try:
            subprocess.run(cmd, shell=True, timeout=2, check=True)
            with open(output_file_copy_path, 'r', encoding='utf-8') as f:
                text = f.read()
                for i, comment in enumerate(comments):
                    text = re.sub(
                        f'\\bCOMMENT{str(i)}\\b\n*', f'COMMENT{str(i)}\n', text)
                lines = [line + '\n' for line in text.split('\n')]
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            print('clang-format failed.')
            with open(output_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

        lines = simplified_brace_complement_gstyle(lines)

        # Convert COMMENT+number back to original comments
        text = ''.join(lines)
        for i, comment in enumerate(comments):
            text = re.sub(f'\\bCOMMENT{str(i)}\\b', comment, text)

        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(text)

        shutil.copyfile(output_file_path, output_file_copy_path)
        clang_format_path = 'clang-format.exe'
        cmd = ' '.join([clang_format_path, '-i', output_file_copy_path])
        # cmd = ' '.join(['clang-format', '-i', output_file_copy_path])
        try:
            subprocess.run(cmd, shell=True, timeout=2, check=True)
            shutil.copyfile(output_file_copy_path, output_file_path)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            print('clang-format failed.')

    except Exception as e:
        error_info = traceback.format_exc()
        with open(error_file_path, 'a') as f:
            f.write(error_info)


def rule_based_repair(tmp_dir: str, input_file_path: str, output_file_path: str) -> None:
    """
    The implementation of CRec's rule-based repair process.

    Args:
        tmp_dir: The path to the temporary directory.
        input_file_path: The path to the input file.
        output_file_path: The path to the output file.
    """
    with open(input_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            lines[i] = full_to_half(line)

    repair_multi_line_comment(lines)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    istream = FileStream(output_file_path, encoding='utf-8')
    lexer = JavaLexer(istream)
    stream = CommonTokenStream(lexer)
    stream.fill()  # Force the lexer to tokenize the input

    reserved_words = [
        'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char',
        'Class', 'class', 'const', 'continue', 'default', 'do', 'double', 'else',
        'enum', 'extends', 'final', 'finally', 'float', 'for', 'if', 'goto',
        'implements', 'import', 'instanceof', 'int', 'interface', 'lang', 'long',
        'native', 'new', 'package', 'private', 'protected', 'public', 'return',
        'short', 'static', 'strictfp', 'super', 'switch', 'synchronized', 'this',
        'throw', 'throws', 'transient', 'try', 'void', 'volatile', 'while',
        'module', 'open', 'requires', 'exports', 'opens', 'to', 'uses',
        'provides', 'with', 'transitive', 'var', 'yield', 'record', 'sealed',
        'permits', 'non-sealed', 'null'
    ]

    if stream.tokens:
        token_lines, split_ids_set, linked_id_set, id_occurrence_map = preprocess_tokens(
            stream.tokens, reserved_words)
        combining_two_into_one_repair(
            linked_id_set, id_occurrence_map, reserved_words, token_lines)
        splitting_one_into_many_repair(
            split_ids_set, token_lines, id_occurrence_map, reserved_words)

        lines = []
        for token_line in token_lines:
            if token_line:
                current_line = ' '.join([token.text for token in token_line])
                lines.append(current_line + '\n')

    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    # Replace the comments with COMMENT+number
    istream = FileStream(output_file_path, encoding='utf-8')
    lexer = JavaLexer(istream)
    stream = CommonTokenStream(lexer)
    stream.fill()  # Force the lexer to tokenize the input
    all_tokens = stream.tokens
    comments = []
    text = ''
    for token in all_tokens:
        if token.type in [126, 127]:
            comments.append(token.text)
            token.text = 'COMMENT' + str(len(comments) - 1)
        if token.type != -1:
            text = text + token.text
    lines = text.split('\n')
    for i, line in enumerate(lines):
        lines[i] = line.strip() + '\n'
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    output_file_copy_path = os.path.join(tmp_dir, 'output_copy.java')
    shutil.copyfile(output_file_path, output_file_copy_path)
    clang_format_path = 'clang-format.exe'
    cmd = ' '.join([clang_format_path, '-i', output_file_copy_path])
    # cmd = ' '.join(['clang-format', '-i', output_file_copy_path])
    try:
        subprocess.run(cmd, shell=True, timeout=2, check=True)
        with open(output_file_copy_path, 'r', encoding='utf-8') as f:
            text = f.read()
            for i, comment in enumerate(comments):
                text = re.sub(f'COMMENT{str(i)}\n*',
                              f'COMMENT{str(i)}\n', text)
                lines = [line + '\n' for line in text.split('\n')]
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        print('clang-format failed.')
        with open(output_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

    lines = simplified_brace_complement_gstyle(lines)

    # Convert COMMENT+number back to original comments
    text = ''.join(lines)
    for i, comment in enumerate(comments):
        text = re.sub(f'COMMENT{str(i)}', comment, text)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(text)

    shutil.copyfile(output_file_path, output_file_copy_path)
    clang_format_path = 'clang-format.exe'
    cmd = ' '.join([clang_format_path, '-i', output_file_copy_path])
    # cmd = ' '.join(['clang-format', '-i', output_file_copy_path])
    try:
        subprocess.run(cmd, shell=True, timeout=2, check=True)
        shutil.copyfile(output_file_copy_path, output_file_path)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        print('clang-format failed.')


def parser_guided_repair(model: UniXcoder, tmp_dir: str, output_file_path: str) -> None:
    """
    The implementation of CRec's parser-guided repair process.

    Args:
        model: The UniXcoder model.
        tmp_dir: The path to the temporary directory.
        output_file_path: The path to the output file.
    """
    token_file_path = r'JavaLexer.tokens'
    repair_tool = RepairTool(model, token_file_path)
    rounds = 1
    print('Start repairing...')
    print(f'round {rounds}' + '*' * 50)
    istream = FileStream(output_file_path, encoding='utf-8')
    lexer = JavaLexer(istream)
    stream = CommonTokenStream(lexer)
    parser = JavaParser(stream)
    parser.removeErrorListeners()
    handle_error_listener = HandleErrorListener(
        repair_tool, output_file_path, tmp_dir)
    parser.addErrorListener(handle_error_listener)
    parser.compilationUnit()
    while parser._syntaxErrors != repair_tool.error_tolerance:
        rounds += 1
        print(f'round {rounds}' + '*' * 50)
        istream = FileStream(output_file_path, encoding='utf-8')
        lexer = JavaLexer(istream)
        stream = CommonTokenStream(lexer)
        parser = JavaParser(stream)
        parser.removeErrorListeners()
        parser.addErrorListener(handle_error_listener)
        parser.compilationUnit()


def monitor(proc: Process) -> None:
    """
    Monitor the repair process.

    Args:
        proc: The repair process.
    """
    time.sleep(30)
    if proc.is_alive():
        proc.terminate()


if __name__ == '__main__':
    model_path = "microsoft/unixcoder-base"
    input_dir = "input"
    output_dir = "output"
    failure_dir = "output/failures"
    tmp_dir = "tmp"

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    files = os.listdir(input_dir)
    for file in files:
        input_file_path = os.path.join(input_dir, file)
        output_file_path = os.path.join(output_dir, file)
        error_file_path = os.path.join(failure_dir, file)

        try:
            proc = Process(target=repair, args=(model_path, tmp_dir, input_file_path,
                           output_file_path, error_file_path,))
            thread = Thread(target=monitor, args=(proc,))
            proc.start()
            thread.start()
            proc.join()
        except Exception as e:
            error_info = traceback.format_exc()
            with open(error_file_path, 'a') as f:
                f.write(error_info)
    
    shutil.rmtree(tmp_dir)
