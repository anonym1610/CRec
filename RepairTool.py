import kenlm
import nltk
import random
import string
from antlr4 import *
from antlr4.error.ErrorListener import *


class RepairTool:

    def __init__(self, tokens_file_path: str, token_model_file_path: str, output_file_path: str) -> None:
        self.error_tolerance = 0
        self.latest_repaired_line = ''
        self.second_latest_repaired_line = ''
        self.tokens_info = self.load_tokens_info(tokens_file_path)
        self.line_token_model = kenlm.LanguageModel(token_model_file_path)
        self.output_file_path = output_file_path

    def load_tokens_info(self, tokens_file_path: str) -> list:

        tokens_info = []

        with open(tokens_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if not line.isspace():
                    equal_sign_index = line.rfind('=')
                    type = int(line[equal_sign_index + 1:].rstrip())
                    name = line[: equal_sign_index]
                    if type > len(tokens_info):
                        tokens_info.append([name])
                    else:
                        tokens_info[type - 1].append(name[1: -1])
        tokens_info.insert(0, ['PLACEHOLDER'])

        return tokens_info

    def estimate_sentence(self, sentence: str, model_choice: int) -> float:
        token_stream = nltk.word_tokenize(sentence)
        sentence = ''
        for token in token_stream:
            sentence = sentence + token + ' '
        if model_choice == 0:
            return self.line_basic_model.perplexity(sentence)
        else:
            return self.line_token_model.perplexity(sentence)

    def output(self, output_file_path: str, line_index: int, repaired_line: str):
        with open(output_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if line_index - 1 < len(lines):
            lines[line_index - 1] = repaired_line.strip() + '\n'
        else:
            lines.append(repaired_line.strip() + '\n')
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

    def repair_mismatched_input(self, recognizer: Parser, offending_symbol: Token, line_index: int, column_index: int):

        # print('calling repair_mismatched_input...')

        all_tokens = recognizer.getInputStream().tokens
        tokens_of_error_line = []
        tokens_of_last_line = []  
        error_token_index = 0

        for token in all_tokens:
            if line_index != 1 and token.line == line_index - 1 and token.type not in [125, 126, 127]:
                tokens_of_last_line.append(token)
            if token.line == line_index and token.type not in [125, 126, 127]:
                if token.tokenIndex == offending_symbol.tokenIndex:
                    error_token_index = len(tokens_of_error_line)
                tokens_of_error_line.append(token)
            elif token.line > line_index:
                break

        type_ranges_of_excepted_tokens = recognizer.getExpectedTokens().intervals  
        excepted_tokens_types = []  

        for type_range in type_ranges_of_excepted_tokens:
            for type in type_range:
                excepted_tokens_types.append(type)

        token_stream = recognizer.getInputStream()
        lines = token_stream.getText().split('\n')
        error_line = lines[line_index - 1]

        if column_index == 0 and line_index != 1:  

            last_line = lines[line_index - 2]
            candidate_set_last = []

            token_symbols_of_last_line = []
            for token in tokens_of_last_line:
                token_symbols_of_last_line.append(self.tokens_info[token.type][0])

            for type in excepted_tokens_types:
                candidate = ' '.join(token_symbols_of_last_line) + ' ' + self.tokens_info[type][0]
                candidate_set_last.append([candidate, type])

            best_ppl_last = float('inf')
            type_of_best_candidate_last = 1

            for candidate in candidate_set_last:
                ppl = self.estimate_sentence(candidate[0], 1)
                if ppl < best_ppl_last:
                    best_ppl_last = ppl
                    type_of_best_candidate_last = candidate[1]
            last_line_ppl = self.estimate_sentence(' '.join(token_symbols_of_last_line), 1)
            if best_ppl_last < last_line_ppl:
                if len(self.tokens_info[type_of_best_candidate_last]) == 2:
                    senetnce = last_line.strip() + ' ' + self.tokens_info[type_of_best_candidate_last][1]
                else:
                    senetnce = last_line.strip() + ' ' + self.tokens_info[type_of_best_candidate_last][0]
                self.output(self.output_file_path, line_index - 1, senetnce)
                return

        candidate_set = []  

        token_symbols_of_error_line = []
        for token in tokens_of_error_line:
            token_symbols_of_error_line.append(self.tokens_info[token.type][0])

        for type in excepted_tokens_types:
            copy = token_symbols_of_error_line.copy()
            copy[error_token_index] = self.tokens_info[type][0]
            senetnce = ' '.join(copy)
            candidate_set.append([senetnce, type, 0])
            copy = token_symbols_of_error_line.copy()
            copy.insert(error_token_index, self.tokens_info[type][0])
            senetnce = ' '.join(copy)
            candidate_set.append([senetnce, type, 1])

        best_ppl = float('inf')
        best_candidate = ''
        for candidate in candidate_set:
            ppl = self.estimate_sentence(candidate[0], 1)
            if ppl < best_ppl:
                best_ppl = ppl
                best_candidate = candidate

        start = offending_symbol.start
        stop = offending_symbol.stop

        if len(self.tokens_info[best_candidate[1]]) == 2:
            patch_token = self.tokens_info[best_candidate[1]][1]
        else:
            
            if best_candidate[1] == 128:
                patch_token = patch_token = 'IDENTIFIER_' + ''.join(
                    random.choice(string.ascii_letters + string.digits) for _ in range(3))
            elif best_candidate[1] == 67:
                patch_token = '1'
            elif best_candidate[1] == 68:
                patch_token = '0x1'
            elif best_candidate[1] == 69:
                patch_token = '01'
            elif best_candidate[1] == 70:
                patch_token = '0b1'
            elif best_candidate[1] == 71:
                patch_token = '1.0'
            elif best_candidate[1] == 72:
                patch_token = '0x1.0p-3'
            elif best_candidate[1] == 73:
                patch_token = 'true'
            elif best_candidate[1] == 74:
                patch_token = '\'a\''
            elif best_candidate[1] == 75:
                patch_token = '\"hello\"'
            elif best_candidate[1] == 76:
                patch_token = '\"\"\"hello\"\"\"'
            else:
                patch_token = self.tokens_info[best_candidate[1]][0]

        error_line = lines[line_index - 1]

        if best_candidate[2] == 1:
            repaired_line = error_line[: column_index] + ' ' + patch_token + ' ' + error_line[column_index:]
        else:
            repaired_line = error_line[: column_index] + ' ' + patch_token + ' ' + error_line[
                                                                                   column_index + stop - start + 1:]
        self.output(self.output_file_path, line_index, repaired_line.strip())

    def repair_missing_token(self, recognizer: Parser, line_index: int, column_index: int):

        # print('calling repair_missing_token...')
        missing_token_type = recognizer._errHandler.getMissingSymbol(recognizer).type
        if len(self.tokens_info[missing_token_type]) == 2:
            missing_token = self.tokens_info[missing_token_type][1]
        else:
            missing_token = self.tokens_info[missing_token_type][0]

        token_stream = recognizer.getInputStream()
        lines = token_stream.getText().split('\n')
        error_line = lines[line_index - 1]
        repaired_line = error_line[: column_index] + ' ' + missing_token + ' ' + error_line[column_index:]
        self.output(self.output_file_path, line_index, repaired_line)

    def repair_no_viable_alt(self, recognizer: Parser, offending_symbol: Token, line_index: int, column_index: int):

        # print('calling repair_no_viable_alt...')

        all_tokens = recognizer.getInputStream().tokens
        tokens_of_error_line = []
        error_token_index = 0

        for token in all_tokens:
            if token.line == line_index and token.type != 125:
                if token.tokenIndex == offending_symbol.tokenIndex:
                    error_token_index = len(tokens_of_error_line)
                tokens_of_error_line.append(token)
            elif token.line > line_index:
                break

        candidate_set = []  

        for i, tokenInfo in enumerate(self.tokens_info):
            if not (i == offending_symbol.type or i in [125, 126, 127]):
                candidate = ''
                for j, token in enumerate(tokens_of_error_line):
                    if j == error_token_index:
                        candidate = candidate + tokenInfo[0] + ' '
                    else:
                        candidate = candidate + self.tokens_info[token.type][0] + ' '
                candidate_set.append([candidate, i, 0])

        for i, tokenInfo in enumerate(self.tokens_info):
            if not (i == offending_symbol.type or i in [125, 126, 127]):
                candidate = ''
                for j, token in enumerate(tokens_of_error_line):
                    if j == error_token_index:
                        candidate = candidate + tokenInfo[0] + ' '
                    candidate = candidate + self.tokens_info[token.type][0] + ' '
                candidate_set.append([candidate, i, 1])

        tokens_of_error_line.pop(error_token_index)
        candidate = ''
        for token in tokens_of_error_line:
            candidate = candidate + self.tokens_info[token.type][0] + ' '
        candidate_set.append([candidate, i, 2])

        best_ppl = float('inf')
        best_candidate = None
        for candidate in candidate_set:
            ppl = self.estimate_sentence(candidate[0], 1)
            if ppl < best_ppl:
                best_ppl = ppl
                best_candidate = candidate

        token_stream = recognizer.getInputStream()
        lines = token_stream.getText().split('\n')
        errorLine = lines[line_index - 1]
        start = offending_symbol.start
        stop = offending_symbol.stop

        if len(self.tokens_info[best_candidate[1]]) == 2:
            patch_token = self.tokens_info[best_candidate[1]][1]
        else:
            
            if best_candidate[1] == 128:
                patch_token = 'IDENTIFIER_' + ''.join(
                    random.choice(string.ascii_letters + string.digits) for _ in range(3))
            elif best_candidate[1] == 67:
                patch_token = '1'
            elif best_candidate[1] == 68:
                patch_token = '0x1'
            elif best_candidate[1] == 69:
                patch_token = '01'
            elif best_candidate[1] == 70:
                patch_token = '0b1'
            elif best_candidate[1] == 71:
                patch_token = '1.0'
            elif best_candidate[1] == 72:
                patch_token = '0x1.0p-3'
            elif best_candidate[1] == 73:
                patch_token = 'true'
            elif best_candidate[1] == 74:
                patch_token = '\'a\''
            elif best_candidate[1] == 75:
                patch_token = '\"hello\"'
            elif best_candidate[1] == 76:
                patch_token = '\"\"\"hello\"\"\"'
            else:
                patch_token = self.tokens_info[best_candidate[1]][0]

        if best_candidate[2] == 0:
            repaired_line = errorLine[: column_index] + ' ' + patch_token + ' ' + errorLine[
                                                                                  column_index + stop - start + 1:]
        elif best_candidate[2] == 1:
            repaired_line = errorLine[: column_index] + ' ' + patch_token + ' ' + errorLine[column_index:]
        else:
            repaired_line = errorLine[: column_index] + errorLine[column_index + stop - start + 1:]
        self.output(self.output_file_path, line_index, repaired_line)

    def repair_extraneous_input(self, recognizer: Parser, offending_symbol: Token, line_index: int, column_index: int):

        # print('calling repair_extraneous_input...')

        if offending_symbol.text == '<EOF>':
            # print('Hello, this is <eof>.')
            self.error_tolerance = self.error_tolerance + 1
        else:
            token_stream = recognizer.getInputStream()
            lines = token_stream.getText().split('\n')
            error_line = lines[line_index - 1]
            start = offending_symbol.start
            stop = offending_symbol.stop
            repaired_line = error_line[: column_index] + error_line[column_index + stop - start + 1:]
            self.output(self.output_file_path, line_index, repaired_line)
