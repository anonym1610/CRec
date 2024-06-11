import nltk
import torch
import re

from antlr4 import *
from antlr4.error.ErrorListener import *
import distance

from JavaLexer import JavaLexer
from template import *
from unixcoder import UniXcoder


class PatchCandidate(object):
    """
    A patch candidate.

    Attributes:
        patch: The patch to replace the buggy line.
        score: The score of the patch.
        line_index: The line index where the patch will be applied (1-based index).
    """

    def __init__(self, patch: str, score: float, line_index: int) -> None:
        self.patch = patch
        self.score = score
        self.line_index = line_index


class RepairTool(object):
    """
    A repair tool used to fix errors.

    Attributes:
        model: The UniXcoder model.
        error_tolerance: The number of tolerable errors (that is, errors that failed to fix).
        token_info: A dictionary mapping token types to their names.
    """

    def __init__(self, model: UniXcoder, token_file_path: str) -> None:
        self.model = model
        self.error_tolerance = 0
        self.token_info = self.get_token_info(token_file_path)

    def get_token_info(self, token_file_path: str) -> dict:
        """
        Read the token file and return a dictionary mapping token types to their names.

        Args:
            tokens_file_path: The path to the token file.

        Returns:
            dict: A dictionary where the keys are token types (int) and the values are lists of token names (str).
        """
        token_info = {}

        with open(token_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if not line.isspace():
                    assign_symbol_index = line.rfind('=')
                    token_type = int(line[assign_symbol_index + 1:].rstrip())
                    name = line[: assign_symbol_index]
                    if token_type not in token_info:
                        token_info[token_type] = [name]
                    else:
                        token_info[token_type].append(name[1: -1])

        return token_info

    def get_context(self, code_without_comments: str, error_line_index: int, mask_template: str, max_length=512) -> list:
        """
        Adds context lines to the mask template of the error line.

        Args:
            code_without_comments: The code snippet without comments.
            error_line_index: The index of the line containing the error (1-based index).
            mask_template: The mask template for the error line.
            max_length: The maximum length of the context in terms of tokens.

        Returns:
            list: The context of the error line and the mask template, organized as [before_lines, mask_template, after_lines].
        """
        lines = [line for line in code_without_comments.split('\n')]

        # Pre-calculate token lengths for each line.
        line_token_lengths = [len(self.model.tokenize(
            line, mode='<encoder-decoder>')) for line in lines]
        token_count = len(self.model.tokenize(
            mask_template, mode='<encoder-decoder>'))

        # Variables to control loop
        forward_added = True
        backward_added = True

        # The number of lines considered before and after the mask index
        before_lines = []
        after_lines = []

        error_line_index = error_line_index - 1  # 0-based index

        while token_count <= max_length and (forward_added or backward_added):
            # Try adding from before the mask line
            forward_added = False
            if error_line_index - len(before_lines) - 1 >= 0:
                next_line_index = error_line_index - len(before_lines) - 1
                potential_new_count = token_count + \
                    line_token_lengths[next_line_index]
                if potential_new_count <= max_length:
                    before_lines.insert(0, lines[next_line_index])
                    token_count += line_token_lengths[next_line_index]
                    forward_added = True

            # Try adding from after the mask line
            backward_added = False
            if error_line_index + len(after_lines) + 1 < len(lines):
                next_line_index = error_line_index + len(after_lines) + 1
                potential_new_count = token_count + \
                    line_token_lengths[next_line_index]
                if potential_new_count <= max_length:
                    after_lines.append(lines[next_line_index])
                    token_count += line_token_lengths[next_line_index]
                    backward_added = True

        return ['\n'.join(before_lines), mask_template, '\n'.join(after_lines)]

    def gen_patch_candidates(self, context: list, line_index: int, beam_size=5, context_max_length=512, pred_max_length=128) -> list:
        """
        Generate patch candidates for a given context.

        Args:
            context: The context of the error line and the mask template, organized as [before_lines, mask_template, after_lines].
            line_index: The line index where the patch will be applied (1-based index).
            beam_size: The beam size for beam search.
            context_max_length: The maximum length of the context.
            pred_max_length: The maximum length of the generated patch.

        Returns:
            list: A list of patch candidates.
        """
        masked_code = '\n'.join(context)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokens_ids = self.model.tokenize(
            [masked_code], mode='<encoder-decoder>', max_length=context_max_length)
        source_ids = torch.tensor(tokens_ids).to(device)
        prediction_ids, prediction_scores = self.model.generate(
            source_ids, decoder_only=False, beam_size=beam_size, max_length=pred_max_length)
        predictions = self.model.decode(prediction_ids)
        patches = [x.replace("<mask0>", "").strip() for x in predictions[0]]

        for i in range(len(patches)):
            patches[i] = context[1].replace("<mask0>", patches[i])
            patches[i] = re.sub(r'// buggy line:.*\n', '', patches[i])

        return [PatchCandidate(patch, score, line_index) for patch, score in zip(patches, prediction_scores)]

    def rank_patch_candidates(self, patch_candidates: list, error_line: str, k=5) -> list:
        """
        Rank patches based on their scores and the token-based NLD values between them and the error line.

        All ranked patches target the same line.

        Args:
            patch_candidates: A list of patch candidates.
            error_line: The error line.
            k: The number of top-k ranked patch candidates to return.

        Returns:
            list: A list of top-k ranked patch candidates.
        """
        if len(patch_candidates) != 0:
            # Remove duplicate patches, and if two identical patches have different scores, retain the one with the higher score
            patch_candidate_dict = {}
            for patch_candidate in patch_candidates:
                patch_candidate_dict[patch_candidate.patch] = max(
                    patch_candidate.score, patch_candidate_dict.get(patch_candidate.patch, -float('inf')))
            line_index = patch_candidates[0].line_index
            patch_candidates = [PatchCandidate(
                patch, score, line_index) for patch, score in patch_candidate_dict.items()]

            # Softmax the scores
            scores = torch.softmax(torch.tensor(
                [patch_candidate.score for patch_candidate in patch_candidates]), dim=-1)
            scores = scores.cpu().tolist()

            print("Before ranking:")
            for patch, score in list(zip([patch_candidate.patch for patch_candidate in patch_candidates], scores)):
                print(patch.rstrip(), score, sep=' score: ')

            # Remove additional whitespace in patches
            # The error line has already removed additional whitespace
            patches = [' '.join(patch_candidate.patch.strip().split())
                       for patch_candidate in patch_candidates]

            # Calculate token-based NLD between the patch candidates and the error line
            error_line_tokens = nltk.word_tokenize(error_line)
            tokenized_patches = [nltk.word_tokenize(
                patch) for patch in patches]

            token_based_ld_values = [distance.levenshtein(
                error_line_tokens, patch_tokens) for patch_tokens in tokenized_patches]
            token_based_nld_values = []
            for i, ld in enumerate(token_based_ld_values):
                token_based_nld = 1 - ld / \
                    max(len(error_line_tokens), len(tokenized_patches[i]))
                token_based_nld_values.append(token_based_nld)

            # rank patches and return top-k
            score_weight = 0.5
            nld_weight = 0.5

            for i, patch_candidate in enumerate(patch_candidates):
                score = scores[i]
                nld = token_based_nld_values[i]
                patch_candidates[i].score = score_weight * \
                    score + nld_weight * nld

            patch_candidates.sort(key=lambda x: x.score, reverse=True)

        print("After ranking:")
        for patch_candidate in patch_candidates:
            print(patch_candidate.patch.rstrip(),
                  patch_candidate.score, sep=' score: ')

        return patch_candidates[:k]

    def repair_missing_token(self, recognizer: Parser, line_index: int, column_index: int) -> list:
        """
        Generate a unique patch candidate for a 'missing token' error.

        Args:
            recognizer: The parser object.
            line_index: The line index where the missing token should be inserted (1-based index).
            column_index: The column index where the offending symbol is located.

        Returns:
            list: The unique patch candidate.
        """
        print('calling repair_missing_token...')

        # Get the missing token
        missing_token_type = recognizer._errHandler.getMissingSymbol(
            recognizer).type
        missing_token = self.token_info[missing_token_type][-1]

        # Insert the missing token
        token_stream = recognizer.getInputStream()
        lines = token_stream.getText().split('\n')
        error_line = lines[line_index - 1]
        patch = ' '.join([error_line[:column_index],
                         missing_token, error_line[column_index:]])

        return [PatchCandidate(patch, 1.0, line_index)]

    def repair_extraneous_input(self, recognizer: Parser, offending_symbol: Token, line_index: int, column_index: int) -> list:
        """
        Generate a unique patch candidate for an 'extraneous input' error.

        Args:
            recognizer: The parser object.
            offending_symbol: The token that caused the 'extraneous input' error.
            line_index: The line index where the extraneous token should be removed (1-based index).
            column_index: The column index where the offending symbol is located.

        Returns:
            list: The unique patch candidate.
        """
        print('calling repair_extraneous_input...')

        token_stream = recognizer.getInputStream()
        lines = token_stream.getText().split('\n')
        error_line = lines[line_index - 1]
        if offending_symbol.text == '<EOF>':
            return [PatchCandidate(error_line.rstrip() + '\n}', 1.0, line_index)]
        else:
            start = offending_symbol.start
            stop = offending_symbol.stop
            patch = error_line[:column_index] + \
                error_line[column_index + stop - start + 1:]

            return [PatchCandidate(patch, 1.0, line_index)]

    def repair_mismatched_input(self, recognizer: Parser, offending_symbol: Token, line_index: int) -> list:
        """
        Generate patch candidates for a 'mismatched input' error.

        Args:
            recognizer: The parser object.
            offending_symbol: The token that caused the 'mismatched input' error.
            line_index: The index of the error line (1-based index).
            column_index: The column index where the offending symbol is located.

        Returns:
            List: The top-k ranked patch candidates.
        """
        print('calling repair_mismatched_input...')

        all_tokens = recognizer.getInputStream().tokens
        error_line_tokens = []
        offending_token_index = -1

        # Get tokens of the error line
        for token in all_tokens:
            if token.line < line_index:
                continue
            # -1, 125, 126 and 127 are the token types for EOF, whitespaces, multi-line comment and single line comment respectively
            if token.line == line_index and token.type not in [-1, 125, 126, 127]:
                if token.tokenIndex == offending_symbol.tokenIndex:
                    offending_token_index = len(error_line_tokens)
                error_line_tokens.append(token)
                continue
            if token.line > line_index:
                break

        if offending_token_index == -1:
            offending_token_index = len(error_line_tokens)

        # Get patch candidates
        code_without_comments = ''
        for token in all_tokens:
            if token.type == -1:
                break
            elif token.type in [126, 127]:
                line_count = token.text.count('\n')
                code_without_comments += '\n' * line_count
            else:
                code_without_comments += token.text

        mask_templates = set()
        mask_templates.add(gen_middle_mask_template(
            error_line_tokens, offending_token_index))
        mask_templates.add(gen_after_mask_template(
            error_line_tokens, offending_token_index))

        print('Mask templates:')
        for i, mask_template in enumerate(mask_templates):
            print(f'template {i}:', mask_template, sep='\n')

        mask_template_contexts = []
        for mask_template in mask_templates:
            mask_template_context = self.get_context(
                code_without_comments, line_index, mask_template)
            mask_template_contexts.append(mask_template_context)

        patch_candidates = []
        for mask_template_context in mask_template_contexts:
            patch_candidates.extend(self.gen_patch_candidates(
                mask_template_context, line_index))

        print('Patch candidates:')
        for patch_candidate in patch_candidates:
            print(patch_candidate.patch.rstrip(),
                  patch_candidate.score, sep=' score: ')

        # Get the expected token types
        expected_token_type_ranges = recognizer.getExpectedTokens().intervals
        expected_token_types = []

        for token_type_ranges in expected_token_type_ranges:
            for token_type in token_type_ranges:
                expected_token_types.append(token_type)

        print('Expected tokens:', [self.token_info[token_type][-1]
              for token_type in expected_token_types])

        # Call lexer to tokenize patches
        tokenized_patches = []
        for patch_candidate in patch_candidates:
            lexer = JavaLexer(InputStream(patch_candidate.patch))
            stream = CommonTokenStream(lexer)
            stream.fill()  # Force the lexer to tokenize the input
            patch_tokens = stream.tokens
            tokenized_patches.append(patch_tokens)

        # Retain the patches whose first non-whitespace token is expected
        retained_patch_candidates = []
        for i, patch_tokens in enumerate(tokenized_patches):
            patch_tokens = [
                token for token in patch_tokens if token.type not in [-1, 125, 126, 127]]
            if len(patch_tokens) != 0 and offending_token_index < len(patch_tokens):
                if patch_tokens[offending_token_index].type in expected_token_types:
                    retained_patch_candidates.append(patch_candidates[i])

        print('Retained patches:')
        for patch_candidate in retained_patch_candidates:
            print(patch_candidate.patch.rstrip(),
                  patch_candidate.score, sep=' score: ')

        # Rank the retained patches
        error_line = ' '.join(token.text for token in error_line_tokens)
        top_k_patch_candidates = self.rank_patch_candidates(
            retained_patch_candidates, error_line)

        return top_k_patch_candidates

    def repair_no_viable_alt(self, recognizer: Parser, offending_symbol: Token, line_index: int) -> list:
        """
        Generate patch candidates for a 'no viable alternative' error.

        Args:
            recognizer: The parser object.
            offending_symbol: The token that caused the 'no viable alternative' error.
            line_index: The index of the error line (1-based index).

        Returns:
            List: The top-ranked patch candidates.
        """
        print('calling repair_no_viable_alt...')

        all_tokens = recognizer.getInputStream().tokens
        error_line_tokens = []

        # Get tokens of the error line
        for token in all_tokens:
            if token.line < line_index:
                continue
            # -1, 125, 126 and 127 are the token types for EOF, whitespaces, multi-line comment and single line comment respectively
            if token.line == line_index and token.type not in [-1, 125, 126, 127]:
                if token.tokenIndex == offending_symbol.tokenIndex:
                    offending_token_index = len(error_line_tokens)
                error_line_tokens.append(token)
                continue
            if token.line > line_index:
                break

        # Generate patch candidates with all templates
        code_without_comments = ''
        for token in all_tokens:
            if token.type == -1:
                break
            elif token.type in [126, 127]:
                line_count = token.text.count('\n')
                code_without_comments += '\n' * line_count
            else:
                code_without_comments += token.text

        mask_templates = set()
        mask_templates.add(gen_middle_mask_template(
            error_line_tokens, offending_token_index))
        mask_templates.add(gen_after_mask_template(
            error_line_tokens, offending_token_index))
        mask_templates.add(gen_before_mask_template(
            error_line_tokens, offending_token_index))
        mask_templates.add(gen_start_mask_template(error_line_tokens))
        mask_templates.add(gen_end_mask_template(error_line_tokens))
        mask_templates.add(gen_line_mask_template(error_line_tokens))

        print('Mask templates:')
        for i, mask_template in enumerate(mask_templates):
            print(f'template {i}:', mask_template, sep='\n')

        mask_template_contexts = []
        for mask_template in mask_templates:
            mask_template_context = self.get_context(
                code_without_comments, line_index, mask_template)
            mask_template_contexts.append(mask_template_context)

        patch_candidates = []
        for mask_template_context in mask_template_contexts:
            patch_candidates.extend(self.gen_patch_candidates(
                mask_template_context, line_index))

        print('Patch candidates:')
        for patch_candidate in patch_candidates:
            print(patch_candidate.patch.rstrip(),
                  patch_candidate.score, sep=' score: ')

        error_line = ' '.join(token.text for token in error_line_tokens)
        top_k_patch_candidates = self.rank_patch_candidates(
            patch_candidates, error_line)

        return top_k_patch_candidates

    def repair_mismatched_input_new(self, recognizer: Parser, offending_symbol: Token, line_index: int) -> list:
        """
        "Generate patch candidates for a 'mismatched input' error, without the constraint that the first non-whitespace token of the patch must be an expected token."

        Args:
            recognizer: The parser object.
            offending_symbol: The token that caused the 'mismatched input' error.
            line_index: The index of the error line (1-based index).
            column_index: The column index where the offending symbol is located.

        Returns:
            List: The top-k ranked patch candidates.
        """
        print('calling repair_mismatched_input...')

        all_tokens = recognizer.getInputStream().tokens
        error_line_tokens = []
        offending_token_index = -1

        # Get tokens of the error line
        for token in all_tokens:
            if token.line < line_index:
                continue
            # -1, 125, 126 and 127 are the token types for EOF, whitespaces, multi-line comment and single line comment respectively
            if token.line == line_index and token.type not in [-1, 125, 126, 127]:
                if token.tokenIndex == offending_symbol.tokenIndex:
                    offending_token_index = len(error_line_tokens)
                error_line_tokens.append(token)
                continue
            if token.line > line_index:
                break

        if offending_token_index == -1:
            offending_token_index = len(error_line_tokens)

        # Get patch candidates
        code_without_comments = ''
        for token in all_tokens:
            if token.type == -1:
                break
            elif token.type in [126, 127]:
                line_count = token.text.count('\n')
                code_without_comments += '\n' * line_count
            else:
                code_without_comments += token.text

        mask_templates = set()
        mask_templates.add(gen_middle_mask_template(
            error_line_tokens, offending_token_index))
        mask_templates.add(gen_after_mask_template(
            error_line_tokens, offending_token_index))

        print('Mask templates:')
        for i, mask_template in enumerate(mask_templates):
            print(f'template {i}:', mask_template, sep='\n')

        mask_template_contexts = []
        for mask_template in mask_templates:
            mask_template_context = self.get_context(
                code_without_comments, line_index, mask_template)
            mask_template_contexts.append(mask_template_context)

        patch_candidates = []
        for mask_template_context in mask_template_contexts:
            patch_candidates.extend(self.gen_patch_candidates(
                mask_template_context, line_index))

        print('Patch candidates:')
        for patch_candidate in patch_candidates:
            print(patch_candidate.patch.rstrip(),
                  patch_candidate.score, sep=' score: ')

        error_line = ' '.join(token.text for token in error_line_tokens)
        top_k_patch_candidates = self.rank_patch_candidates(
            patch_candidates, error_line)

        return top_k_patch_candidates
