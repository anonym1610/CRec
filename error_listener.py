import os
import re
import shutil

from JavaLexer import JavaLexer
from JavaParser import JavaParser
from repair_tool import RepairTool

from antlr4 import *
from antlr4.error.ErrorListener import *


class ValidateErrorListener(ErrorListener):
    """
    An error listener used to validate a patch.

    Attributes:
        error_tolerance: The number of tolerable errors (that is, errors that failed to fix).
        error_line_index: The index of the error line to be patched (1-based index).
        offending_symbol_index: The index of the offending symbol in code.
        is_fixed: A flag indicating whether the error is fixed.
    """

    def __init__(self, error_tolerance: int, error_line_index: int, offending_symbol_index: int) -> None:
        super().__init__()
        self.error_tolerance = error_tolerance
        self.error_line_index = error_line_index
        self.offending_symbol_index = offending_symbol_index
        self.is_fixed = False

    def syntaxError(self, recognizer: Parser, offendingSymbol: Token, line: int, column: int, msg: str, e) -> None:
        """
        Override the syntaxError method to validate a patch.

        Args:
            recognizer: The parser object.
            offendingSymbol: The token that caused the error.
            line: The line number where the error occurs.
            column: The character position within that line where the error occurred.
            msg: The error message.
            e: The exception generated by the parser that led to the reporting of an error.
        """
        # Check if the error is fixed
        if recognizer._syntaxErrors == self.error_tolerance + 1:
            # A 'no viable alternative' error is considered fixed if the first 'out of tolerance' error is on a different line after patching.
            if re.match('no viable alternative', msg) is not None:
                if line != self.error_line_index:
                    self.is_fixed = True
            # For other error types, a fix is considered successful if the offending token index of the first 'out of tolerance' error is moved backward after patching.
            else:
                if offendingSymbol.tokenIndex > self.offending_symbol_index:
                    self.is_fixed = True


class HandleErrorListener(ErrorListener):
    """
    An error listener used to handle syntax errors.

    Attributes:
        repair_tool: The repair tool used to fix the error.
        fix_tries: The number of fix attempts for an error.
        max_fix_tries: The max number of fix attempts for an error.
        output_file_path: The path to the output file.
    """

    def __init__(self, repair_tool: RepairTool, output_file_path: str, tmp_dir: str) -> None:
        super().__init__()
        self.repair_tool = repair_tool
        self.fix_tries = 1
        self.max_fix_tries = 2
        self.output_file_path = output_file_path
        self.tmp_dir = tmp_dir

    def patch_code(self, output_file_path: str, patched_code_path: str, line_index: int, patch: str) -> None:
        """
        Patch the error line.

        Args:
            output_file_path: The path to the output file.
            line_index: The index of the line to be patched (1-based index).
            patch: The patch to be applied to the line.

        Returns:
            None
        """
        with open(output_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print('<before repair>' + '-' * 10)
        for i, line in enumerate(lines):
            if i == line_index - 1:
                print(f">>> {line.strip()}")
            else:
                print(line.strip())

        patched_line_index = line_index - 1
        if patched_line_index < len(lines):
            lines[patched_line_index] = patch.strip() + '\n'
        else:
            lines.append(patch.strip() + '\n')
            patched_line_index += 1

        print('<after repair>' + '-' * 11)
        for i, line in enumerate(lines):
            if i == patched_line_index:
                print(f">>> {line.strip()}")
            else:
                print(line.strip())

        with open(patched_code_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

    def validate_patch(self, patched_code_path: str, error_line_index: int, offending_symbol_index: int) -> bool:
        """
        Validate a patch.

        Args:
            patched_code_path: The path to the patched code file.
            error_line_index: The index of the error line to be patched (1-based index).

        Returns:
            True if the patch is valid; False otherwise.
        """
        istream = FileStream(patched_code_path, encoding='utf-8')
        lexer = JavaLexer(istream)
        stream = CommonTokenStream(lexer)
        parser = JavaParser(stream)
        parser.removeErrorListeners()
        validate_error_listener = ValidateErrorListener(
            self.repair_tool.error_tolerance, error_line_index, offending_symbol_index)
        parser.addErrorListener(validate_error_listener)
        parser.compilationUnit()
        return (parser._syntaxErrors == self.repair_tool.error_tolerance) or validate_error_listener.is_fixed

    def syntaxError(self, recognizer: Parser, offendingSymbol: Token, line: int, column: int, msg: str, e) -> None:
        """
        Override the syntaxError method to handle syntax errors.

        Args:
            recognizer: The parser object.
            offendingSymbol: The token that caused the error.
            line: The line number where the error occurs.
            column: The character position within that line where the error occurred.
            msg: The error message.
            e: The exception generated by the parser that led to the reporting of an error.
        """
        if recognizer._syntaxErrors <= self.repair_tool.error_tolerance:
            print("Failed to fix the following error:")
            print(f'line {line}:{column} {msg}')
            self.underlineError(recognizer, offendingSymbol, line, column)
        elif recognizer._syntaxErrors == self.repair_tool.error_tolerance + 1:
            # Give up and move to next error after 'max_fix_tries' unsuccessful fixes
            if self.fix_tries > self.max_fix_tries:
                print("Max fix attempts reached for the following error:")
                print(f'line {line}:{column} {msg}')
                self.underlineError(recognizer, offendingSymbol, line, column)
                self.repair_tool.error_tolerance = self.repair_tool.error_tolerance + 1
                self.fix_tries = 1
                return
            # Use all templates to fix the error for the last try
            elif self.fix_tries == self.max_fix_tries:
                print("Final attempt to fix the following error:")
                print(f'line {line}:{column} {msg}')
                self.underlineError(recognizer, offendingSymbol, line, column)
                patch_candidates = self.repair_tool.repair_no_viable_alt(
                    recognizer, offendingSymbol, line)
            else:
                print(f"Attempt {self.fix_tries} to fix the following error:")
                print(f'line {line}:{column} {msg}')
                self.underlineError(recognizer, offendingSymbol, line, column)
                if re.match('missing', msg) is not None:
                    patch_candidates = self.repair_tool.repair_missing_token(
                        recognizer, line, column)
                elif re.match('extraneous input', msg) is not None:
                    patch_candidates = self.repair_tool.repair_extraneous_input(
                        recognizer, offendingSymbol, line, column)
                elif re.match('mismatched input', msg) is not None:
                    patch_candidates = self.repair_tool.repair_mismatched_input(
                        recognizer, offendingSymbol, line)
                    if column == 0 and line > 1:
                        patch_candidates.extend(self.repair_tool.repair_mismatched_input(
                            recognizer, offendingSymbol, line - 1))
                        patch_candidates.sort(
                            key=lambda x: x.score, reverse=True)
                # Include the 'no viable alternative' error and other unknown error types here
                else:
                    patch_candidates = self.repair_tool.repair_no_viable_alt(
                        recognizer, offendingSymbol, line)
                    self.fix_tries = self.max_fix_tries
            # Validate patches
            patch_success_flags = []
            for i, patch_candidate in enumerate(patch_candidates):
                # Use the potential patch to fix the error, and generate the patched code
                print(f"Patch {i}: {patch_candidate.patch}")
                patched_code_path = os.path.join(
                    self.tmp_dir, f'patched_code_{i}.java')
                self.patch_code(self.output_file_path, patched_code_path,
                                patch_candidate.line_index, patch_candidate.patch)
                # Check if the error is fixed
                isfixed = self.validate_patch(
                    patched_code_path, patch_candidate.line_index, offendingSymbol.tokenIndex)
                patch_success_flags.append(isfixed)
            # Choose the successful patch with the highest score
            if any(patch_success_flags):
                best_patch_index = patch_success_flags.index(True)
                shutil.copyfile(os.path.join(
                    self.tmp_dir, f'patched_code_{best_patch_index}.java'), self.output_file_path)
                self.fix_tries = 1
            else:
                self.fix_tries += 1

    def underlineError(self, recognizer: Parser, offendingToken: Token, line_index: int, column_index: int) -> None:
        """
        Underline the error in the code.

        Args:
            recognizer: The parser object.
            offendingToken: The token that caused the error.
            line_index: The index of the line where the error occurs.
            column_index: The index of the column where the error occurs.
        """
        tokens = recognizer.getInputStream()
        lines = tokens.getText().split('\n')
        errorLine = lines[line_index - 1]
        print(errorLine)
        for i in range(column_index):
            print(' ', sep='', end='')
        start = offendingToken.start
        stop = offendingToken.stop
        if start >= 0 and stop >= 0:
            for i in range(start, stop + 1):
                print('^', sep='', end='')
        print()
