import re
from antlr4 import *
from antlr4.error.ErrorListener import *

from RepairTool import RepairTool


class HandleErrorListener(ErrorListener):

    def __init__(self, repair_tool: RepairTool) -> None:

        super().__init__()
        self.stuck_line_range = [-1, 0]  # group the erroneous code line and the preceding line into a range, within which the repair may be stuck
        self.same_range_stuck_times = 0
        self.same_range_stuck_tolerance = 5  # the maximum number of fix attempts within "stuck_line_range"
        self.latest_stuck_position = (0, -1)
        self.second_latest_stuck_position = (0, -1)
        self.repair_tool = repair_tool

    def syntaxError(self, recognizer: Parser, offendingSymbol: Token, line: int, column: int, msg: str, e):

        if recognizer._syntaxErrors == self.repair_tool.error_tolerance + 1:
            if not self.is_stuck(recognizer, offendingSymbol, line):
                # print("Trying to repair following error:\n")
                # self.underlineError(recognizer, offendingSymbol, line, column)
                if re.match('mismatched input', msg) is not None:
                    # print(msg)
                    self.repair_tool.repair_mismatched_input(recognizer, offendingSymbol, line, column)
                elif re.match('missing', msg) is not None:
                    # print(msg)
                    self.repair_tool.repair_missing_token(recognizer, line, column)
                elif re.match('no viable alternative', msg) is not None:
                    # print(msg)
                    self.repair_tool.repair_no_viable_alt(recognizer, offendingSymbol, line, column)
                elif re.match('extraneous input', msg) is not None:
                    # print(msg)
                    self.repair_tool.repair_extraneous_input(recognizer, offendingSymbol, line, column)
                else:
                    self.repair_tool.error_tolerance = self.repair_tool.error_tolerance + 1
            else:
                # print('stuck!')
                self.repair_tool.error_tolerance = self.repair_tool.error_tolerance + 1

    def underlineError(self, recognizer: Parser, offendingToken: Token, line_index: int, column_index: int):

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

    def is_stuck(self, recognizer: Parser, offendingSymbol: Token, line_index: int) -> bool:

        allTokens = recognizer.getInputStream().tokens
        errorLineTokens = []
        errorTokenPtr = 0
        for token in allTokens:
            if token.line == line_index and token.type not in [125, 126, 127]:
                if token.tokenIndex == offendingSymbol.tokenIndex:
                    errorTokenPtr = len(errorLineTokens)
                errorLineTokens.append(token)
            if token.line > line_index:
                break
        if line_index == self.second_latest_stuck_position[0] and errorTokenPtr == self.second_latest_stuck_position[1]:
            self.same_range_stuck_times = self.same_range_stuck_times + 1
            if self.same_range_stuck_times > self.same_range_stuck_tolerance:
                return True
        else:
            self.second_latest_stuck_position = self.latest_stuck_position
            self.latest_stuck_position = (line_index, errorTokenPtr)

            if line_index in self.stuck_line_range:
                self.same_range_stuck_times = self.same_range_stuck_times + 1
                if self.same_range_stuck_times > self.same_range_stuck_tolerance:
                    return True
            else:
                self.stuck_line_range = [line_index - 1, line_index]
                self.same_range_stuck_times = 0

            return False
