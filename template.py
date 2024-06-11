def gen_middle_mask_template(token_line: list, mask_index: int) -> str:
    """
    Generate a template like this: before_code <mask0> after_code

    Args:
        token_line: The error line composed of tokens.
        mask_index: The index of the offending token to be masked.

    Returns:
        A string representing the template.
    """
    before_token = token_line[:mask_index]
    if mask_index > len(token_line) - 1:
        after_token = []
    else:
        after_token = token_line[mask_index + 1:]
    before_code = ' '.join(token.text for token in before_token)
    after_code = ' '.join(token.text for token in after_token)
    comment = '// buggy line: ' + ' '.join(token.text for token in token_line)
    # template = f'{before_code} <mask0> {after_code} {comment}\n'
    template = f'{comment}\n{before_code} <mask0> {after_code}\n'

    return template


def gen_after_mask_template(token_line: list, mask_index: int) -> str:
    """
    Generate a template like this: before_code <mask0>

    Args:
        token_line: The error line composed of tokens.
        mask_index: The index of the offending token to be masked.

    Returns:
        A string representing the template.
    """
    before_token = token_line[:mask_index]
    before_code = ' '.join(token.text for token in before_token)
    comment = '// buggy line: ' + ' '.join(token.text for token in token_line)
    # template = f'{before_code} <mask0> {comment}\n'
    template = f'{comment}\n{before_code} <mask0> \n'

    return template


def gen_before_mask_template(token_line: list, mask_index: int) -> str:
    """
    Generate a template like this: <mask0> after_code

    Args:
        token_line: The error line composed of tokens.
        mask_index: The index of the offending token to be masked.

    Returns:
        A string representing the template.
    """
    if mask_index > len(token_line) - 1:
        after_token = []
    else:
        after_token = token_line[mask_index + 1:]
    after_code = ' '.join(token.text for token in after_token)
    comment = '// buggy line: ' + ' '.join(token.text for token in token_line)
    # template = f'<mask0> {after_code} {comment}\n'
    template = f'{comment}\n <mask0> {after_code}\n'

    return template


def gen_start_mask_template(token_line: list) -> str:
    """
    Generate a template like this: <mask0> last_token

    Args:
        token_line: The error line composed of tokens.

    Returns:
        A string representing the template.
    """
    last_token = token_line[-1].text
    comment = '// buggy line: ' + ' '.join(token.text for token in token_line)
    # template = f'<mask0> {last_token} {comment}\n'
    template = f'{comment}\n <mask0> {last_token}\n'

    return template


def gen_end_mask_template(token_line: list) -> str:
    """
    Generate a template like this: first_token <mask0> 

    Args:
        token_line: The error line composed of tokens.

    Returns:
        A string representing the template.
    """
    first_token = token_line[0].text
    comment = '// buggy line: ' + ' '.join(token.text for token in token_line)
    # template = f'{first_token} <mask0> {comment}\n'
    template = f'{comment}\n{first_token} <mask0> \n'

    return template


def gen_line_mask_template(token_line: list) -> str:
    """
    Generate a template like this: <mask0>

    Args:
        token_line: The error line composed of tokens.

    Returns:
        A string representing the template.
    """
    comment = '// buggy line: ' + ' '.join(token.text for token in token_line)
    # template = f'<mask0> {comment}\n'
    template = f'{comment}\n <mask0> \n'

    return template
