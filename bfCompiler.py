def parse(code):
    code = filter(lambda x: x in ['.', ',', '[', ']', '<', '>', '+', '-'], code)
    open_braces = 0
    unmatched_braces = 0
    open_braces_pos = []
    for position, char in enumerate(code):
        if char == "[":
            open_braces += 1
            open_braces_pos.append(position)
        if char == "]":
            if open_braces:
                open_braces -= 1
                open_braces_pos.pop()
            else:
                unmatched_braces += 1
                code = code[:position] + "N" + code[position + 1:]
    unmatched_braces += len(open_braces_pos)
    for i, position in enumerate(open_braces_pos):
        code = code[:(position - i)] + code[(position - i + 1):]
    code = code.replace("N", "")
    return code, unmatched_braces


def buildbracemap(code):
    bracestack, bracemap = [], {}
    for position, command in enumerate(code):
        if command == "[":
            bracestack.append(position)
        if command == "]":
            start = bracestack.pop()
            bracemap[start] = position
            bracemap[position] = start
    return bracemap


def bfEvaluate(code, input_list, bracemap, ops_limit, max_ops):
    output = ''
    input_list += '0'
    input_len = len(input_list) - 1
    code_ptr = cell_ptr = input_ptr = 0

    cells_num = 3000
    cells = [0] * cells_num

    operations_counter = 0

    while code_ptr < len(code):
        command = code[code_ptr]
        if ops_limit and operations_counter > max_ops:
            return "", max_ops
        operations_counter += 1
        if command == ">":
            cell_ptr = cell_ptr + 1 if cell_ptr + 1 < cells_num else 0

        if command == "<":
            cell_ptr = max(0, cell_ptr - 1)

        if command == "+":
            cells[cell_ptr] = cells[cell_ptr] + \
                1 if cells[cell_ptr] < 255 else 0

        if command == "-":
            cells[cell_ptr] = cells[cell_ptr] - \
                1 if cells[cell_ptr] > 0 else 255

        if command == "[" and cells[cell_ptr] == 0:
            code_ptr = bracemap[code_ptr]
        if command == "]" and cells[cell_ptr] != 0:
            code_ptr = bracemap[code_ptr]

        if command == ".":
            output += chr(cells[cell_ptr])
        if command == ",":
            cells[cell_ptr] = ord(input_list[input_ptr])
            input_ptr += (input_ptr < input_len)

        code_ptr += 1

    return output, operations_counter


def BF(code, input_list='', ops_limit=True, max_ops=5000):
    '''
        compiler of BF code
        Args:
            code (string): BF code
            input_list (string): input assigned to code (if needed)
            ops_limit (bool): to prevent infinite loops you can return None after max_ops steps (default: False)
            max_ops (int): if ops_limit is set to true determines the maximum number of operations (default: 5000)
        Returns:
            output: string with output for given code (may be None if ops_limit is true)
            opertations_counter: number of program operations
            unmatched_braces: number of unmatched braces (used in strict mode)
    '''
    code, unmatched_braces = parse(code)
    bracemap = buildbracemap(code)
    output, operations_counter = bfEvaluate(
        code, input_list, bracemap, ops_limit, max_ops)
    return output, operations_counter, unmatched_braces
