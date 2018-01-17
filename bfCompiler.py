def parse(code):
    return filter(lambda x: x in ['.', ',', '[', ']', '<', '>', '+', '-'], code)


def buildbracemap(code):
    bracestack, bracemap = [], {}
    unmatched_braces, unmatched_braces_pos = 0, []
    for position, command in enumerate(code):
        if command == "[":
            bracestack.append(position)
        if command == "]":
            if bracestack == []:
                unmatched_braces += 1
                unmatched_braces_pos.append(position)
            else:
                start = bracestack.pop()
                bracemap[start] = position
                bracemap[position] = start
    unmatched_braces += len(bracestack)
    unmatched_braces_pos += bracestack
    for i in xrange(len(unmatched_braces_pos)):
        position = unmatched_braces_pos[i]
        code = code[:(position - i)] + code[(position - i + 1):]
    return code, bracemap, unmatched_braces


def bfEvaluate(code, input_list, bracemap, ops_limit, max_ops):
    output = ''
    code_ptr = cell_ptr = input_ptr = 0

    cells_num = 3000
    cells = [0] * cells_num

    operations_counter = 0

    while code_ptr < len(code):
        command = code[code_ptr]
        if ops_limit and operations_counter > max_ops:
            return None, max_ops
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
            input_ptr += 1

        code_ptr += 1

    return output, operations_counter


def BF(code, input_list='', ops_limit=False, max_ops=5000):
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
    code = parse(code)
    code, bracemap, unmatched_braces = buildbracemap(code)
    output, operations_counter = bfEvaluate(
        code, input_list, bracemap, ops_limit, max_ops)
    return output, operations_counter, unmatched_braces
