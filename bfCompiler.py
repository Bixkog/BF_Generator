def parse(code):
    return filter(lambda x: x in ['.', ',', '[', ']', '<', '>', '+', '-'], code)


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


def bfEvaluate(code, input_list, bracemap):

    code_ptr = cell_ptr = input_ptr = 0

    cells_num = 3000
    cells = [0] * cells_num

    while code_ptr < len(code):
        command = code[code_ptr]

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
            print(chr(cells[cell_ptr]))
        if command == ",":
            cells[cell_ptr] = ord(input_list[input_ptr])
            input_ptr += 1

        code_ptr += 1


def BF(code, input_list=''):
    '''
        compiler of BF code
        input:
            code (string): BF code
            input_list (string): input assigned to code (if needed)
    '''
    code = parse(code)
    bracemap = buildbracemap(code)
    bfEvaluate(code, input_list, bracemap)
