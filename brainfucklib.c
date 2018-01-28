/*
 Brainfuck-C ( http://github.com/kgabis/brainfuck-c )
 Copyright (c) 2012 Krzysztof Gabis
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this sosftware and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define OP_END          0
#define OP_INC_DP       1
#define OP_DEC_DP       2
#define OP_INC_VAL      3
#define OP_DEC_VAL      4
#define OP_OUT          5
#define OP_IN           6
#define OP_JMP_FWD      7
#define OP_JMP_BCK      8

#define SUCCESS         0
#define FAILURE         1

#define PROGRAM_SIZE    128
#define STACK_SIZE      128
#define DATA_SIZE       1024
#define OUTPUT_SIZE     1024

#define MAX_INSTR 500

typedef struct {
    unsigned int operator;
    unsigned int operand;
} instruction_t;

#define STACK_PUSH(A)   (STACK[SP++] = A)
#define STACK_POP()     (STACK[--SP])
#define STACK_EMPTY()   (SP == 0)
#define STACK_CLEAN()   (SP = 0)
#define STACK_FULL()    (SP == STACK_SIZE)

int compile_bf(instruction_t* program, const char* code) {
    unsigned int pc = 0, jmp_pc;
    char c;
    unsigned int STACK[STACK_SIZE];
    unsigned int SP = 0;
    while ((c = code[pc]) != '\0') {
        switch (c) {
            case '>': program[pc].operator = OP_INC_DP; break;
            case '<': program[pc].operator = OP_DEC_DP; break;
            case '+': program[pc].operator = OP_INC_VAL; break;
            case '-': program[pc].operator = OP_DEC_VAL; break;
            case '.': program[pc].operator = OP_OUT; break;
            case ',': program[pc].operator = OP_IN; break;
            case '[':
                program[pc].operator = OP_JMP_FWD;
                program[pc].operand = pc;
                STACK_PUSH(pc);
                break;
            case ']':
                program[pc].operator = OP_JMP_BCK;
                if (STACK_EMPTY()) { // jump 1 instr forward
                    program[pc].operand = pc;
                    break;
                }
                jmp_pc = STACK_POP();
                program[pc].operand = jmp_pc;
                program[jmp_pc].operand = pc;
                break;
            default: return FAILURE; break;
        }
        pc++;
    }
    if (pc >= PROGRAM_SIZE) {
        return FAILURE;
    }
    program[pc].operator = OP_END;
    return SUCCESS;
}

int execute_bf(instruction_t* program, const char* input, char* output) {
    char* out_begin = output;
    unsigned int pc = 0;
    unsigned int ptr = 0;
    unsigned int instr_q = 0;
    char data[DATA_SIZE];
    memset(data, 0, DATA_SIZE);
    while (program[pc].operator != OP_END && instr_q < MAX_INSTR) {
        switch (program[pc].operator) {
            case OP_INC_DP: ptr++; break;
            case OP_DEC_DP:  if(ptr)ptr--; break;
            case OP_INC_VAL: data[ptr]++; break;
            case OP_DEC_VAL: data[ptr]--; break;
            case OP_OUT: *(output++) = (data[ptr]); break;
            case OP_IN: data[ptr] = (unsigned int)(*(input++)); break;
            case OP_JMP_FWD: if(!data[ptr]) { pc = program[pc].operand; } break;
            case OP_JMP_BCK: if(data[ptr]) { pc = program[pc].operand; } break;
            default: return FAILURE;
        }
        pc++;
        instr_q++;
    }
    *output = '\0';
    if(instr_q == MAX_INSTR)
        strcpy(out_begin, "INFINFINFINFINFINFINFINFINFINFINFINFINFINFINFINF\0"); // probably wrong output for all cases
    return SUCCESS;
}

char* compute_bf(const char* code, const char* input, char* output)
{
    instruction_t program[PROGRAM_SIZE];
    memset(output, 0, OUTPUT_SIZE);
    if(compile_bf(program, code))
        printf("compilation error");
    if(execute_bf(program, input, output))
        printf("runtime error");
    return output;
}

