import ctypes
import os
path = os.path.dirname(os.path.realpath(__file__))
bf = ctypes.cdll.LoadLibrary("%s/brainfucklib.so"%path)

bf.compute_bf.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p)
bf.compute_bf.restype = ctypes.c_void_p

output_buf = ctypes.create_string_buffer(1024)

def BF(code, input):
	c_code = ctypes.c_char_p(code)
	c_input = ctypes.c_char_p(input.ljust(500, '\0'))
	c_output = bf.compute_bf(c_code, c_input, output_buf)
	output = ctypes.cast(c_output, ctypes.c_char_p).value # save before free
	return output