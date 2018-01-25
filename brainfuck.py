import ctypes
import os
path = os.path.dirname(os.path.realpath(__file__))
bf = ctypes.cdll.LoadLibrary("%s/brainfucklib.so"%path)

bf.compute_bf.argtypes = (ctypes.c_char_p, ctypes.c_char_p)
bf.compute_bf.restype = ctypes.c_void_p

bf.free_out.argtypes = (ctypes.c_void_p,)

def BF(code, input):
	c_code = ctypes.c_char_p(code)
	c_input = ctypes.c_char_p(input)
	c_output = bf.compute_bf(c_code, c_input)
	output = ctypes.cast(c_output, ctypes.c_char_p).value # save before free
	bf.free_out(c_output) # free C memory
	return output

print BF(",>++++++[<-------->-],[<+>-]<.", "11")