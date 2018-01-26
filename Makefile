

brainfucklib: brainfucklib.c
	gcc -o brainfucklib.so -shared -fPIC -O3 brainfucklib.c
