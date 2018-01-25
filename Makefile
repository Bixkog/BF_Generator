

brainfucklib: brainfucklib.c
	gcc -o brainfucklib.so -shared -fPIC -g brainfucklib.c
