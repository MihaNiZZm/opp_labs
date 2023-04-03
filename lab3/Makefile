all: refresh_directories parallel distribute_files

refresh_directories:
	rm -rf executable_files
	mkdir executable_files

parallel:
	gcc -fopenmp ./src/parallel.c -o parallel.out

distribute_files:
	mv parallel.out executable_files
