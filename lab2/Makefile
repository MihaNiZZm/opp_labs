all: refresh_directories straight parallel distribute_files

refresh_directories:
	rm -rf executable_files
	mkdir executable_files

straight:
	mpicxx ./src/straight.c -o straight.out

parallel:
	mpicxx ./src/parallel.c-o parallel.out

distribute_files:
	mv straight.out executable_files
	mv parallel.out executable_files
