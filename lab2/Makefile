all: refresh_directories straight parallel distribute_files

refresh_directories:
	rm -rf executable_files
	mkdir executable_files

straight:
	mpicxx ./src/straight.c -o straight.out

parallel:


distribute_files:
	mv straight.out executable_files
