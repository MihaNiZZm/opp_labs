all: refresh_dirs straight point_point collective distribute_files

refresh_dirs:
	rm -rf executable_files
	mkdir executable_files

straight:
	mpicxx straight.c -o straight.out

point_point:
	mpicxx point_point.c -o point_to_point.out

collective:
	mpicxx collective.c -o collective.out

distribute_files:
	mv straight.out executable_files
	mv point_to_point.out executable_files
	mv collective.out executable_files