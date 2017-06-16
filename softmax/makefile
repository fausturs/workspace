#softmax makefile

file_name=softmax
target_file=$(file_name)
cpp_compiler=g++
cpp_version=-std=c++14
optimization_level=-O3
#include_path=
#link_path=

./bin/$(target_file).out: ./src/$(file_name).cpp ./bin ../tools/Timer.o
	$(cpp_compiler) $(cpp_version) $(optimization_level) \
	./src/$(file_name).cpp \
	../tools/Timer.o \
	-o ./bin/$(target_file).out

./bin:
	mkdir bin

../tools/Timer.o: ../tools/Timer.cpp ../tools/Timer.hpp
	$(cpp_compiler) $(cpp_version) $(optimization_level) \
	-c ../tools/Timer.cpp \
	-o ../tools/Timer.o

clean:
	#clean rm ./bin/*.o
	rm ./bin/$(target_file).out
