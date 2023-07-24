# Makefile that builds all .cpp files and combines them into an executable

NAME = main
LINK = -lpng -L./libpng/build
INCLUDE = -I. -I./libpng -I./libpng/build
CXXFLAGS = -O2 -Wall -Wextra -std=c++14

CPP_FILES := $(wildcard *.cpp)
OBJ_FILES := $(patsubst %.cpp,%.o,$(CPP_FILES))

all: program

.PHONY: png program

png:
	mkdir -p libpng/build
	cd libpng/build && cmake .. && make

$(NAME): $(OBJ_FILES)
	g++ $(CXXFLAGS) $(INCLUDE) $^ $(LINK) -o $@

program: png $(NAME)

%.o: %.cpp
	g++ $(CXXFLAGS) $(INCLUDE) -c $< $(LINK) -o $@

clean:
	rm -f *.o $(NAME)
	rm -rf libpng/build
