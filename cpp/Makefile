# Makefile that builds all .cpp files and combines them into an executable

NAME = main
INCLUDE = -I.
CXXFLAGS = -O2 -Wall -Wextra

CPP_FILES := $(wildcard *.cpp)
OBJ_FILES := $(patsubst %.cpp,%.o,$(CPP_FILES))

all: $(NAME)

$(NAME): $(OBJ_FILES)
	g++ $(CXXFLAGS) $(INCLUDE) $^ -o $@

%.o: %.cpp
	g++ $(CXXFLAGS) $(INCLUDE) -c $< -o $@

clean:
	rm -f *.o $(NAME)
