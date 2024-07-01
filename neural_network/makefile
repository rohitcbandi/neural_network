CXX = g++
CXXFLAGS = -std=c++11 -I include

SRC = src/activation.cpp src/dense.cpp src/conv1d.cpp src/recurrent.cpp src/adam.cpp src/neural_network.cpp src/random.cpp main.cpp
OBJ = $(SRC:.cpp=.o)

main: $(OBJ)
    $(CXX) $(CXXFLAGS) -o main $(OBJ)

%.o: %.cpp
    $(CXX) $(CXXFLAGS) -c $< -o $@

clean:
    rm -f $(OBJ) main
