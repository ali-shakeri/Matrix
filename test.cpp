#include "matrix.hpp"

int main(int argc, char **argv) {
  Matrix<double,2> m = {{1.,2.,3.},{4.,5.,6.}};
  Matrix<int,1> A;
  Matrix<std::string,3> B(3,4,2);
  return 0;
}
