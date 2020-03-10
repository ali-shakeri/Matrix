#include "matrix.hpp"

using namespace Matrix_operations;

int main(int argc, char **argv) {
  Matrix<int,1> A;            // A vector of integers
  
  Matrix<double,2> B {        // A matrix of doubles
    {1.,2.,3.},
    {4.,5.,6.}
  };
  
  Matrix<float,3> C(3,4,2);   // A 3 by 2 by 4 tensor of rank 3
  
  Matrix<Matrix<int,2>,2> D { // Nested use of Matrix
    {
      {{1,2},{3,4}},
      {{5,6},{7,8}},
    },
    {
      {{8,7},{5,6}},
      {{4,3},{2,1}},
    }
  };
  
  auto E = B;       // copy assignment is defined
  auto F(B);        // copy constructor is defined
  
//   error: ‘std::is_convertible_v<int, long unsigned int>’ cannot be used as a function
//   auto elem = B(1,2);
//   B (slice{0,1},slice{0}) = {7., 8., 9.};
  
//   auto F{B};     // Error: only use of {} for elements
  
  //Matrix<double,1> b = B[1]; // Error: Matrix_ref is incomplete
  
//   std::cout << B << std::endl;  // two functions rows() and cols() are missing
  return 0;
}
