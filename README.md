Matrix
===============================================

The concept of Matrices or Tensors is widely used in mathematical modeling of physical phenomena. For instnace, the stress tensor is a 2nd rank tensor or full Navier-Stokes equations can be written in terms of five dimensional vectors which are 1st rank tensors. Here a Matrix implementation in C++ is provided which can be used for many applications from linear algebra to simulation of physical systems.


Table of contents
-----------------

- [Usage](#usage)
- [Performance](#performance)
- [Acknowledgement](#acknowledgement)
- [Licence](#licence)


Usage
-----

The first step is to include the following library

```C++
#include "matrix.hpp"
```

You will need C++17 to compile the code.

#### Creating Matrices

You can define Tensors, Matrices and Vectors like this:
```C++
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
  
  auto E = B;           // copy assignment is defined
  auto F(B);            // copy constructor is defined
  
  auto elem = B(1,2);   // elem is the 1st row and 2nd column of B which is 6.
```

#### Matrix operations
These Matrix operations are defined:

- Matrix multiplication,
- Addition,
- Elementwise multiplication
- Operations (*, /, %, +, ...) with a scalar is defined

These operations are defined in the namespace "Matrix_operations". Therefore, you have use this namespace:

```C++
  using namespace Matrix_operations;
```

Here are som examples of how to use Matrix operations:
```C++
  Matrix<double,2> v {{1.,0.}};
  Matrix<double,2> u {{2.,3.}};
  auto M = u*v;     // M is a tensor resulting from tensor product of two vectors
  
  Matrix<double,2> M1 {
    {1., 0., 3.},
    {2., 1., 0.}
  };
  Matrix<double,2> M2 {
    {2.,3.},
    {4.,5.},
    {6.,1.}
  };
  auto K1 = M1*M2;    // K1 is a 2 by 2 Matrix
  auto K2 = M2*M1;    // K2 is a 3 by 3 Matrix

  //auto L = K1+K2;   // Error: Matrices are not the same size
```

#### Extension to Linear Algebra operations
This Matrix class can be extended to do linear algebra calculations such as LU decomposition, Conjugate Gradient and so on.


Performance
-----------
STUDIS should not incur any noticeable overhead at runtime. Most of the dimension checks are performed during compilation and incur no cost at runtime.


Acknowledgement
---------------

This is inspired by the idea of Matrix design discussed in the book _The C++ Programming Language_ by _Bjarne Stroustrup_.


Licence
-------

This library is distributed under the terms of Non-Discriminatory Public Licence. You can read the exact licence terms in the 'LICENSE' file, but here is a summary:

- You can use and modify the software
- You can distribute the original or the modified version of the software under the same terms in a non-discriminatory manner if you also provide the source code

If you have to comply with laws that compels you to restrict access of certain groups of people (such as export control laws), you can only use and modify this software for your own purposes, but you can no longer distribute it.
