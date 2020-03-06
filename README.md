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

Then you can define Tensors, Matrices and Vectors like this:
```C++
  Matrix<int,1> v;                      // A vector of ints
  Matrix<double,2> m = {{1.,2.,3.},     // A matrix of doubles
                        {4.,5.,6.}};
  Matrix<double,3> B(3,2,4);            // A 3 by 2 by 4 tensor of rank 3
```


Performance
-----------



Acknowledgement
---------------

This is inspired by the idea of Matrix design discussed in the book _The C++ Programming Language_ by _Bjarne Stroustrup_.


Licence
-------

This library is distributed under the terms of Non-Discriminatory Public Licence. You can read the exact licence terms in the 'LICENSE' file, but here is a summary:

- You can use and modify the software
- You can distribute the original or the modified version of the software under the same terms in a non-discriminatory manner if you also provide the source code

If you have to comply with laws that compels you to restrict access of certain groups of people (such as export control laws), you can only use and modify this software for your own purposes, but you can no longer distribute it.
