#include <vector>
#include <array>
#include <numeric>
#include <initializer_list>
#include <type_traits>
#include <cassert>
#include <algorithm>
#include <iostream>

//TODO list:
// * The clases nead not be in namespace Matrix_impl
// * Specializtions of Matrix_slice does not work
// * Matrix_ref should be a clone of Matrix with almost all functions

constexpr bool All() {return true;}

template<typename... Args>
constexpr bool All(bool b, Args... args) 
{
  return b && All(args...);
}

// ***** Matrix_init *****//
template<typename T, size_t N>
struct Matrix_init {
  using type = std::initializer_list<typename Matrix_init<T,N-1>::type>;
};

template<typename T>
struct Matrix_init<T,1> {
  using type = std::initializer_list<T>;
};

template<typename T>
struct Matrix_init<T,0>;

template<typename T, size_t N>
using Matrix_initializer = typename Matrix_init<T,N>::type;


// declarations  
template<size_t N>
struct Matrix_slice;

namespace Matrix_impl {
  template<size_t N, typename List>
  std::array<size_t,N> derive_extents (const List&);
  
  template<size_t N, typename Itr, typename List>
  std::enable_if_t<(N>1),void>  add_extents (Itr&, const List&);
  
  template<size_t N, typename Itr, typename List>
  std::enable_if_t<(N==1),void>  add_extents (Itr&, const List&);
  
  template<typename T, typename Vec>
  void add_list (const std::initializer_list<T>*,
                 const std::initializer_list<T>*, Vec&);
  
  template<typename T, typename Vec>
  void add_list (const T*, const T*, Vec&);
  
  template<size_t N, typename List>
  bool check_non_jagged (const List&);
  
  template<size_t N>
  void compute_strides (Matrix_slice<N>&);
  
  template<typename... Args>
  constexpr bool Requesting_element ();
  
  template<typename... Args>
  constexpr bool Requesting_slice ();
  
  template<typename T, typename Vec>
  void insert_flat (std::initializer_list<T>, Vec&);
  
  template<size_t N, typename... Dims>
  bool check_bounds (const Matrix_slice<N>&, Dims...);
  
  template<size_t N, typename T, typename... Args>
  size_t do_slice (const Matrix_slice<N>&, Matrix_slice<N>&,
                   const T&, const Args&... );
  
  template<size_t N>
  size_t do_slice(const Matrix_slice<N>&, Matrix_slice<N>&);
  
  template<size_t N, typename T, typename... Args>
  size_t do_slice_dim (const Matrix_slice<N>&, Matrix_slice<N>&, const T&);
  
}


// TODO do we need this class for 2D Specializtions?
struct slice {
  slice () : start(-1), length(-1), stride(1) {}
  explicit slice (size_t s) : start(s), length(-1), stride(1) {}
  slice (size_t s, size_t l, size_t n=1) : start(s), length(l), stride(n) {}
  
  size_t operator()(size_t i) const {return start + i*stride;}
  
  static slice all;
  
  size_t start;
  size_t length;
  size_t stride;
};

// ***** Matrix_slice *****//
// This structure holds the sizes necessary to access the elements in an N-dim Matrix.
template<size_t N>
struct Matrix_slice {
  Matrix_slice () = default;
  
  Matrix_slice (size_t, std::initializer_list<size_t>);
  Matrix_slice (size_t, std::initializer_list<size_t>,
                        std::initializer_list<size_t>);
  
  template<typename... Dims>
    Matrix_slice (Dims... dims);
  
  template<typename... Dims>
    size_t operator()(Dims... dims) const;
    //TODO: fix the error of following function:
//   template<typename... Dims, std::enable_if_t<All(std::is_convertible_v<Dims,size_t>()...)> >
//     size_t operator()(Dims... dims) const;
  
  size_t size;                    // total number of elements
  size_t start;                   // starting offset
  std::array<size_t,N> extents;   // number of elements in each direction
  std::array<size_t,N> strides;   // offset between elements in each dimension
};

template<size_t N>
Matrix_slice<N>::Matrix_slice (size_t start, std::initializer_list<size_t> exts)
  : start {start}
{
  assert(exts.size()==N);
  std::copy_n(exts.begin(), N, extents.begin());
  std::fill_n(strides.begin(), N, 1);
  size = std::accumulate (exts.begin(), exts.end(), 1, std::multiplies<size_t>());
}

template<size_t N>
Matrix_slice<N>::Matrix_slice (size_t start, std::initializer_list<size_t> exts,
  std::initializer_list<size_t> strs) : start{start}
{
  assert(exts.size()==N);
  assert(strs.size()==N);
  std::copy_n(exts.begin(), N, extents.begin());
  std::copy_n(strs.begin(), N, strides.begin());
  size = std::accumulate (exts.begin(), exts.end(), 1, std::multiplies<size_t>());
}

template<size_t N>
template<typename... Dims>
Matrix_slice<N>::Matrix_slice (Dims... dims) : extents {{size_t(dims)...}}
{
  static_assert(sizeof...(Dims)==N, "dimension mismatch");
  size = std::accumulate (extents.begin(), extents.end(), 1, std::multiplies<size_t>());
}

template<size_t N>
template<typename... Dims>
size_t Matrix_slice<N>::operator()(Dims... dims) const
{
  static_assert(sizeof...(Dims)==N, "dimension mismatch");
  size_t args[N] {size_t(dims)...};  
  return start + std::inner_product(args, args+N, strides.begin(), size_t{0});
}


// ***** Matrix_ref *****//
template<typename T, size_t N>
class Matrix_ref {
public:
  Matrix_ref (const Matrix_slice<N>& s, T* p) : desc{s}, ptr{p} {}
  // TODO: complete the body
private:
  Matrix_slice<N> desc;
  T* ptr;
};



// ***** Matrix *****//
template<typename T, size_t N>
class Matrix {
public:
  static constexpr size_t order = N;
  using value_type = T;
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;
  
  Matrix () = default;
  Matrix (Matrix &&) = default;
  Matrix& operator=(Matrix &&) = default;
  Matrix (const Matrix&) = default;
  Matrix& operator=(const Matrix&) = default;
  ~Matrix() = default;
  
  template<typename U>
    Matrix (const Matrix_ref<U,N>&);
  template<typename U>
    Matrix& operator=(const Matrix_ref<U,N>&);
  
  Matrix (Matrix_initializer<T,N>);
  Matrix& operator=(Matrix_initializer<T,N>);
  
  template<typename U>
    Matrix (std::initializer_list<U>) = delete;
  template<typename U>
    Matrix& operator=(std::initializer_list<U>) = delete;
  
  template<typename... Exts>
    explicit Matrix (Exts... extents);
  
  size_t extent (size_t n) const {return desc.extents[n];}
  size_t size () const {return elems.size();}
  const Matrix_slice<N>& descriptor() const {return desc;}
  
  T* data () {return elems.data();}
  const T* data () const {return elems.data();}
  
  // m(i,j,k) subscripting with integers
  template<typename... Args>
    std::enable_if_t<(std::is_convertible_v<Args,size_t> && ...), T&>
    operator()(Args... args);
  template<typename... Args>
    std::enable_if_t<(std::is_convertible_v<Args,size_t> && ...), const T&>
    operator()(Args... args) const;
  
  // m(s1,s2,s3) subscripting with slices
  template<typename... Args>
    std::enable_if_t<(std::is_convertible_v<Args,slice> && ...), Matrix_ref<T,N>>
    operator()(const Args&... args);
  template<typename... Args>
    std::enable_if_t<(std::is_convertible_v<Args,slice> && ...), Matrix_ref<const T,N>>
    operator()(Args... args) const;
  
  // m[i] row access  
  Matrix_ref<T,N-1> operator[](size_t i) {return row(i);}
  Matrix_ref<const T,N-1> operator[](size_t i) const {return row(i);}
  
  // row access
  Matrix_ref<T,N-1> row (size_t n);
  Matrix_ref<const T,N-1> row (size_t n) const;
  
  // column access
  Matrix_ref<T,N-1> col (size_t n);
  Matrix_ref<const T,N-1> col (size_t n) const;
  
  // arithmetic operations
  template<typename F>
    Matrix& apply(F);
  template<typename M, typename F>
    std::enable_if_t<std::is_same_v<M,Matrix>, Matrix&> apply (const M&, F);
    
  Matrix& operator=(const T& value)  {return apply([&](T& a){a=value;});}
  Matrix& operator+=(const T& value) {return apply([&](T& a){a+=value;});}
  Matrix& operator-=(const T& value) {return apply([&](T& a){a-=value;});}
  Matrix& operator*=(const T& value) {return apply([&](T& a){a*=value;});}
  Matrix& operator/=(const T& value) {return apply([&](T& a){a/=value;});}
  Matrix& operator%=(const T& value) {return apply([&](T& a){a%=value;});}
  
  template<typename M>
    std::enable_if_t<std::is_same_v<M,Matrix>, Matrix&> operator+=(const M&);
  template<typename M>
    std::enable_if_t<std::is_same_v<M,Matrix>, Matrix&> operator-=(const M&);
private:
  Matrix_slice<N> desc;
  std::vector<T> elems;
};

template<typename T, size_t N>
  template<typename... Exts>
  Matrix<T,N>::Matrix(Exts... extents) : desc{extents...}, elems(desc.size) {}

template<typename T, size_t N>
Matrix<T,N>::Matrix(Matrix_initializer<T,N> init) {
  desc.extents = Matrix_impl::derive_extents<N> (init);
  Matrix_impl::compute_strides (desc);
  elems.reserve (desc.size);
  Matrix_impl::insert_flat (init, elems);
  assert (elems.size() == desc.size);
}

template<typename T, size_t N>
Matrix<T,N>& Matrix<T,N>::operator=(Matrix_initializer<T,N> init) {
  desc.extents = Matrix_impl::derive_extents<N> (init);
  Matrix_impl::compute_strides (desc);
  elems.reserve (desc.size);
  Matrix_impl::insert_flat (init, elems);
  assert (elems.size() == desc.size);
  return *this;
}

template<typename T, size_t N>
template<typename U>
Matrix<T,N>::Matrix (const Matrix_ref<U,N>& x)
: desc{x.desc}, elems{x.begin(), x.end()}
{
  static_assert(std::is_convertible_v<U,T>(), "Can not convert: Incompatible types");
}

template<typename T, size_t N>
template<typename U>
Matrix<T,N>& Matrix<T,N>::operator=(const Matrix_ref<U,N>& x)
{
  static_assert(std::is_convertible_v<U,T>(), "Can not convert: Incompatible types");
  desc = x.desc;
  elems.assign (x.begin(), x.end());
  return *this;
}


// m(i,j,k) subscripting with integers
template<typename T, size_t N>
  template<typename... Args>
  std::enable_if_t<(std::is_convertible_v<Args,size_t> && ...), T&>
  Matrix<T,N>::operator()(Args... args)
  {
    assert(Matrix_impl::check_bounds(desc, args...));
    return *(data() + desc(args...));
  }

// m(s1,s2,s3) subscripting with slices
template<typename T, size_t N>
  template<typename... Args>
  std::enable_if_t<(std::is_convertible_v<Args,slice> && ...), Matrix_ref<T,N>>
  Matrix<T,N>::operator()(const Args&... args)
  {
    Matrix_slice<N> d;
    d.start = Matrix_impl::do_slice (desc, d, args...);
    return {d, data()};
  }


template<typename T, size_t N>
Matrix_ref<T,N-1> Matrix<T,N>::row (size_t n) 
{
//   assert (n<rows());
  Matrix_slice<N-1> row;
//   Matrix_impl::slice_dim<0>(n, desc, row);
  return {row, data()};
}

// template<typename T>
// T& Matrix<T,1>::row (size_t n) {return elems[n];}
// 
// template<typename T>
// T& Matrix<T,0>::row (size_t n) = delete;

template<typename T, size_t N>
Matrix_ref<T,N-1> Matrix<T,N>::col (size_t n) 
{
//   assert (n<cols());
  Matrix_slice<N-1> col;
//   Matrix_impl::slice_dim<1>(n, desc, col);
  return {col, data()};
}

// template<typename T>
// T& Matrix<T,0>::col (size_t n) = delete;


template<typename T, size_t N>
  template<typename F>
  Matrix<T,N>& Matrix<T,N>::apply(F f)
  {
    for (auto& x : elems) f(x);
    return *this;
  }
  
template<typename T, size_t N>
  template<typename M, typename F>
  std::enable_if_t<std::is_same_v<M,Matrix<T,N>>, Matrix<T,N>&>
  Matrix<T,N>::apply(const M& m, F f)
  {
    assert(same_extents(desc, m.descriptor()));
    for (auto i=elems.begin(),j=m.begin(); i!=elems.end(); ++i, ++j)
      f(*i, *j);
    return *this;
  }
  
template<typename T, size_t N>
  template<typename M>
  std::enable_if_t<std::is_same_v<M,Matrix<T,N>>, Matrix<T,N>&>
  Matrix<T,N>::operator+=(const M& m)
  {
    static_assert (m.order==N, "Matrix dimensions do not match");
    assert(same_extents(desc,m.descriptor()));
    return apply(m, [](T& a, const typename M::value_type& b){a+=b;});
  }

template<typename T, size_t N>
  template<typename M>
  std::enable_if_t<std::is_same_v<M,Matrix<T,N>>, Matrix<T,N>&>
  Matrix<T,N>::operator-=(const M& m)
  {
    static_assert (m.order==N, "Matrix dimensions do not match");
    assert(same_extents(desc,m.descriptor()));
    return apply(m, [](T& a, const typename M::value_type& b){a-=b;});
  }
 
template<typename T>
class Matrix<T,0> {
public:
  static constexpr size_t order = 0;
  using value_type = T;
  
  Matrix (const T& elem) : elem{elem} {}
  Matrix& operator=(const T& value) 
  {
    elem=value;
    return *this;
  }
  T& operator()() {return elem;}
  const T& operator()() const {return elem;}
  
  operator T&() {return elem;}
  operator const T&() {return elem;}
  
  T& col (size_t) = delete;
  T& row (size_t) = delete;
private:
  T elem;
};




// ***** Matrix operations *****//
namespace Matrix_operations {
  template<typename T>
  T dot_product (const Matrix_ref<T,1>& a, const Matrix_ref<T,1>& b)
  {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
  }
  
  template<typename T, size_t N>
  Matrix<T,N> operator+(const Matrix<T,N>& a, const Matrix<T,N>& b)
  {
    Matrix<T,N> res = a;
    res+=b;
    return res;
  }
  
  // Multiply N by 1 vector u by a 1 by M vector v, returns a N by M Matrix
  template<typename T>
  Matrix<T,2> operator*(const Matrix<T,1>& u, const Matrix<T,1>& v)
  {
    const size_t n = u.extent(0);
    const size_t m = v.extent(0);
    Matrix<T,2> res(n,m);
    for (size_t i=0; i!=n; ++i)
      for (size_t j=0; j!=m; ++j)
        res(i,j) = u[i]*v[j];
    return res;
  } // TODO: check the overhead discussed in page 839
  
  // Multiply N by M matrix m by a M by 1 vector v, returns a N by 1 vector
  template<typename T>
  Matrix<T,1> operator*(const Matrix<T,2>& m, const Matrix<T,1>& v)
  {
    assert(m.extent(1)==v.extent(0));
    const size_t nrows = m.extent(0);
    const size_t ncols = m.extent(1);
    Matrix<T,1> res(nrows);
    for (size_t i=0; i!=nrows; ++i)
      for (size_t j=0; j!=ncols; ++j)
        res(i) += m(i,j)*v(j);
    return res;
  }
  
  // Multiply N by M matrix m1 by a M by P matrix m2, returns a N by P Matrix
  template<typename T>
  Matrix<T,2> operator*(const Matrix<T,2>& m1, const Matrix<T,2>& m2)
  {
    assert(m1.extent(1)==m2.extent(0));
    const size_t nrows = m1.extent(0);
    const size_t ncols = m1.extent(1);
    const size_t p = m2.extent(0);
    Matrix<T,2> res(nrows, p);
    for (size_t i=0; i!=nrows; ++i)
      for (size_t j=0; j!=p; ++j)
        res(i,j) = dot_product(m1[i], m2.column(j));
    return res;
  }
  
  //TODO: Make it more general: see page 829
  template<typename T, size_t N>
  std::ostream& operator<< (std::ostream& os, const Matrix<T,N>& m)
  {
    os << '{';
    for (size_t i=0; i!=rows(m); ++i) {
      os < m[i];
      if (i+1 != rows(m)) os << ',';
    }
    return os << '}';
  }
  
  
}



// implementations
namespace Matrix_impl {
  template<size_t N, typename Itr, typename List>
  std::enable_if_t<(N>1),void> add_extents (Itr& first, const List& list)
  {
    assert (check_non_jagged<N>(list));
    *first++ = list.size();
    add_extents<N-1> (first, *list.begin());
  }
  
  template<size_t N, typename Itr, typename List>
  std::enable_if_t<(N==1),void> add_extents (Itr& first, const List& list)
  {
    *first++ = list.size();
  }
  
  template<size_t N, typename List>
  std::array<size_t,N> derive_extents (const List& list)
  {
    std::array<size_t,N> a;
    auto first = a.begin();
    add_extents<N> (first, list);
    return a;
  }

  template <size_t N, typename List>
  bool check_non_jagged (const List& list)
  {
    auto i = list.begin();
    for (auto j=i+1; j!=list.end(); ++j)
      if (derive_extents<N-1>(*i) != derive_extents<N-1>(*j))
        return false;
    return true;
  }
  
  template<size_t N>
  void compute_strides (Matrix_slice<N>& matrix_slice)
  {
    size_t stride = 1;
    for (int i=N-1; i>=0; --i) {
      matrix_slice.strides[i] = stride;
      stride *= matrix_slice.extents[i];
    }
    matrix_slice.size = stride;
  }
  
  template<typename T, typename Vec>
  void insert_flat (std::initializer_list<T> list, Vec& vec)
  {
    add_list (list.begin(), list.end(), vec);
  }
  
  template<typename T, typename Vec>
  void add_list (const std::initializer_list<T>* first,
                 const std::initializer_list<T>* last, Vec& vec)
  {
    for ( ; first!=last; ++first)
      add_list (first->begin(), first->end(), vec);
  }
  
  template<typename T, typename Vec>
  void add_list (const T* first, const T* last, Vec& vec)
  {
    vec.insert(vec.end(), first, last);
  }
  
  template<typename... Args>
  constexpr bool Requesting_element ()
  {
    return All(std::is_convertible_v<Args,size_t>()...);
  }
  
  template<typename... Args>
  constexpr bool Requesting_slice ()
  {
    return All(std::is_convertible_v<Args,size_t>()...);
//     return All((std::is_convertible_v<Args,size_t>() || Same<Args,slice>())...)
//       && Some(Same<Args,slice>()...);
  }
  
  template<size_t N, typename... Dims>
  bool check_bounds (const Matrix_slice<N>& slice, Dims... dims)
  {
    size_t indexes[N] {size_t(dims)...};
    return std::equal (indexes, indexes+N, slice.extents.begin(),
                       std::less<size_t>{});
  }
  
  template<size_t N, typename T, typename... Args>
  size_t do_slice (const Matrix_slice<N>& os, Matrix_slice<N>& ns, const T& s,
                   const Args&... args)
  {
    size_t m = do_slice_dim<sizeof...(Args)+1> (os, ns, s);
    size_t n = do_slice (os, ns, args...);
    return n+m;
  }
  
  template<size_t N>
  size_t do_slice(const Matrix_slice<N>& os, Matrix_slice<N>& ns)
  {
    return 0;
  }
  

  template<size_t N, typename T, typename... Args>
  size_t do_slice_dim (const Matrix_slice<N>& os, Matrix_slice<N>& ns, const T& s)
  {
    //TODO: implement this function
  }
//   template<>
//   void slice_dim<0> (size_t n, Matrix_slice<N>&, Matrix_ref<T,N-1> 
}
