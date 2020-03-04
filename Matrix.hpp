#include <vector>
#include <array>
#include <numeric>
#include <initializer_list>
#include <type_traits>

//TODO: fix this Convertible return
template<typename T, typename U>
using Convertible = typename std::is_convertible<T,U>::value;

template<bool B, typename T = void>
using Enable_if = typename std::enable_if<B,T>::type;

namespace Matrix_impl {
  
  template<size_t N, typename List>
  std::array<size_t,N> derive_extents (const List& list)
  {
    std::array<size_t,N> a;
    auto first = a.begin();
    add_extents<N> (first, list);
    return a;
  }
  
  template<size_t N, typename I, typename List>
  Enable_if<(N>1),void> add_extents (I& first, const List& list)
  {
    assert (check_non_jagged<N>(list));
    *first++ = list.size();
    add_extents<N-1> (first, *list.begin());
  }
  
  template<size_t N, typename I, typename List>
  Enable_if<(N==1),void> add_extents (I& first, const List& list)
  {
    *first++ = list.size();
  }
  
  template <size_t N, typename List>
  bool check_non_jagged (const List& list)
  {
    auto i = list.begin();
    for (auto j=i+1; j!=list.end(); ++i)
      if (derive_extents<N-1>(*i) != derive_extents<N-1>(*j)
        return false;
      return true;
  }
  
  template<int N>
  void compute_strides (Matrix_slice<N>& ms)
  {
    size_t stride = 1;
    for (int i=N; i>=0; --i) {
      ms.strides[i] = stride;
      stride *= ms.extents[i];
    }
    ms.size = stride;
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
  // End of Matrix_impl namespace
}




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
  
  Matrix_slice (size_t offset, std::initializer_list<size_t> extents);
  Matrix_slice (size_t offset, std::initializer_list<size_t> extents,
                std::initializer_list<size_t> strs);
  
  template<typename... Dims>
  Matrix_slice (Dims... dims);
  
  template<typename... Dims, Enable_if<All(Convertible<Dims, size_t()...)>>
  size_t operator()(Dims... dims) const;
  
  size_t size;
  size_t start;
  std::array<size_t,N> extents;
  std::array<size_t,N> strides;
};

template<size_t N>
template<typename... Dims>
size_t Matrix_slice<N>::operator()(Dims... dims) const
{
  static_assert(sizeof...(Dims)==N, "dimension mismatch");
  
  size_t args[N] {size_t(dims)...};
  
  return start + std::inner_product(args, args+N, strides.begin(), size_t{0});
}

template<>
struct Matrix_slice<1> {
  size_t operator()(size_t i) const {return i;}
};

template<>
struct Matrix_slice<2> {
  size_t operator()(size_t i, size_t j) const
  {
    return start + i*strides[0] + j;
  }
};





// ***** Matrix_init *****//
template<typename T, size_t N>
struct Matrix_init {
  using type = std::initializer_list<typename Matrix_init<T,N-1>::type>
};

template<typename T>
struct Matrix_init<T,1> {
  using type = std::initializer_list<T>;
};

template<typename T>
struct Matrix_init<T,0>;

template<typename T, size_t N>
using Matrix_initializer = typename Matrix_init<T,N>::type;



// ***** Matrix_base *****//
template<typename T, size_t N>
class Matrix_base {
public:
  static constexpr size_t order = N;
  using value_type = T;
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;
  

};



// ***** Matrix_ref *****//
template<typename T, size_t N>
class Matrix_ref : public Matrix_base {
public:
  Matrix_ref (const Matrix_slice<N>& s, T* p) : desc{s}, ptr{p} {}
  // TODO: complete the body
private:
  Matrix_slice<N> desc;
  T* ptr;
};



// ***** Matrix *****//
template<typename T, size_t N>
class Matrix : public Matrix_base<T,N> {
public:
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
    Enable_if<Matrix_impl::Requesting_elements<Args...>(), T&>
    operator()(Args... args);
  template<typename... Args>
    Enable_if<Matrix_impl::Requesting_elements<Args...>(), const T&>
    operator()(Args... args) const;
  
  // m(s1,s2,s3) subscripting with slices
  template<typename... Args>
    Enable_if<Matrix_impl::Requesting_slice<Args...>(), Matrix_ref<T,N>>
    operator()(const Args&... args);
  template<typename... Args>
    Enable_if<Matrix_impl::Requesting_slice<Args...>(), Matrix_ref<const T,N>>
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
    Enable_if<Matrix_type<M>(), Matrix&> apply (const M&, F);
    
  Matrix& operator=(const T& value)  {return apply([&](T& a){a=value;});}
  Matrix& operator+=(const T& value) {return apply([&](T& a){a+=value;});}
  Matrix& operator-=(const T& value) {return apply([&](T& a){a-=value;});}
  Matrix& operator*=(const T& value) {return apply([&](T& a){a*=value;});}
  Matrix& operator/=(const T& value) {return apply([&](T& a){a/=value;});}
  Matrix& operator%=(const T& value) {return apply([&](T& a){a%=value;});}
  
  template<typename M>
    Enable_if<Matrix_type<M>(),Matrix&> operator+=(const M&);
  template<typename M>
    Enable_if<Matrix_type<M>(),Matrix&> operator-=(const M&);
private:
  Matrix_slice<N> desc;
  std::vector<T> elems;
};

template<typename T, size_t N>
  template<typename... Exts>
  Matrix<T,N>::Matrix(Exts... extents) : desc{extents...}, elems{desc.size} {}

template<typename T, size_t N>
Matrix<T,N>::Matrix(Matrix_initializer<T,N> init) {
  desc.extents = Matrix_impl::derive_extents(init);
  Matrix_impl::compute_strides (desc);
  elems.reserve (desc.size);
  Matrix_impl::insert_flat (init, elems);
  assert (elems.size() = desc.size);
}

template<typename T, size_t N>
template<typename U>
Matrix<T,N>::Matrix (const Matrix_ref<U,N>& x)
: desc{x.desc}, elems{x.begin(), x.end()}
{
  static_assert(Convertible<U,T>(), "Can not convert: Incompatible types");
}

template<typename T, size_t N>
template<typename U>
Matrix<T,N>& Matrix<T,N>::operator=(const Matrix_ref<U,N>& x)
{
  static_assert(Convertible<U,T>(), "Can not convert: Incompatible types");
  desc = x.desc;
  elems.assign (x.begin(), x.end());
  return *this;
}

template<typename t, size_t N>
  template<typename F>
  Matrix<T,N>& Matrix<T,N>::apply(F f)
  {
    for (auto& x : elems) f(x);
    return *this;
  }
  
template<typename t, size_t N>
  template<typename M, typename F>
  Enable_if<Matrix_type<M>,Matrix<T,N>&> Matrix<T,N>::apply(M& m, F f)
  {
    assert(same_extents(desc, m.descriptor());
    for (auto i=begin(),j=m.begin(); i!=end(); ++i, ++j)
      f(*i, *j);
    return *this;
  }
  
template<typename T, size_t N>
  template<typename M>
  Enable_if<Matrix_type<M>(), Matrix<T,N>&> Matrix<T,N>::operator+=(const M& m)
  {
    static_assert (m.order==N, "Matrix dimensions do not match");
    assert(same_extents(desc,m.descriptor()));
    return apply(m, [](T& a, const value_type<M>& b){a+=b});
  }
  
  
  
// Matrix operations
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
      for (size_t j=0; j!m; ++j)
        res(i,j) = u[i]*v[j];
    return res;
  } // TODO: check the overhead discussed in page 839
  
  // Multiply N by M matrix m by a M by 1 vector v, returns a N by 1 vector
  template<typename T>
  Matrix<T,1> operator*(const Matrix<T,1>& u, const Matrix<T,1>& v)
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
  
}
