#include <vector>
#include <array>
#include <initializer_list>


template<typename T, size_t N>
class Matrix_ref {};

template<typename T, size_t N>
class Matrix_initializer {};

template<typename T, typename U>
constexpr bool Convertible() {return true;}

namespace Matrix_impl {

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
template<size_t N>
struct Matrix_slice {
	Matrix_slice () = default;

	Matrix_slice (size_t offset, std::initializer_list<size_t> exts);
	Matrix_slice (size_t offset, std::initializer_list<size_t> exts,
			                     std::initializer_list<size_t> strs);

	template<typename... Dims>
		Matrix_slice (Dims... dims);

	template<typename... Dims, Enable_if<All(Convertible<Dims, size_t()...)>>
		size_t operator()(Dims... dims) const;

	size_t size;
	size_t start;
	std::array<size_t,N> extends;
	std::array<size_t,N> strides;
};

template<size_t N>
	template<typename... Dims>
	size_t Matrix_slice<N>::operator()(Dims... dims) const
	{
		static_assert(sizeof...(Dims)==N, "dimension mismatch");

		size_t args[N] {size_t(dims)...};

		return start+inner_product(args, args+N, strides.begin(), size_t{0});
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

	template<typename... Exts>
		explicit Matrix (Exts... exts);

	Matrix (Matrix_initializer<T,N>);
	Matrix& operator=(Matrix_initializer<T,N>);

	template<typename U>
		Matrix (std::initializer_list<U>) = delete;
	template<typename U>
		Matrix& operator=(std::initializer_list<U>) = delete;

	size_t extend (size_t n) const {return desc.extents[n];}
	size_t size () const {return elems.size();}
	const Matrix_slice<N>& descriptor() const {return desc;}

	T* data () {return elems.data();}
	const T* data () const {return elems.data();}
private:
	Matrix_slice<N> desc;
	std::vector<T> elems;
};

template<typename T, size_t N>
	template<typename... Exts>
	Matrix<T,N>::Matrix(Exts... exts) : desc{exts...}, elems{desc.size} {}

template<typename T, size_t N>
Matrix<T,N>::Matrix(Matrix_initializer<T,N> init) {
	//desc.extents = Matrix_impl::derive_extents(init);
	//Matrix_impl::compute_strides (desc);
	elems.reserve (desc.size);
	//Matrix_impl::insert_flat (init, elems);
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



int main(int argc, char **argv) {
	return 0;
}
