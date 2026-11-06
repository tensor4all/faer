use faer::diag::Diag;
use faer::traits::ComplexField;
use faer::{Col, Mat, Row, c32, c64};
use std::sync::Mutex;

fn native_capacity<T: ComplexField + Copy>(value: T, fill: T) {
	let mut a = Mat::<T>::with_capacity(11, 13);
	assert_eq!(a.shape(), (0, 0));
	assert!(a.row_capacity() >= 11);
	assert!(a.col_capacity() >= 13);
	a.resize_with(3, 4, |_, _| value);
	a.try_reserve(17, 19).unwrap();
	let capacities = (a.row_capacity(), a.col_capacity());
	assert!(capacities.0 >= 17 && capacities.1 >= 19);
	assert_eq!(a.col_stride() as usize, capacities.0);
	let pointer = a.as_ptr();
	for _ in 0..4 {
		assert_eq!((a.row_capacity(), a.col_capacity()), capacities);
		assert_eq!(a.as_ptr(), pointer);
		assert_eq!(a.shape(), (3, 4));
	}
	a.try_reserve(0, 0).unwrap();
	assert_eq!((a.row_capacity(), a.col_capacity()), capacities);
	a.truncate(1, 2);
	assert_eq!((a.row_capacity(), a.col_capacity()), capacities);
	a.resize_with(17, 19, |_, _| fill);
	assert_eq!(a.as_ptr(), pointer);
	assert_eq!((a.row_capacity(), a.col_capacity()), capacities);
	assert_eq!(a[(0, 0)], value);
	assert_eq!(a[(16, 18)], fill);

	let mut column = Col::<T>::full(3, value);
	column.try_reserve(17).unwrap();
	let capacity = column.capacity();
	let pointer = column.as_ptr();
	assert!(capacity >= 17);
	column.try_reserve(0).unwrap();
	column.truncate(1);
	assert_eq!(column.capacity(), capacity);
	column.resize_with(17, |_| fill);
	assert_eq!(column.capacity(), capacity);
	assert_eq!(column.as_ptr(), pointer);
	assert_eq!(column[0], value);
	assert_eq!(column[16], fill);
	let row = column.into_transpose();
	assert_eq!(row.capacity(), capacity);
	assert_eq!(row.as_ptr(), pointer);
	let diagonal = row.into_diagonal();
	assert_eq!(diagonal.capacity(), capacity);
	assert_eq!(diagonal.column_vector().as_ptr(), pointer);
	let column = diagonal.into_column_vector();
	assert_eq!(column.capacity(), capacity);
	assert_eq!(column.as_ptr(), pointer);

	let mut row = Row::<T>::full(3, value);
	row.try_reserve(19).unwrap();
	let capacity = row.capacity();
	let pointer = row.as_ptr();
	assert!(capacity >= 19);
	row.try_reserve(0).unwrap();
	row.truncate(1);
	row.resize_with(19, |_| fill);
	assert_eq!(row.capacity(), capacity);
	assert_eq!(row.as_ptr(), pointer);
	assert_eq!(row[0], value);
	assert_eq!(row[18], fill);
}

#[test]
fn native_scalar_capacity_preserves_storage() {
	native_capacity::<f32>(2.0, 3.0);
	native_capacity::<f64>(2.0, 3.0);
	native_capacity(c32::new(2.0, -1.0), c32::new(-3.0, 4.0));
	native_capacity(c64::new(2.0, -1.0), c64::new(-3.0, 4.0));
}

#[test]
fn empty_and_zero_extent_capacities() {
	let empty = Mat::<f64>::new();
	assert_eq!((empty.row_capacity(), empty.col_capacity()), (0, 0));
	for (rows, cols) in [(0, 0), (0, 7), (9, 0)] {
		let mut a = Mat::<f64>::with_capacity(rows, cols);
		assert!(a.row_capacity() >= rows && a.col_capacity() >= cols);
		let capacities = (a.row_capacity(), a.col_capacity());
		a.resize_with(rows, cols, |_, _| 1.0);
		assert_eq!((a.row_capacity(), a.col_capacity()), capacities);
		assert_eq!(a.shape(), (rows, cols));
	}
	assert_eq!(Col::<f64>::zeros(0).capacity(), 0);
	assert_eq!(Row::<f64>::zeros(0).capacity(), 0);
	assert_eq!(Diag::<f64>::zeros(0).capacity(), 0);
}

#[test]
fn typed_shapes_preserve_capacities() {
	faer::with_dim!(m, 3);
	faer::with_dim!(n, 4);
	let mut a = Mat::<f64>::with_capacity(17, 19);
	a.resize_with(3, 4, |_, _| 2.0);
	let capacities = (a.row_capacity(), a.col_capacity());
	let pointer = a.as_ptr();
	let a = a.into_shape(m, n);
	assert_eq!((a.row_capacity(), a.col_capacity()), capacities);
	assert_eq!(a.as_ptr(), pointer);
	let mut column = Col::<f64>::zeros(3);
	column.reserve(17);
	let capacity = column.capacity();
	let column = column.into_row_shape(m);
	assert_eq!(column.capacity(), capacity);
	let row = column.into_transpose();
	assert_eq!(row.capacity(), capacity);
	let diagonal = row.into_diagonal();
	assert_eq!(diagonal.capacity(), capacity);
}

#[test]
fn zero_sized_elements_reserve_without_panicking() {
	// the row padding that keeps every column aligned divides by
	// `size_of::<T>()`, which is zero here (tensor4all/faer#286)
	let empty = Mat::<()>::with_capacity(0, 0);
	assert_eq!(empty.shape(), (0, 0));

	let mut a = Mat::<()>::with_capacity(1, 1);
	assert_eq!(a.shape(), (0, 0));
	assert!(a.row_capacity() >= 1 && a.col_capacity() >= 1);
	a.resize_with(1, 1, |_, _| ());
	assert_eq!(a.shape(), (1, 1));
	a.try_reserve(9, 7).unwrap();
	a.resize_with(9, 7, |_, _| ());
	assert_eq!(a.shape(), (9, 7));

	let mut b = Mat::<f64>::with_capacity(1, 1);
	assert_eq!(b.shape(), (0, 0));
	b.resize_with(1, 1, |_, _| 4.0);
	assert_eq!(b[(0, 0)], 4.0);
}

#[test]
fn cleanup_of_aligned_and_zero_sized_elements_uses_a_byte_stride() {
	// `DropMat` advances its column pointer in bytes, so it needs
	// `row_capacity * size_of::<T>()` here, not the element count
	// (tensor4all/faer#286)
	static DROPPED: Mutex<Vec<u32>> = Mutex::new(Vec::new());

	#[repr(align(8))]
	struct Aligned(u32);
	impl Drop for Aligned {
		fn drop(&mut self) {
			DROPPED.lock().unwrap().push(self.0);
		}
	}
	let mut a = Mat::from_fn(1, 3, |_, j| Aligned(j as u32));
	a.truncate(1, 0);
	let mut dropped = DROPPED.lock().unwrap().clone();
	dropped.sort_unstable();
	assert_eq!(dropped, [0, 1, 2]);

	DROPPED.lock().unwrap().clear();
	// a zero-sized element keeps every column at the same address
	#[repr(align(64))]
	struct AlignedZst;
	impl Drop for AlignedZst {
		fn drop(&mut self) {
			DROPPED.lock().unwrap().push(7);
		}
	}
	let mut b = Mat::from_fn(1, 3, |_, _| AlignedZst);
	assert_eq!(b.shape(), (1, 3));
	b.truncate(1, 0);
	assert_eq!(DROPPED.lock().unwrap().as_slice(), [7, 7, 7]);
}

#[test]
fn capacity_getters_do_not_require_clone_or_numeric_traits() {
	struct Opaque(u8);
	let a = Mat::from_fn(2, 3, |i, j| Opaque((i + j) as u8));
	assert!(a.row_capacity() >= 2 && a.col_capacity() >= 3);
	assert_eq!(a[(1, 2)].0, 3);
	let column = Col::from_fn(3, |_| Opaque(7));
	assert!(column.capacity() >= 3);
	let row = column.into_transpose();
	assert!(row.capacity() >= 3);
	let diagonal = row.into_diagonal();
	assert!(diagonal.capacity() >= 3);
	assert_eq!(diagonal.column_vector()[0].0, 7);
}
