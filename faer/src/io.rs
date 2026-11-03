use crate::prelude::*;
/// npy format conversions
#[cfg(feature = "npy")]
pub mod npy {
	use super::*;
	/// memory view over a buffer in `npy` format
	pub struct Npy<'a> {
		aligned_bytes: &'a [u8],
		nrows: usize,
		ncols: usize,
		prefix_len: usize,
		dtype: NpyDType,
		fortran_order: bool,
		swap_bytes: bool,
	}
	/// data type of an `npy` buffer
	#[derive(Debug, Copy, Clone, PartialEq, Eq)]
	pub enum NpyDType {
		/// 32-bit floating point
		F32,
		/// 64-bit floating point
		F64,
		/// 32-bit complex floating point
		C32,
		/// 64-bit complex floating point
		C64,
		/// unknown type
		Other,
	}
	/// trait implemented for native types that can be read from a npy buffer
	pub trait FromNpy: bytemuck::Pod {
		/// data type of the buffer data
		const DTYPE: NpyDType;
		/// reads one value from its npy encoding
		///
		/// reverses the byte order of every real component when swap is set
		fn from_npy_bytes(bytes: &[u8], swap: bool) -> Self;
	}

	/// returns the native-order bytes of one npy value, reversing the source
	/// order when swap is set
	fn native_bytes<const N: usize>(bytes: &[u8], swap: bool) -> [u8; N] {
		let mut out = [0u8; N];
		out.copy_from_slice(&bytes[..N]);
		if swap {
			out.reverse();
		}
		out
	}

	impl FromNpy for f32 {
		const DTYPE: NpyDType = NpyDType::F32;
		fn from_npy_bytes(bytes: &[u8], swap: bool) -> Self {
			f32::from_ne_bytes(native_bytes::<4>(bytes, swap))
		}
	}

	impl FromNpy for f64 {
		const DTYPE: NpyDType = NpyDType::F64;
		fn from_npy_bytes(bytes: &[u8], swap: bool) -> Self {
			f64::from_ne_bytes(native_bytes::<8>(bytes, swap))
		}
	}

	impl FromNpy for c32 {
		const DTYPE: NpyDType = NpyDType::C32;
		fn from_npy_bytes(bytes: &[u8], swap: bool) -> Self {
			c32::new(
				f32::from_ne_bytes(native_bytes::<4>(bytes, swap)),
				f32::from_ne_bytes(native_bytes::<4>(&bytes[4..], swap)),
			)
		}
	}

	impl FromNpy for c64 {
		const DTYPE: NpyDType = NpyDType::C64;
		fn from_npy_bytes(bytes: &[u8], swap: bool) -> Self {
			c64::new(
				f64::from_ne_bytes(native_bytes::<8>(bytes, swap)),
				f64::from_ne_bytes(native_bytes::<8>(&bytes[8..], swap)),
			)
		}
	}
	impl<'a> Npy<'a> {
		fn parse_npyz(
			data: &[u8],
			npyz: npyz::NpyFile<&[u8]>,
		) -> Result<(NpyDType, usize, usize, usize, bool, bool), std::io::Error> {
			let ver_major = data[6] - b'\x00';
			let length = if ver_major <= 1 {
				2usize
			} else if ver_major <= 3 {
				4usize
			} else {
				return Err(std::io::Error::new(
					std::io::ErrorKind::Other,
					"unsupported version",
				));
			};
			let header_len = if length == 2 {
				u16::from_le_bytes(data[8..10].try_into().unwrap()) as usize
			} else {
				u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize
			};
			let dtype = || -> NpyDType {
				match npyz.dtype() {
					npyz::DType::Plain(str) => {
						let is_complex = match str.type_char() {
							npyz::TypeChar::Float => false,
							npyz::TypeChar::Complex => true,
							_ => return NpyDType::Other,
						};
						let byte_size = str.size_field();
						if byte_size == 8 && is_complex {
							NpyDType::C32
						} else if byte_size == 16 && is_complex {
							NpyDType::C64
						} else if byte_size == 4 && !is_complex {
							NpyDType::F32
						} else if byte_size == 8 && !is_complex {
							NpyDType::F64
						} else {
							NpyDType::Other
						}
					},
					_ => NpyDType::Other,
				}
			};
			let dtype = dtype();
			let swap_bytes = match npyz.dtype() {
				npyz::DType::Plain(str) => match str.endianness() {
					npyz::Endianness::Little => cfg!(target_endian = "big"),
					npyz::Endianness::Big => cfg!(target_endian = "little"),
				npyz::Endianness::Irrelevant => false,
			},
				_ => false,
			};
			let order = npyz.header().order();
			let shape = npyz.shape();
			let nrows = shape.get(0).copied().unwrap_or(1) as usize;
			let ncols = shape.get(1).copied().unwrap_or(1) as usize;
			let prefix_len = 8 + length + header_len;
			let fortran_order = order == npyz::Order::Fortran;
			Ok((dtype, nrows, ncols, prefix_len, fortran_order, swap_bytes))
		}

		/// parse a npy file from a memory buffer
		#[inline]
		pub fn new(data: &'a [u8]) -> Result<Self, std::io::Error> {
			let npyz = npyz::NpyFile::new(data)?;
			let (dtype, nrows, ncols, prefix_len, fortran_order, swap_bytes) =
				Self::parse_npyz(data, npyz)?;
			Ok(Self {
				aligned_bytes: data,
				prefix_len,
				nrows,
				ncols,
				dtype,
				fortran_order,
				swap_bytes,
			})
		}

		/// returns the data type of the memory buffer
		#[inline]
		pub fn dtype(&self) -> NpyDType {
			self.dtype
		}

		/// checks if the memory buffer is aligned, in which case the data can
		/// be referenced in-place
		#[inline]
		pub fn is_aligned(&self) -> bool {
			self.aligned_bytes.as_ptr().align_offset(64) == 0
		}

		/// checks whether the buffer stores its values in the target's byte order
		///
		/// a buffer with a different byte order cannot be viewed in place; to_mat
		/// copies and converts it instead
		#[inline]
		pub fn is_native_endian(&self) -> bool {
			!self.swap_bytes
		}

		/// if the memory buffer is aligned, and the provided type matches the
		/// one stored in the buffer, returns a matrix view over the data
		#[inline]
		pub fn as_aligned_ref<T: FromNpy>(&self) -> MatRef<'_, T> {
			assert!(self.is_aligned());
			assert!(!self.swap_bytes, "cannot view a non-native-endian npy buffer in place, use to_mat");
			assert!(self.dtype == T::DTYPE);
			if self.fortran_order {
				MatRef::from_column_major_slice(
					bytemuck::cast_slice(
						&self.aligned_bytes[self.prefix_len..],
					),
					self.nrows,
					self.ncols,
				)
			} else {
				MatRef::from_row_major_slice(
					bytemuck::cast_slice(
						&self.aligned_bytes[self.prefix_len..],
					),
					self.nrows,
					self.ncols,
				)
			}
		}

		/// if the provided type matches the one stored in the buffer, returns a
		/// matrix containing the data
		#[inline]
		pub fn to_mat<T: FromNpy>(&self) -> Mat<T> {
		assert!(self.dtype == T::DTYPE);
		let mut mat = Mat::<T>::with_capacity(self.nrows, self.ncols);
		unsafe { mat.set_dims(self.nrows, self.ncols) };
		let data = &self.aligned_bytes[self.prefix_len..];
		let size = core::mem::size_of::<T>();
		let swap = self.swap_bytes;
		for j in 0..self.ncols {
			for i in 0..self.nrows {
				// read byte by byte rather than through an aligned cast
				let offset = if self.fortran_order {
					(j * self.nrows + i) * size
				} else {
					(i * self.ncols + j) * size
				};
				mat[(i, j)] = T::from_npy_bytes(&data[offset..offset + size], swap);
			}
		}
		mat
	}
	}

	#[cfg(test)]
	mod tests {
		use super::*;

		#[repr(align(64))]
		struct Aligned([u8; 1024]);

		fn build(descr: &str, fortran: bool, shape: &str, payload: &[u8]) -> Vec<u8> {
			let mut header = format!(
				"{{'descr': '{}', 'fortran_order': {}, 'shape': {}, }}",
				descr,
				if fortran { "True" } else { "False" },
				shape
			);
			let padding = (64 - ((10 + header.len() + 1) % 64)) % 64;
			header.push_str(&" ".repeat(padding));
			header.push('\n');
			let mut file = b"\x93NUMPY\x01\x00".to_vec();
			file.extend_from_slice(&(header.len() as u16).to_le_bytes());
			file.extend_from_slice(header.as_bytes());
			file.extend_from_slice(payload);
			file
		}

		fn native_marker() -> char {
			if cfg!(target_endian = "little") {
				'<'
			} else {
				'>'
			}
		}

		fn foreign_marker() -> char {
			if cfg!(target_endian = "little") {
				'>'
			} else {
				'<'
			}
		}

		fn f64_bytes(value: f64, foreign: bool) -> Vec<u8> {
			if foreign == cfg!(target_endian = "little") {
				value.to_be_bytes().to_vec()
			} else {
				value.to_le_bytes().to_vec()
			}
		}

		fn descr(char_: char) -> String {
			format!("{}f8", char_)
		}

		// #211: the copying API must not require an aligned source
		#[test]
		fn to_mat_reads_unaligned_c_order() {
			let payload = f64_bytes(1.5, false);
			let file = build(&descr(native_marker()), false, "(1, 1)", &payload);
			let mut storage = vec![0u8; file.len() + 1];
			storage[1..].copy_from_slice(&file);
			let npy = Npy::new(&storage[1..]).unwrap();
			let mat = npy.to_mat::<f64>();
			core::assert!((mat[(0, 0)] - 1.5).abs() < 1e-15);
		}

		// #211: the Fortran-order path copies into aligned storage already
		#[test]
		fn to_mat_reads_unaligned_fortran_order() {
			let payload = f64_bytes(2.5, false);
			let file = build(&descr(native_marker()), true, "(1, 1)", &payload);
			let mut storage = vec![0u8; file.len() + 1];
			storage[1..].copy_from_slice(&file);
			let npy = Npy::new(&storage[1..]).unwrap();
			let mat = npy.to_mat::<f64>();
			core::assert!((mat[(0, 0)] - 2.5).abs() < 1e-15);
		}

		// #210: a foreign byte order must be converted, not reinterpreted
		#[test]
		fn to_mat_converts_foreign_endian() {
			let payload = f64_bytes(1.5, true);
			let file = build(&descr(foreign_marker()), false, "(1, 1)", &payload);
			let mut storage = Aligned([0; 1024]);
			storage.0[..file.len()].copy_from_slice(&file);
			let npy = Npy::new(&storage.0[..file.len()]).unwrap();
			core::assert!(npy.is_aligned());
			core::assert!(!npy.is_native_endian());
			let mat = npy.to_mat::<f64>();
			core::assert!((mat[(0, 0)] - 1.5).abs() < 1e-15);
		}

		// #210: a foreign byte order cannot be viewed in place
		#[test]
		#[should_panic]
		fn as_aligned_ref_rejects_foreign_endian() {
			let payload = f64_bytes(1.5, true);
			let file = build(&descr(foreign_marker()), false, "(1, 1)", &payload);
			let mut storage = Aligned([0; 1024]);
			storage.0[..file.len()].copy_from_slice(&file);
			let npy = Npy::new(&storage.0[..file.len()]).unwrap();
			core::assert!(npy.is_aligned());
			let _ = npy.as_aligned_ref::<f64>();
		}

		// the native byte order still works through the zero-copy path
		#[test]
		fn as_aligned_ref_reads_native_endian() {
			let payload = f64_bytes(3.5, false);
			let file = build(&descr(native_marker()), false, "(1, 1)", &payload);
			let mut storage = Aligned([0; 1024]);
			storage.0[..file.len()].copy_from_slice(&file);
			let npy = Npy::new(&storage.0[..file.len()]).unwrap();
			core::assert!(npy.is_aligned());
			core::assert!(npy.is_native_endian());
			core::assert!((npy.as_aligned_ref::<f64>()[(0, 0)] - 3.5).abs() < 1e-15);
		}
	}
}
