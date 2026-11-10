# Generates the reference fixtures read by `faer/tests/gevd_reference_fixtures.rs`
# (tensor4all/faer#234, item 1). Run once, offline, and check the output in:
#
#     julia --startup-file=no faer/tests/fixtures/gevd_reference/generate.jl
#
# The reference is the LAPACK that Julia's LinearAlgebra loads through
# libblastrampoline (OpenBLAS's bundled copy of Reference LAPACK). The fixtures
# in this directory were written by Julia 1.12.5 with OpenBLAS 0.3.29
# (OpenBLAS_jll 0.3.29+0, ILP64), whose `LAPACK.version()` is 3.12.0. The
# script prints the versions it ran with into every fixture header.
#
# For each real pencil (A, B) the fixture holds:
#
# - the pencil itself, written with 17 significant digits (exact round trip);
# - `dggev`'s alpha_re, alpha_im, beta and its right and left eigenvectors
#   (LAPACK conventions: `A x = lambda B x`, `y^H A = lambda y^H B`; a complex
#   pair is stored as the real and imaginary parts in columns j, j + 1);
# - RCONDE, the reciprocal condition number of each eigenvalue, by the formula
#   `dtgsna` uses: `sqrt(|y^H A x|^2 + |y^H B x|^2) / (|x|_2 |y|_2)`;
# - a clustering of the spectrum (eigenvalues within a relative 1e-5 of each
#   other, a complex value together with its conjugate) and, per cluster, an
#   orthonormal basis of its right deflating subspace and of its left one, from
#   `dgges` + `dtgsen` (Julia's `schur` / `ordschur`), together with
#   `min(Difu, Difl)` for that cluster against the rest of the spectrum,
#   computed exactly as the smallest singular value of the Kronecker matrix
#   that `dtgsen`'s documentation defines (not `dtgsen`'s estimate). The left
#   basis spans the left eigenvectors of the cluster, whatever their
#   conjugation convention (see `deflating`).
#
# The test compares against the conditioning, not elementwise: a vector in a
# repeated or nearly repeated cluster is not determined, its cluster's
# subspace is.
using LinearAlgebra, Printf, Random, OpenBLAS_jll

const OUT = @__DIR__
const CLUSTER_RTOL = 1e-5

fmt(x::Float64) = @sprintf("%.17g", x)
rowline(io, v) = println(io, join(fmt.(Float64.(v)), " "))
function writemat(io, name, M)
	println(io, name, " ", size(M, 1), " ", size(M, 2))
	for i in 1:size(M, 1)
		rowline(io, M[i, :])
	end
end

rot(a, b) = [a b; -b a]  # eigenvalues a +- b i

function blockdiag(blocks...)
	n = sum(size(b, 1) for b in blocks)
	M = zeros(n, n)
	o = 0
	for b in blocks
		k = size(b, 1)
		M[o+1:o+k, o+1:o+k] = b
		o += k
	end
	M
end

# a well-conditioned nonsingular transformation: orthogonal times a mild diagonal
function mild(rng, n)
	Q = Matrix(qr(randn(rng, n, n)).Q)
	Q * Diagonal(exp.(range(-0.7, 0.7, length = n)[randperm(rng, n)]))
end

# vectors from dggev's real storage, as complex columns
function complex_vectors(V, ai)
	n = size(V, 1)
	W = zeros(ComplexF64, n, n)
	j = 1
	while j <= n
		if ai[j] == 0
			W[:, j] = V[:, j]
			j += 1
		else
			W[:, j] = V[:, j] + im * V[:, j+1]
			W[:, j+1] = V[:, j] - im * V[:, j+1]
			j += 2
		end
	end
	W
end

function cluster_ids(lambda)
	n = length(lambda)
	id = zeros(Int, n)
	centers = ComplexF64[]
	for j in 1:n
		for (c, z) in enumerate(centers)
			for w in (z, conj(z))
				if abs(lambda[j] - w) <= CLUSTER_RTOL * max(1.0, abs(w))
					id[j] = c
				end
			end
			id[j] != 0 && break
		end
		if id[j] == 0
			push!(centers, lambda[j])
			id[j] = length(centers)
		end
	end
	id, centers
end

near(z, c) = any(abs(z - w) <= CLUSTER_RTOL * max(1.0, abs(w)) for w in (c, conj(c)))

# sigma_min of Zu = [kron(I, A11) -kron(A22', I); kron(I, B11) -kron(B22', I)]
function difu(A11, B11, A22, B22)
	n1, n2 = size(A11, 1), size(A22, 1)
	Zu = [kron(I(n2), A11) -kron(transpose(A22), I(n1)); kron(I(n2), B11) -kron(transpose(B22), I(n1))]
	minimum(svdvals(Zu))
end

# orthonormal bases of the right and left deflating subspaces of the
# eigenvalues of (A, B) near `c`, and min(Difu, Difl) of that cluster against the
# rest. Ordered first, the cluster's right subspace is the leading columns of
# Z; ordered last, its left eigenvectors lie in the trailing columns of Q (if
# w^H S = lambda w^H T with lambda outside S11, the leading part of w is zero).
# One QZ serves both sides: a QZ of (A^T, B^T) would split a long Jordan chain
# by eps^(1/k) and lose the cluster.
function deflating(A, B, c)
	F = schur(A, B)
	n = size(A, 1)
	sel = [near(F.alpha[i] / F.beta[i], c) for i in 1:n]
	k = count(sel)
	R = ordschur(F, sel)
	L = ordschur(F, .!sel)
	dif = if k == n
		-1.0  # no complement
	else
		S, T = R.S, R.T
		r1, r2 = 1:k, k+1:n
		min(difu(S[r1, r1], T[r1, r1], S[r2, r2], T[r2, r2]),
			difu(S[r2, r2], T[r2, r2], S[r1, r1], T[r1, r1]))
	end
	R.Z[:, 1:k], L.Q[:, n-k+1:n], dif
end

# sin of the largest principal angle between span(U) and span(W), W orthonormal
function sin_angle(U, W)
	Q = Matrix(qr(U).Q)[:, 1:size(U, 2)]
	opnorm(Q - W * (W' * Q))
end

function realspan(v)
	isreal(v) ? reshape(real(v), :, 1) : hcat(real(v), imag(v))
end

# `defective` lists the eigenvalues that are defective by construction (a
# Jordan pair before the pencil was rounded); every other cluster is semisimple
# by construction. The flag is not read off the computed vectors: rounding
# splits a Jordan pair by about sqrt(eps), and the reference's basis then sits
# right at any sqrt(eps) rank threshold (printed below as sigma_min).
function write_fixture(name, description, A, B; defective = ComplexF64[])
	n = size(A, 1)
	alphar, alphai, beta, VL, VR = LAPACK.ggev!('V', 'V', copy(A), copy(B))
	alpha = complex.(alphar, alphai)
	lambda = alpha ./ beta
	X = complex_vectors(VR, alphai)
	Y = complex_vectors(VL, alphai)
	rconde = [sqrt(abs(Y[:, j]' * A * X[:, j])^2 + abs(Y[:, j]' * B * X[:, j])^2) /
			  (norm(X[:, j]) * norm(Y[:, j])) for j in 1:n]
	id, centers = cluster_ids(lambda)
	abnorm = hypot(opnorm(A, 1), opnorm(B, 1))
	path = joinpath(OUT, name * ".txt")
	open(path, "w") do io
		println(io, "# ", description)
		println(io, "# generated by faer/tests/fixtures/gevd_reference/generate.jl")
		println(io, "# julia ", VERSION, ", OpenBLAS_jll ", pkgversion(OpenBLAS_jll),
			", LAPACK ", LAPACK.version(), " (dggev, dgges, dtgsen)")
		println(io, "name ", name)
		println(io, "n ", n)
		writemat(io, "a", A)
		writemat(io, "b", B)
		println(io, "alpha_re ", n); rowline(io, alphar)
		println(io, "alpha_im ", n); rowline(io, alphai)
		println(io, "beta ", n); rowline(io, beta)
		writemat(io, "vr", VR)
		writemat(io, "vl", VL)
		println(io, "rconde ", n); rowline(io, rconde)
		println(io, "cluster_of ", n); println(io, join(id .- 1, " "))
		println(io, "clusters ", length(centers))
		println("== $name (n = $n, ||(A,B)|| = $(fmt(abnorm)))")
		for (c, z) in enumerate(centers)
			Wr, Wl, dif = deflating(A, B, z)
			members = findall(==(c), id)
			# the reference's own vectors against the reference subspaces, and
			# the conditioning of its vector basis (tiny when defective)
			ref_r = maximum(sin_angle(realspan(X[:, j]), Wr) for j in members)
			ref_l = maximum(sin_angle(realspan(Y[:, j]), Wl) for j in members)
			basis(V) = (M = reduce(hcat, [realspan(V[:, j] / norm(V[:, j])) for j in members if imag(lambda[j]) >= 0]);
						minimum(svdvals(M)))
			smin_r, smin_l = basis(X), basis(Y)
			semisimple = !any(near(z, d) for d in defective)
			@printf("  cluster %d: lambda ~ %s, dim %d, dif %.3e, min rconde %.3e, ref angle r/l %.1e/%.1e, ref basis sigma_min r/l %.1e/%.1e, semisimple %s\n",
				c - 1, string(round(z, sigdigits = 6)), size(Wr, 2), dif,
				minimum(rconde[members]), ref_r, ref_l, smin_r, smin_l, semisimple)
			println(io, "cluster ", c - 1, " dim ", size(Wr, 2), " dif ", fmt(dif),
				" semisimple ", semisimple ? 1 : 0)
			writemat(io, "right_basis", Wr)
			writemat(io, "left_basis", Wl)
		end
	end
	println("  wrote ", path)
end

# faer#220 / #233: diag(R, R, 3) with R = [[2, 1], [-1, 2]] and B = I, already
# in generalized Schur form; `coupling` at (1, 3) (0-based) makes it defective
function schur_pencil(blocks, coupling)
	n = 2 * blocks + 1
	A = zeros(n, n)
	for k in 0:blocks-1
		o = 2k
		A[o+1:o+2, o+1:o+2] = rot(2.0, 1.0)
		k + 1 < blocks && (A[o+2, o+4] = coupling)
	end
	A[n, n] = 3.0
	A, Matrix(1.0I, n, n)
end

rng = MersenneTwister(234)

let (A, B) = schur_pencil(2, 0.0)
	write_fixture("semisimple_schur",
		"semisimple repeated pair 2 +- i, diag(R, R, 3), B = I (faer#220, uncoupled)", A, B)
end
let (A, B) = schur_pencil(2, 0.75)
	write_fixture("defective_schur",
		"defective repeated pair 2 +- i, R coupled to R by 0.75 at (1, 3), B = I (faer#220 report)", A, B;
		defective = [2.0 + 1.0im])
end
let (A, B) = schur_pencil(22, 0.75)
	write_fixture("defective_chain22_schur",
		"22 coupled copies of 2 +- i, B = I (faer#233 f64 overflow chain)", A, B;
		defective = [2.0 + 1.0im])
end
let
	A0 = blockdiag(rot(2.0, 1.0), rot(2.0, 1.0), fill(3.0, 1, 1), rot(-1.0, 0.5), fill(-2.0, 1, 1), fill(0.5, 1, 1))
	n = size(A0, 1)
	B0 = Matrix(1.0I, n, n)
	X, Y = mild(rng, n), mild(rng, n)
	write_fixture("semisimple_transformed",
		"X * (diag(R, R, 3, R', -2, 0.5), I) * Y, R = [[2, 1], [-1, 2]], R' = [[-1, 0.5], [-0.5, -1]], X, Y orthogonal * mild diagonal (seed 234)",
		X * A0 * Y, X * B0 * Y)
end
let
	C = zeros(4, 4)
	C[1:2, 1:2] = rot(2.0, 1.0)
	C[3:4, 3:4] = rot(2.0 + 1e-6, 1.0)
	C[2, 4] = 0.75
	A0 = blockdiag(C, fill(3.0, 1, 1), rot(-1.0, 0.5), fill(-2.0, 1, 1), fill(0.5, 1, 1))
	n = size(A0, 1)
	B0 = Matrix(1.0I, n, n)
	X, Y = mild(rng, n), mild(rng, n)
	write_fixture("near_defective_transformed",
		"X * (A0, I) * Y, A0 holds 2 +- i and 2 + 1e-6 +- i coupled by 0.75, then 3, -1 +- 0.5i, -2, 0.5 (seed 234)",
		X * A0 * Y, X * B0 * Y)
end
let
	n = 12
	A = randn(rng, n, n)
	B = randn(rng, n, n)
	write_fixture("random12", "control: A, B with independent standard normal entries (seed 234)", A, B)
end
let
	C = zeros(4, 4)
	C[1:2, 1:2] = rot(2.0, 1.0)
	C[3:4, 3:4] = rot(2.0, 1.0)
	C[2, 4] = 0.75
	A0 = blockdiag(C, fill(3.0, 1, 1), rot(-1.0, 0.5), fill(-2.0, 1, 1), fill(0.5, 1, 1))
	n = size(A0, 1)
	B0 = Matrix(1.0I, n, n)
	X, Y = mild(rng, n), mild(rng, n)
	write_fixture("defective_transformed",
		"X * (A0, I) * Y, A0 holds 2 +- i twice coupled by 0.75 (a Jordan pair before rounding), then 3, -1 +- 0.5i, -2, 0.5 (seed 234)",
		X * A0 * Y, X * B0 * Y; defective = [2.0 + 1.0im])
end
