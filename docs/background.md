# Background: what SDPB solves

This page fixes notation used throughout the API. The authoritative
references are the SDPB manual (`docs/SDPB_Manual` in the submodule),
[arXiv:1502.02033](https://arxiv.org/abs/1502.02033) and
[arXiv:1909.09745](https://arxiv.org/abs/1909.09745).

## Polynomial matrix programs

A *polynomial matrix program* (PMP) is, in the form of eq. (3.1) of the manual,

$$
\begin{aligned}
&\text{maximize} && a \cdot z \quad \text{over } z \in \mathbb{R}^{N+1} \\
&\text{such that} && n \cdot z = 1, \\
& && \sum_{i=0}^{N} z_i\, M^j_i(x) \succeq 0 \quad \text{for all } x \ge 0,\ j = 1, \dots, J .
\end{aligned}
$$

Each $M^j_i(x)$ is a symmetric $m_j \times m_j$ matrix of polynomials in $x$.
In sdpb-python:

- {class}`~sdpb_python.PMP` holds `objective` $= a$, `normalization` $= n$, and
  `matrices`.
- {class}`~sdpb_python.PolynomialMatrix` holds one $j$: `polynomials[r][s]`
  is the list $(P^{rs}_0, \dots, P^{rs}_N)$ of the $(r, s)$ entries of
  $M^j_0, \dots, M^j_N$.
- A {class}`~sdpb_python.DampedRational` *prefactor* $\chi(x) = c\, b^x / \prod_k (x - p_k)$
  may multiply a matrix. It does not change the constraint (it is positive on
  $x \ge 0$) but tells SDPB how the entries decay, which determines where the
  polynomial is sampled. The default is $e^{-x}$.

### Normalization and the dual variables

SDPB eliminates the normalization: with $k = \arg\max_i |n_i|$ it sets
$z_k = (1 - \sum_{i \ne k} n_i z_i) / n_k$ and works with the remaining $N$
components, called $y$. The objective becomes $b_0 + b \cdot y$. In a
{class}`~sdpb_python.Solution`:

- `y` are these $N$ dual variables,
- `z` is the full $z \in \mathbb{R}^{N+1}$ reconstructed from `y` and the
  normalization (`None` if the PMP has no normalization, in which case
  $z = (1, y)$).

### From polynomials to a semidefinite program

Positivity of a polynomial matrix on $x \ge 0$ is imposed by sampling it at
$d_j + 1$ points $x_k$ (where $d_j$ is the maximal degree in block $j$) and
expressing it through *bilinear bases*: orthogonal polynomials $q_m(x)$ for the
even part and $\sqrt{x}\, q_m(x)$ for the odd part. The result is the SDP of
eq. (2.2),

$$
\begin{aligned}
\text{Dual:} \quad & \text{maximize } b_0 + b \cdot y \quad \text{s.t. } \operatorname{Tr}(A_p Y) + (B y)_p = c_p,\ Y \succeq 0, \\
\text{Primal:} \quad & \text{minimize } b_0 + c \cdot x \quad \text{s.t. } X = \sum_p A_p x_p,\ B^T x = b,\ X \succeq 0 .
\end{aligned}
$$

$X$ and $Y$ are block diagonal with two blocks (even and odd parity) per
polynomial matrix. {meth}`PMP.to_sdp <sdpb_python.PMP.to_sdp>` returns
exactly this data ($b_0$, $b$, and per block $c$, $B$ and the sampled bases);
{meth}`PolynomialMatrix.sampled <sdpb_python.PolynomialMatrix.sampled>`
returns the sample points, scalings and bases for one block.

### What the solver returns

SDPB runs a primal-dual interior point method on $(x, X, y, Y)$. When it stops
it reports:

| Quantity | Meaning |
|---|---|
| `primal_objective` | $b_0 + c \cdot x$ |
| `dual_objective` | $b_0 + b \cdot y$; the PMP objective $a \cdot z$ at the optimum |
| `duality_gap` | $\lvert P - D\rvert / \max(\lvert P\rvert + \lvert D\rvert, 1)$ |
| `primal_error`, `dual_error` | Largest violations of the primal and dual constraints |
| `x`, `X`, `y`, `Y` | The iterate |
| `c_minus_By` | $c - B y$ per block: the *extremal functional* on the sampled constraints; SDPB's `spectrum` tool reads it to find the zeros of $\sum_i z_i M^j_i(x)$ |

The solve is successful when `status` is
{attr}`~sdpb_python.TerminateReason.PRIMAL_DUAL_OPTIMAL`, meaning the duality
gap and both errors are below their thresholds. Other statuses mean an early
stop (iteration or time limit, a requested feasibility stop, a step that became
too small, or an interruption).

## Linear matrix inequalities

A {class}`~sdpb_python.LMI` is the degree-zero special case: every block is
sampled at a single point with a trivial basis, so SDPB acts as a dense
arbitrary-precision solver for

$$
\text{maximize } f + b \cdot y \quad \text{s.t. } M_0 + \sum_{n=1}^{N} y_n M_n \succeq 0 .
$$

`Solution.Y[j][0]` is then $M_0 + \sum_n y_n M_n$ for block $j$. Because an
$N \times N$ Schur complement is factored every iteration, this suits moderate
$N$ (thousands) and block sizes (hundreds), not large sparse SDPs.

## Precision

Everything is computed with GMP floats of a fixed number of bits. SDPB's
sample points are computed by a Newton iteration whose accuracy depends on
that precision, and its binary checkpoints and `block_data` files are only
readable at the precision they were written with. sdpb-python therefore treats
the precision as a per-process constant (see {ref}`precision`).
