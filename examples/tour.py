"""End-to-end tour of sdpb-python, runnable against an installed wheel.

    python -m venv .venv && source .venv/bin/activate
    pip install sdpb-python --find-links https://github.com/YPWang0818/sdpb-python/releases/expanded_assets/v0.2.0
    python examples/tour.py

Covers: solving a polynomial matrix program (the SDPB manual's 1d example),
sampling/conversion introspection, a linear matrix inequality, the solver
handle (staged runs, checkpoints, warm starts), pmp.json round trips, and the
errors a user can expect.  Prints ALL CHECKS PASSED at the end.
"""
import json, tempfile, time
import mpmath
import sdpb_python as sdpb

print("sdpb-python", sdpb.__version__, "| SDPB", sdpb.sdpb_version(), "| from", sdpb.__file__)
sdpb.set_precision(768)
mpmath.mp.prec = 768

# 1. PMP from the SDPB manual: maximize -y s.t. 1 + x^4 + y (x^4/12 + x^2) >= 0
pmp = sdpb.PMP(objective=[0, -1], normalization=[1, 0],
               matrices=[sdpb.PolynomialMatrix([[[[1, 0, 0, 0, 1], [0, 0, 1, 0, "1/12"]]]])])
t = time.time()
sol = pmp.solve(duality_gap_threshold="1e-30", primal_error_threshold="1e-30", dual_error_threshold="1e-30",
                want=("x", "y", "z", "X", "Y", "c_minus_By"))
ref = mpmath.mpf("1.84026576313204924668804017173055420056358532030282556465761906133430166726537336826049865612094019")
assert sol.optimal, sol.status
assert abs(sol.dual_objective - ref) < mpmath.mpf(10) ** -28, sol.dual_objective  # converged to the 1e-30 gap threshold
print(f"1. PMP: {sol.status.name}, objective matches reference to 1e-28, {sol.iterations} iterations, {time.time()-t:.1f}s")
print("   y =", mpmath.nstr(sol.y[0], 30), "| z =", [mpmath.nstr(v, 10) for v in sol.z])
print("   X even block", sol.X[0][0].rows, "x", sol.X[0][0].cols, "| c-B.y entries", len(sol.c_minus_By[0]))

# 2. sampling / conversion introspection
s = pmp.matrices[0].sampled()
sdp = pmp.to_sdp()
print(f"2. sampling: {s.num_points} points, block c has {len(sdp.blocks[0].c)} rows, B is {sdp.blocks[0].B.rows}x{sdp.blocks[0].B.cols}")

# 3. LMI: maximize 3 + y1 + y2 s.t. diag(1-y1, 1-y2, 1+y1+y2) >= 0  -> y = (1, 1), value 5
d = lambda a, b, c: [[a, 0, 0], [0, b, 0], [0, 0, c]]
lmi = sdpb.LMI(b=[1, 1], f=3, blocks=[(d(1, 1, 1), d(-1, 0, 1), d(0, -1, 1))]).solve(duality_gap_threshold="1e-40")
assert lmi.optimal and abs(lmi.dual_objective - 5) < mpmath.mpf(10) ** -30
print("3. LMI: objective", mpmath.nstr(lmi.dual_objective, 15), "y =", [mpmath.nstr(v, 10) for v in lmi.y])

# 4. solver handle: staged run, checkpoint, warm start
with tempfile.TemporaryDirectory() as tmp:
    with pmp.solver(duality_gap_threshold="1e-30") as solver:
        first = solver.run(max_iterations=10)
        final = solver.run()
        solver.save_checkpoint(tmp + "/ck")
        assert first.status.name == "MAX_ITERATIONS_EXCEEDED" and final.optimal
        assert solver.total_iterations == sol.iterations, (solver.total_iterations, sol.iterations)
    restarted = pmp.solve(checkpoint_dir=tmp + "/ck", duality_gap_threshold="1e-30")
    assert restarted.optimal and restarted.iterations < 5
    print(f"4. handle: 10 + {final.iterations} iterations = one-shot {sol.iterations}; restart from checkpoint: {restarted.iterations} iterations")
    warm = pmp.solver(duality_gap_threshold="1e-30").warm_start(y=sol.y, X=sol.X, Y=sol.Y).run()
    print("   warm start from the optimum:", warm.iterations, "iterations")

    # 5. pmp.json round trip
    sdpb.write_pmp_json(pmp, tmp + "/pmp.json")
    again = sdpb.read_pmp_json(tmp + "/pmp.json").solve(duality_gap_threshold="1e-30")
    assert abs(again.dual_objective - sol.dual_objective) < mpmath.mpf(10) ** -28
    print("5. pmp.json written and re-solved identically;", len(json.load(open(tmp + "/pmp.json"))["PositiveMatrixWithPrefactorArray"]), "matrix")

# 6. error handling: precision lock and validation
try:
    pmp.solve(precision=1024)
except sdpb.SDPBError as e:
    print("6. precision lock:", str(e)[:70], "...")
try:
    sdpb.PolynomialMatrix([[[[1], [2]], [[1], [3]]], [[[1], [2]], [[1], [3]]]])
except ValueError as e:
    print("   validation:", e)
print("ALL CHECKS PASSED")
