from albert.api.interface import init_metric, trace_geodesic

init_metric("kerr", {"M": 1, "a": 0.5})
sol = trace_geodesic([0, 3, 1.5708, 0], [1, -0.1, 0, 0], (0, 20), 500)

for i, point in enumerate(sol.y.T):
    print(f"λ={sol.t[i]:.2f}  →  t={point[0]:.3f}, r={point[1]:.3f}, θ={point[2]:.3f}, φ={point[3]:.3f}")
