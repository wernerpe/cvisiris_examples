from pydrake.all import Hyperellipsoid, MathematicalProgram, Solve
import numpy as np

def switch_ellipse_description(A, b):
    d = np.linalg.solve(A.T@A, -A.T@b)
    return Hyperellipsoid(A,d), A, d

# def get_seed_ellipse(pt1, pt2, eps = 0.01):
#     dim = pt1.shape[0]
#     pts = [pt1, pt2]
#     for _ in range(2*dim):
#         m = 0.5*(pt1+pt2) + eps*(np.random.rand(2,1)-0.5)
#         pts.append(m)

#     prog = MathematicalProgram()
#     A = prog.NewSymmetricContinuousVariables(dim, 'A')
#     b = prog.NewContinuousVariables(dim, 'b')
#     prog.AddMaximizeLogDeterminantCost(A)
#     for idx, pt in enumerate(pts):
#         S = prog.NewSymmetricContinuousVariables(dim+1, 'S')
#         prog.AddPositiveSemidefiniteConstraint(S)
#         prog.AddLinearEqualityConstraint(S[0,0] == 1)
#         v = (A@pt + b.reshape(2,1)).T
#         c = (S[1:,1:]-np.eye(dim)).reshape(-1)
#         for idx in range(dim):
#             prog.AddLinearEqualityConstraint(S[0,1 + idx]-v[0,idx], 0 )
#         for ci in c:
#             prog.AddLinearEqualityConstraint(ci, 0 )

#     prog.AddPositiveSemidefiniteConstraint(A)

#     sol = Solve(prog)
#     if sol.is_success():



#         return switch_ellipse_description(sol.GetSolution(A), sol.GetSolution(b))
#     else:
#         return None, None, None

def get_lj_ellipse(pts):
    if len(pts) ==1:
        S = Hyperellipsoid.MakeHypersphere(1e-3, pts[0, :])
        return S#, S.A(), S.center()
    # if len(pts) ==2:
    #     S, _,_ = get_seed_ellipse(pts[0,:].reshape(2,1), pts[1,:].reshape(2,1), eps= 0.01)
    #     return S 
    dim = pts[0].shape[0]
    # pts = #[pt1, pt2]
    # for _ in range(2*dim):
    #     m = 0.5*(pt1+pt2) + eps*(np.random.rand(2,1)-0.5)
    #     pts.append(m)
    prog = MathematicalProgram()
    A = prog.NewSymmetricContinuousVariables(dim, 'A')
    b = prog.NewContinuousVariables(dim, 'b')
    prog.AddMaximizeLogDeterminantCost(A)
    for idx, pt in enumerate(pts):
        pt = pt.reshape(dim,1)
        S = prog.NewSymmetricContinuousVariables(dim+1, 'S')
        prog.AddPositiveSemidefiniteConstraint(S)
        prog.AddLinearEqualityConstraint(S[0,0] == 0.99)
        v = (A@pt + b.reshape(dim,1)).T
        c = (S[1:,1:]-np.eye(dim)).reshape(-1)
        for idx in range(dim):
            prog.AddLinearEqualityConstraint(S[0,1 + idx]-v[0,idx], 0 )
        for ci in c:
            prog.AddLinearEqualityConstraint(ci, 0 )

    prog.AddPositiveSemidefiniteConstraint(A) # eps * identity
    prog.AddPositiveSemidefiniteConstraint(1e6*np.eye(dim)-A)
    from pydrake.all import MosekSolver, SolverId
    prog.SetSolverOption(SolverId("Mosek"), "tol", 1e-16)
    sol = Solve(prog)
    print(sol.get_solver_id().name())
    if sol.is_success():
        Aval = sol.GetSolution(A)
        bval = sol.GetSolution(b)
        print(f"nr pts {len(pts)}")
        print('first descr')
        for i,pt in enumerate(pts):
            print(i," cont ",(Aval@pt + bval).T@(Aval@pt + bval)<=1.0, " val ", (Aval@pt + bval).T@(Aval@pt + bval))

        HE, _, _ = switch_ellipse_description(sol.GetSolution(A), sol.GetSolution(b))
        print('second descr')
        for pt in pts:
            print(i," cont ",HE.PointInSet(pt), " val ", (pt -HE.center()).T@HE.A().T@HE.A()@(pt -HE.center()))

        return HE
    else:
        return None


def plot_ellipse(ax, H, n_samples, color = None, linewidth = 1):
    A = H.A()
    center = H.center()
    angs = np.linspace(0, 2*np.pi, n_samples+1)
    coords = np.zeros((2, n_samples + 1))
    coords[0, :] = np.cos(angs)
    coords[1, :] = np.sin(angs)
    Bu = np.linalg.inv(A)@coords
    pts = Bu + center.reshape(2,1)
    if color is None:
        ax.plot(pts[0, :], pts[1, :], linewidth = linewidth)
    else:
        ax.plot(pts[0, :], pts[1, :], linewidth = linewidth, color = color)