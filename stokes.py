import firedrake as fd
from firedrake import ufl
import numpy as np
import IPython

nx1 = 10
nx2 = 10
nx = nx1 + nx2
ny1 = 5
ny2 = 10
height = np.pi/ny2
degree = 3

"""
b = 0.1
h = 0.5
mesh = fd.PeriodicSquareMesh(nx, nx, direction="x", L=1, quadrilateral=True)
Vc = mesh.coordinates.function_space()
x, y = fd.SpatialCoordinate(mesh)
g = fd.Function(Vc).interpolate(fd.as_vector([x, y + fd.conditional(fd.ge(x, 0.5 - b/2), 1, 0)*fd.conditional(fd.le(x, 0.5 + b/2), 1, 0)*(-h*y + h)]))
mesh.coordinates.assign(g)"""

#mesh = fd.Mesh('periodic_bump.msh')
base = fd.CircleManifoldMesh(nx, degree=degree)
layers = np.zeros((nx, 2))
layers[:, 1] = ny2
layers[:nx1, 0] = ny1
layers[:nx1, 1] = ny2 - ny1
mesh = fd.ExtrudedMesh(base, layers, height)
x = fd.SpatialCoordinate(mesh)

gdim = mesh.geometric_dimension()
tdim = mesh.topological_dimension()

e_v = fd.FiniteElement("Q", cell=mesh.ufl_cell(), degree=degree, variant="fdm")
e_p = fd.FiniteElement("DQ", cell=mesh.ufl_cell(), degree=degree-2, variant="fdm")

V = fd.VectorFunctionSpace(mesh, e_v, dim=tdim)
W = fd.FunctionSpace(mesh, e_p)
Z = V*W

u, p = fd.TrialFunctions(Z)
v, q = fd.TestFunctions(Z)
f = fd.Constant((0, 0))


if gdim == tdim:
    grad = fd.grad
    div = fd.div
elif tdim == 2 and gdim == 3:
    J = fd.as_matrix([[-x[1], 0], [x[0], 0], [0, 1]])
    grad = lambda u: fd.dot(fd.grad(u), J)
    div = lambda u: fd.tr(grad(u))

eu = fd.sym(grad(u))
ev = fd.sym(grad(v))
du = div(u)
dv = div(v)

a = (2*fd.inner(eu, ev) - fd.inner(p, dv) 
       - fd.inner(du, q))*fd.dx
l = fd.inner(f, v)*fd.dx

gamma = fd.Constant(2)*abs(fd.JacobianDeterminant(mesh))/ufl.CellVolume(mesh)
aP = a + fd.inner(p/gamma, q)*fd.dx + fd.inner(du*gamma, dv)*fd.dx

if mesh.extruded:
    top = "top"
    bottom = "bottom"
else:
    top = (2,)
    bottom = (1,)

bcs = [fd.DirichletBC(Z.sub(0), fd.Constant((1, 0)), top),
       fd.DirichletBC(Z.sub(0), fd.Constant((0, 0)), bottom)]

nullspace = fd.MixedVectorSpaceBasis(
    Z, [Z.sub(0), fd.VectorSpaceBasis(constant=True)])

nullspace = None
coarse = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "cholesky",
}

# FDM without static condensation
fdmstar = {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.FDMPC",
    "fdm": {
        "pc_type": "python",
        "pc_python_type": "firedrake.P1PC",
        "pmg_mg_coarse": coarse,
        "pmg_mg_levels": {
            "ksp_max_it": 1,
            "ksp_norm_type": "none",
            "esteig_ksp_type": "cg",
            "esteig_ksp_norm_type": "natural",
            "ksp_chebyshev_esteig": "0.7,0.1,0.1,1.1",
            "ksp_type": "chebyshev",
            #"pc_type": "python",
            #"pc_python_type": "firedrake.ASMExtrudedStarPC",
            #"pc_star_sub_sub_pc_type": "cholesky",
            "pc_type": "jacobi",
        }
    }
}

fieldsplit_1 = {
    "ksp_type": "preonly",
    "pc_type": "jacobi",
}

parameters = {
    "mat_type": "matfree",
    "ksp_monitor": None,
    "ksp_type": "minres",
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled": {
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",
        "fieldsplit_0": fdmstar,
        "fieldsplit_1": fieldsplit_1,
    }
}

"""parameters = {
    "ksp_type": "gmres",
    "ksp_monitor": None,
    "mat_type": "aij",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}"""

sol = fd.Function(Z)

LVP = fd.LinearVariationalProblem(a, l, sol, bcs=bcs, aP=aP)
LVS = fd.LinearVariationalSolver(LVP, solver_parameters=parameters, nullspace=nullspace)

LVS.solve()


velocity, pressure = sol.subfunctions
velocity.rename("Velocity")
pressure.rename("Pressure")

fd.File("plots/stokes.pvd").write(velocity, pressure)

#IPython.embed()