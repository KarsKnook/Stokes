import firedrake as fd
from firedrake import ufl
import numpy as np


# problem parameters
degree = 3
nx1 = 10
nx2 = 10  # amount of cells for bump width
nx3 = 10
nx = nx1 + nx2 + nx3
ny1 = 20
ny2 = 5  # amount of cells above bump
ny3 = 20  # easiest if the same as ny1


# mesh
#mesh_type = "periodic"
mesh_type = "circular"
if mesh_type == "periodic":
    length = 1
    base = fd.PeriodicIntervalMesh(nx, length)
    layer_height = length/nx  # ensures square cells
elif mesh_type == "circular":
    radius = 1
    base = fd.CircleManifoldMesh(nx, radius=radius, degree=3)
    layer_height = 2*np.pi*radius/nx
layers = np.zeros((nx, 2))
layers[nx1:nx1+nx2, 0] = ny1 - ny2
layers[:nx1, 1] = ny1
layers[nx1:nx1+nx2, 1] = ny2
layers[nx1+nx2:, 1] = ny3

mesh = fd.ExtrudedMesh(base, layers, layer_height)
x = fd.SpatialCoordinate(mesh)
gdim = mesh.geometric_dimension()  # dim of space the mesh lives in
tdim = mesh.topological_dimension()  # dim of the manifold


# finite element space
e_v = fd.FiniteElement("Q", cell=mesh.ufl_cell(), degree=degree, variant="fdm")
e_p = fd.FiniteElement("DQ", cell=mesh.ufl_cell(), degree=degree-2, variant="fdm")
V = fd.VectorFunctionSpace(mesh, e_v, dim=tdim)
W = fd.FunctionSpace(mesh, e_p)
Z = V*W


# defining linear variational problem
u, p = fd.TrialFunctions(Z)
v, q = fd.TestFunctions(Z)

if gdim == tdim:
    grad = fd.grad
    div = fd.div
elif tdim == 2 and gdim == 3:  # if a 2D mesh is embedded in 3D space
    J = fd.as_matrix([[-x[1], 0], [x[0], 0], [0, 1]])
    grad = lambda u: fd.dot(fd.grad(u), J)
    div = lambda u: fd.tr(grad(u))

eu = fd.sym(grad(u))
ev = fd.sym(grad(v))
du = div(u)
dv = div(v)

f = fd.Constant((0, 0))

a = (2*fd.inner(eu, ev) - fd.inner(p, dv) 
       - fd.inner(du, q))*fd.dx
l = fd.inner(f, v)*fd.dx


# bcs
if mesh.extruded:
    top = "top"
    bottom = "bottom"
else:
    top = (2,)
    bottom = (1,)

bcs = [fd.DirichletBC(Z.sub(0), fd.Constant((1, 0)), top),
       fd.DirichletBC(Z.sub(0), fd.Constant((0, 0)), bottom)]


# pressure nullspace
nullspace = fd.MixedVectorSpaceBasis(
    Z, [Z.sub(0), fd.VectorSpaceBasis(constant=True)])


# solving
gamma = fd.Constant(2)*abs(fd.JacobianDeterminant(mesh))/ufl.CellVolume(mesh)  # to account for non-orthogonal quads
aP = a + fd.inner(p/gamma, q)*fd.dx + fd.inner(du*gamma, dv)*fd.dx  # augmented Lagrangian?

coarse = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "cholesky",
}

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

sol = fd.Function(Z)
LVP = fd.LinearVariationalProblem(a, l, sol, bcs=bcs, aP=aP)
LVS = fd.LinearVariationalSolver(LVP, solver_parameters=parameters, nullspace=nullspace)
LVS.solve()

# plotting
velocity, pressure = sol.subfunctions
velocity.rename("Velocity")
pressure.rename("Pressure")
fd.File("plots/stokes.pvd").write(velocity, pressure)