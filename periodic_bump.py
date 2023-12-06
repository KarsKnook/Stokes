import firedrake as fd

nx = 10
degree = 4

b = 0.1
h = 0.5

mesh = fd.PeriodicSquareMesh(nx, nx, direction="x", L=1, quadrilateral=True)
Vc = mesh.coordinates.function_space()
x, y = fd.SpatialCoordinate(mesh)
f = fd.Function(Vc).interpolate(fd.as_vector([x, y + fd.conditional(fd.ge(x, 0.5 - b/2), 1, 0)*fd.conditional(fd.le(x, 0.5 + b/2), 1, 0)*(-h*y + h)]))
mesh.coordinates.assign(f)

CG = fd.FunctionSpace(mesh, "CG", degree)

f = fd.Function(CG)
f.interpolate(fd.sin(x*fd.pi)*fd.sin(y*fd.pi))

fd.File("plots/mesh.pvd").write(f)