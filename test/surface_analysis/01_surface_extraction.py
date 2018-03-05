import os
import vtk

# Set root folder
root = os.path.join(os.getcwd(), '..', '..')


def get_actor(vtk_object, scalar_visibility=False):
    # Generate mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(vtk_object.GetOutputPort())

    if not scalar_visibility:
        mapper.ScalarVisibilityOff()
    else:
        mapper.ScalarVisibilityOn()

    # Generate Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor


def load_surface(filename):
    # Set a VTKReader
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(filename)
    reader.Update()

    return reader


def create_sphere(radius=1, x_origin=0, y_origin=0, z_origin=0):
    # Create a sphere
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(x_origin, y_origin, z_origin)
    sphere.SetRadius(radius)
    sphere.SetPhiResolution(360)
    sphere.SetThetaResolution(360)
    sphere.Update()
    return sphere


def vtk_union(obj_1, obj_2):
    # Triangles
    tri_1 = vtk.vtkTriangleFilter()
    tri_1.SetInputConnection(obj_1.GetOutputPort())

    tri_2 = vtk.vtkTriangleFilter()
    tri_2.SetInputConnection(obj_2.GetOutputPort())

    # Create an intersection operation
    boolean_operation = vtk.vtkBooleanOperationPolyDataFilter()
    boolean_operation.SetOperationToUnion()

    boolean_operation.SetInputConnection(0, tri_1.GetOutputPort())
    boolean_operation.SetInputConnection(1, tri_2.GetOutputPort())
    boolean_operation.Update()

    print('[  OK  ] Intersection found')
    return boolean_operation

def vtk_intersection(obj_1, obj_2):
    # Triangles
    tri_1 = vtk.vtkTriangleFilter()
    tri_1.SetInputConnection(obj_1.GetOutputPort())

    tri_2 = vtk.vtkTriangleFilter()
    tri_2.SetInputConnection(obj_2.GetOutputPort())

    # Create an intersection operation
    boolean_operation = vtk.vtkBooleanOperationPolyDataFilter()
    boolean_operation.SetOperationToIntersection()

    boolean_operation.SetInputConnection(0, tri_1.GetOutputPort())
    boolean_operation.SetInputConnection(1, tri_2.GetOutputPort())
    boolean_operation.Update()

    print('[  OK  ] Intersection found')
    return boolean_operation


def write_vtk_file(obj, filename):
    # Write the stl file to disk
    witer = vtk.vtkPolyDataWriter()
    witer.SetFileName(filename)
    witer.SetInputConnection(obj.GetOutputPort())
    witer.Write()


if __name__ == '__main__':
    vtk_filename = os.path.join(root, 'test', 'test_data', 'surf', 'lh.vtk')

    print('INTERSECTION ANALYSIS')
    r_min = 60
    r_max = 70

    sph_inf = create_sphere(radius=r_min)
    sph_sup = create_sphere(radius=r_max)
    brain = load_surface(vtk_filename)


    # Intersect
    intersection = vtk_intersection(sph_sup, brain)
    intersection = vtk_intersection(intersection, sph_inf)
    # intersection_vtk = load_surface('/home/sssilvar/Downloads/pepe.vtk')
    # intersection_2 = vtk_intersection(sph_inf, intersection_vtk)
    #
    # # Create mapper and actor for all
    # sph_actor = get_actor(sph_inf)
    # # brain_actor = get_actor(brain)
    # intersection_actor = get_actor(intersection_2)
    #
    # # Color
    # # brain_actor.GetProperty().SetColor(1, 0, 0)
    #
    # # write_vtk_file(intersection, '/home/sssilvar/Downloads/pepe.vtk')
    #
    # # Create a Renderer
    # renderer = vtk.vtkRenderer()
    # renderer.AddViewProp(sph_actor)
    # # renderer.AddViewProp(brain_actor)
    # renderer.AddViewProp(intersection_actor)
    # renderer.SetBackground(.1, .2, .3)
    #
    # # Create a Render Window
    # ren_win = vtk.vtkRenderWindow()
    # ren_win.AddRenderer(renderer)
    # ren_win.SetWindowName('Sphere Radius = %d' % r_min)
    #
    # # Add a WindowInteractor
    # inter = vtk.vtkRenderWindowInteractor()
    # inter.SetRenderWindow(ren_win)
    #
    # # Show the thing!
    # ren_win.Render()
    # inter.Start()
    #
    # # --
