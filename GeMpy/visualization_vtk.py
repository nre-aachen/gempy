import vtk
import random


class InterfaceSphere(vtk.vtkSphereSource):
    def __init__(self, index):
        self.index = index  # df index


class FoliationArrow(vtk.vtkArrowSource):
    def __init__(self, index):
        self.index = index  # df index


class CustomInteractor(vtk.vtkInteractorStyleTrackballActor):
    """
    Modified vtkInteractorStyleTrackballActor class to accomodate for interface df modification.
    """

    def __init__(self, ren_list, geo_data, parent=None):
        self.ren_list = ren_list
        self.geo_data = geo_data
        self.AddObserver("MiddleButtonPressEvent", self.middleButtonPressEvent)
        self.AddObserver("MiddleButtonReleaseEvent", self.middleButtonReleaseEvent)

        self.AddObserver("LeftButtonPressEvent", self.leftButtonPressEvent)
        self.AddObserver("LeftButtonReleaseEvent", self.leftButtonReleaseEvent)

        self.PickedActor = None
        self.PickedProducer = None

    def leftButtonPressEvent(self, obj, event):
        print("Pressed left mouse button")

        m = vtk.vtkMatrix4x4()

        clickPos = self.GetInteractor().GetEventPosition()
        pickers = []
        picked_actors = []
        for r in self.ren_list:
            pickers.append(vtk.vtkPicker())
            pickers[-1].Pick(clickPos[0], clickPos[1], 0, r)
            picked_actors.append(pickers[-1].GetActor())
        for pa in picked_actors:
            if pa is not None:
                self.PickedActor = pa
        # vtk.vtkOpenGLActor.GetOrientation?
        # matrix = self.PickedActor.GetMatrix(m)
        # if self.PickedActor is
        # self.PickedActor.SetScale(2)
        # renwin.Render()

        orientation = self.PickedActor.GetOrientation()
        print(str(orientation))

        self.OnLeftButtonDown()

    def leftButtonReleaseEvent(self, obj, event):
        # matrix = self.PickedActor.GetMatrix(vtk.vtkMatrix4x4())
        matrix = self.PickedActor.GetOrientation()
        print(str(matrix))
        self.OnLeftButtonUp()

    def middleButtonPressEvent(self, obj, event):
        # print("Middle Button Pressed")
        clickPos = self.GetInteractor().GetEventPosition()

        pickers = []
        picked_actors = []
        for r in self.ren_list:
            pickers.append(vtk.vtkPicker())
            pickers[-1].Pick(clickPos[0], clickPos[1], 0, r)
            picked_actors.append(pickers[-1].GetActor())

        for pa in picked_actors:
            if pa is not None:
                self.PickedActor = pa

        if self.PickedActor is not None:
            _m = self.PickedActor.GetMapper()
            _i = _m.GetInputConnection(0, 0)
            _p = _i.GetProducer()

            if type(_p) is not InterfaceSphere:
                # then go deeper
                alg = _p.GetInputConnection(0, 0)
                self.PickedProducer = alg.GetProducer()
            else:
                self.PickedProducer = _p
        # print(str(type(self.PickedProducer)))
        self.OnMiddleButtonDown()
        return

    def middleButtonReleaseEvent(self, obj, event):
        # print("Middle Button Released")
        if self.PickedActor is not None or type(self.PickedProducer) is not FoliationArrow:
            try:
                _c = self.PickedActor.GetCenter()
                self.geo_data.interface_modify(self.PickedProducer.index, X=_c[0], Y=_c[1], Z=_c[2])
            except AttributeError:
                pass
        if type(self.PickedProducer) is FoliationArrow:
            print("Yeha, Arrow!")
            _c = self.PickedActor.GetCenter()
            print(str(_c))
            self.geo_data.foliation_modify(self.PickedProducer.index, X=_c[0], Y=_c[1], Z=_c[2])

        self.OnMiddleButtonUp()
        return


def visualize(geo_data):
    """
    Returns:

    """
    spheres = create_interface_spheres(geo_data)
    arrows = create_foliation_arrows(geo_data)
    arrows_transformers = create_arrow_transformers(arrows, geo_data)

    mappers, actors = create_mappers_actors(spheres)
    arrow_mappers, arrow_actors = create_mappers_actors(arrows_transformers)

    renwin = vtk.vtkRenderWindow()
    renwin.SetSize(1000, 800)
    renwin.SetWindowName('Render Window')

    xmins = [0, 0.4, 0.4, 0.4]
    xmaxs = [0.4, 1, 1, 1]
    ymins = [0, 0, 0.33, 0.66]
    ymaxs = [1, 0.33, 0.66, 1]

    ren_list = []
    for i in range(4):
        ren_list.append(vtk.vtkRenderer())
        renwin.AddRenderer(ren_list[-1])
        ren_list[-1].SetViewport(xmins[i], ymins[i], xmaxs[i], ymaxs[i])

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetInteractorStyle(CustomInteractor(ren_list, geo_data))
    interactor.SetRenderWindow(renwin)

    _e = geo_data.extent  # array([ x, X,  y, Y,  z, Z])

    # 3d model camera
    model_cam = vtk.vtkCamera()
    model_cam.SetPosition(50, 50, 50)
    model_cam.SetFocalPoint(0, 0, 0)

    # XY camera
    xy_cam = vtk.vtkCamera()
    xy_cam.SetPosition(_e[1] / 2,
                       _e[3] / 2,
                       _e[5] * 3)

    xy_cam.SetFocalPoint(_e[1] / 2,
                         _e[3] / 2,
                         _e[5] / 2)

    # YZ camera
    yz_cam = vtk.vtkCamera()
    yz_cam.SetPosition(_e[1] * 3,
                       _e[3] / 2,
                       _e[5] / 2)

    yz_cam.SetFocalPoint(_e[1] / 2,
                         _e[3] / 2,
                         _e[5] / 2)

    # XZ camera
    xz_cam = vtk.vtkCamera()
    xz_cam.SetPosition(_e[1] / 2,
                       _e[3] * 3,
                       _e[5] / 2)

    xz_cam.SetFocalPoint(_e[1] / 2,
                         _e[3] / 2,
                         _e[5] / 2)
    xz_cam.SetViewUp(1, 0, 0)

    camera_list = [model_cam, xy_cam, yz_cam, xz_cam]

    ren_list[0].SetActiveCamera(model_cam)
    ren_list[1].SetActiveCamera(xy_cam)
    ren_list[2].SetActiveCamera(yz_cam)
    ren_list[3].SetActiveCamera(xz_cam)

    # ///////////////////////////////////////////////////////////////
    # create AxesActor and customize
    cubeAxesActor = create_axes(geo_data, camera_list)

    for r in ren_list:
        # add axes actor to all renderers
        r.AddActor(cubeAxesActor)
        for a in actors:
            # add "normal" actors to renderers (spheres)
            r.AddActor(a)
        for a in arrow_actors:
            r.AddActor(a)

    interactor.Initialize()
    interactor.Start()

    del renwin, interactor


def create_interface_spheres(geo_data, r=0.33):
    "Creates InterfaceSphere (vtkSphereSource) for all interface positions in dataframe."
    spheres = []
    for index, row in geo_data.interfaces.iterrows():
        spheres.append(InterfaceSphere(index))
        spheres[-1].SetCenter(geo_data.interfaces.iloc[index]["X"],
                              geo_data.interfaces.iloc[index]["Y"],
                              geo_data.interfaces.iloc[index]["Z"])
        spheres[-1].SetRadius(r)
    return spheres


def create_foliation_arrows(geo_data):
    "Creates FoliationArrow (vtkArrowSource) for all foliation positions in dataframe."
    arrows = []
    for index, row in geo_data.foliations.iterrows():
        arrows.append(FoliationArrow(index))
    return arrows


def create_mappers_actors(sources):
    "Creates mappers and connected actors for all given sources."
    mappers = []
    actors = []
    for s in sources:
        mappers.append(vtk.vtkPolyDataMapper())
        mappers[-1].SetInputConnection(s.GetOutputPort())
        actors.append(vtk.vtkActor())
        actors[-1].SetMapper(mappers[-1])
    return (mappers, actors)


def get_transform(startPoint, endPoint):
    # Compute a basis
    normalizedX = [0 for i in range(3)]
    normalizedY = [0 for i in range(3)]
    normalizedZ = [0 for i in range(3)]

    # The X axis is a vector from start to end
    math = vtk.vtkMath()
    math.Subtract(endPoint, startPoint, normalizedX)
    length = math.Norm(normalizedX)
    math.Normalize(normalizedX)

    # The Z axis is an arbitrary vector cross X
    arbitrary = [0 for i in range(3)]
    arbitrary[0] = random.uniform(-10, 10)
    arbitrary[1] = random.uniform(-10, 10)
    arbitrary[2] = random.uniform(-10, 10)
    math.Cross(normalizedX, arbitrary, normalizedZ)
    math.Normalize(normalizedZ)

    # The Y axis is Z cross X
    math.Cross(normalizedZ, normalizedX, normalizedY)
    matrix = vtk.vtkMatrix4x4()

    # Create the direction cosine matrix
    matrix.Identity()
    for i in range(3):
        matrix.SetElement(i, 0, normalizedX[i])
        matrix.SetElement(i, 1, normalizedY[i])
        matrix.SetElement(i, 2, normalizedZ[i])

    # Apply the transforms
    transform = vtk.vtkTransform()
    transform.Translate(startPoint)
    transform.Concatenate(matrix)
    transform.Scale(length, length, length)

    return transform


def create_arrow_transformers(arrows, geo_data):
    "Creates list of arrow transformation objects."
    # grab start and end points for foliation arrows
    arrows_sp = []
    arrows_ep = []
    f = 0.75
    for arrow in arrows:
        _sp = (geo_data.foliations.iloc[arrow.index]["X"] - geo_data.foliations.iloc[arrow.index]["G_x"] / f,
               geo_data.foliations.iloc[arrow.index]["Y"] - geo_data.foliations.iloc[arrow.index]["G_x"] / f,
               geo_data.foliations.iloc[arrow.index]["Z"] - geo_data.foliations.iloc[arrow.index]["G_x"] / f)
        _ep = (geo_data.foliations.iloc[arrow.index]["X"] + geo_data.foliations.iloc[arrow.index]["G_x"] / f,
               geo_data.foliations.iloc[arrow.index]["Y"] + geo_data.foliations.iloc[arrow.index]["G_y"] / f,
               geo_data.foliations.iloc[arrow.index]["Z"] + geo_data.foliations.iloc[arrow.index]["G_z"] / f)
        arrows_sp.append(_sp)
        arrows_ep.append(_ep)

    # ///////////////////////////////////////////////////////////////
    # create transformers for ArrowSource and transform

    arrows_transformers = []
    for i, arrow in enumerate(arrows):
        arrows_transformers.append(vtk.vtkTransformPolyDataFilter())
        arrows_transformers[-1].SetTransform(get_transform(arrows_sp[i], arrows_ep[i]))
        arrows_transformers[-1].SetInputConnection(arrow.GetOutputPort())

    return arrows_transformers


def create_axes(geo_data, camera_list):
    "Create and return cubeAxesActor, settings."
    cubeAxesActor = vtk.vtkCubeAxesActor()
    cubeAxesActor.SetBounds(geo_data.extent)
    cubeAxesActor.SetCamera(camera_list[0])

    # set axes and label colors
    cubeAxesActor.GetTitleTextProperty(0).SetColor(1.0, 0.0, 0.0)
    cubeAxesActor.GetLabelTextProperty(0).SetColor(1.0, 0.0, 0.0)
    # font size doesn't work seem to work - maybe some override in place?
    # cubeAxesActor.GetLabelTextProperty(0).SetFontSize(10)
    cubeAxesActor.GetTitleTextProperty(1).SetColor(0.0, 1.0, 0.0)
    cubeAxesActor.GetLabelTextProperty(1).SetColor(0.0, 1.0, 0.0)
    cubeAxesActor.GetTitleTextProperty(2).SetColor(0.0, 0.0, 1.0)
    cubeAxesActor.GetLabelTextProperty(2).SetColor(0.0, 0.0, 1.0)

    cubeAxesActor.DrawXGridlinesOn()
    cubeAxesActor.DrawYGridlinesOn()
    cubeAxesActor.DrawZGridlinesOn()

    cubeAxesActor.XAxisMinorTickVisibilityOff()
    cubeAxesActor.YAxisMinorTickVisibilityOff()
    cubeAxesActor.ZAxisMinorTickVisibilityOff()

    cubeAxesActor.SetXTitle("X")
    cubeAxesActor.SetYTitle("Y")
    cubeAxesActor.SetZTitle("Z")

    cubeAxesActor.SetXAxisLabelVisibility(0)
    cubeAxesActor.SetYAxisLabelVisibility(0)
    cubeAxesActor.SetZAxisLabelVisibility(0)

    # only plot grid lines furthest from viewpoint
    cubeAxesActor.SetGridLineLocation(cubeAxesActor.VTK_GRID_LINES_FURTHEST)
    return cubeAxesActor
