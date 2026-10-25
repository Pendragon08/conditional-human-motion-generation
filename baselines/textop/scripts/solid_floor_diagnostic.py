"""Diagnostic static-box floor: same flat top and material, different collider.

This is not promoted to the production paper configuration without validation.
"""
import pathlib
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Gf


def create(path):
    path = pathlib.Path(path)
    assert not path.exists()
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, '/Ground')
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    # Retain a flat visual mesh for TerrainImporter's geometry bookkeeping.
    mesh = UsdGeom.Mesh.Define(stage, '/Ground/Visual')
    mesh.CreatePointsAttr([(-2000, -2000, 0), (2000, -2000, 0), (2000, 2000, 0), (-2000, 2000, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    box = UsdGeom.Cube.Define(stage, '/Ground/Solid')
    box.CreateSizeAttr(1.0)
    box.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.1))
    box.AddScaleOp().Set(Gf.Vec3d(4000, 4000, 0.2))
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    material = UsdShade.Material.Define(stage, '/Ground/PhysicsMaterial')
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(1.0)
    api.CreateDynamicFrictionAttr(1.0)
    api.CreateRestitutionAttr(0.0)
    physical = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physical.CreateFrictionCombineModeAttr('multiply')
    physical.CreateRestitutionCombineModeAttr('multiply')
    UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(material, materialPurpose='physics')
    stage.GetRootLayer().Save()
    return str(path)
