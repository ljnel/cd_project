import pinocchio as pin
import hppfcl as fcl
import numpy as np


def create(nb_pendulum: int, body_radius=0.1, body_mass=1.0, length=1.0):
    "Set up the inertial and geometry models for a (double) pendulum."
    # Create models
    model = pin.Model()
    geom_model = pin.GeometryModel()

    parent_id = 0
    joint_placement = pin.SE3.Identity()

    shape0 = fcl.Sphere(body_radius)
    geom0_obj = pin.GeometryObject("base", 1, shape0, pin.SE3.Identity())  # type: ignore
    geom0_obj.meshColor = np.array([1.0, 0.1, 0.1, 1.0])  # type: ignore
    geom_model.addGeometryObject(geom0_obj)

    for k in range(nb_pendulum):
        joint_name = "joint_" + str(k + 1)
        joint_id = model.addJoint(
            parent_id, pin.JointModelRX(), joint_placement, joint_name
        )

        body_inertia = pin.Inertia.FromSphere(body_mass, body_radius)
        body_placement = joint_placement.copy()
        assert body_placement.translation is not None
        body_placement.translation[2] = length
        model.appendBodyToJoint(joint_id, body_inertia, body_placement)

        geom1_name = "ball_" + str(k + 1)
        shape1 = fcl.Sphere(body_radius)
        geom1_obj = pin.GeometryObject(geom1_name, joint_id, shape1, body_placement)
        geom1_obj.meshColor = np.ones(4)
        geom_model.addGeometryObject(geom1_obj)

        geom2_name = "bar_" + str(k + 1)
        shape2 = fcl.Cylinder(body_radius / 4.0, body_placement.translation[2])
        shape2_placement = body_placement.copy()
        shape2_placement.translation[2] /= 2.0

        geom2_obj = pin.GeometryObject(geom2_name, joint_id, shape2, shape2_placement)
        geom2_obj.meshColor = np.array([0.0, 0.0, 0.0, 1.0])
        geom_model.addGeometryObject(geom2_obj)

        parent_id = joint_id
        joint_placement = body_placement.copy()

    end_frame = pin.Frame(  # end effector (TODO: fix this)
        "ee_frame",
        model.getJointId("joint_" + str(nb_pendulum)),
        0,
        body_placement,
        pin.FrameType(3),
    )
    model.addFrame(end_frame)
    return model, geom_model