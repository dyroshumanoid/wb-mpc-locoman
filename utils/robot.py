from os.path import dirname, abspath

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

from .gait_sequence import GaitSequence


class Robot:
    def __init__(self, urdf_path, srdf_path, reference_pose, use_quaternion=True, lock_joints=None):
        urdf_dir = dirname(abspath(urdf_path))
        if use_quaternion:
            joint_model = pin.JointModelFreeFlyer()
        else:
            joint_model = pin.JointModelComposite()
            joint_model.addJoint(pin.JointModelTranslation())
            joint_model.addJoint(pin.JointModelSphericalZYX())

        self.robot = RobotWrapper.BuildFromURDF(urdf_path, [urdf_dir], joint_model)
        if lock_joints:
            self.robot = self.robot.buildReducedRobot(lock_joints)

        self.model = self.robot.model
        self.data = self.robot.data
        if srdf_path and reference_pose:
            pin.loadReferenceConfigurations(self.model, srdf_path)
            self.q0 = self.model.referenceConfigurations[reference_pose]
        else:
            self.q0 = self.robot.q0

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nj = self.nq - 7  # without base position and quaternion
        self.nf = 12  # 6D wrenches at both feet
 
        # Joint limits from URDF (exclude base indices)
        self.joint_pos_min = self.model.lowerPositionLimit[7:]
        self.joint_pos_max = self.model.upperPositionLimit[7:]
        self.joint_vel_max = self.model.velocityLimit[6:]
        self.joint_torque_max = self.model.effortLimit[6:]

        # Arm parameters
        self.arm_ee_frames = []  # end-effector frame in URDF

        # OCP weights
        self.Q_diag = None
        self.R_diag = None

    def set_gait_sequence(self, gait_type, gait_period):
        self.gait_sequence = GaitSequence(gait_type, gait_period)
        self.foot_frames = [self.model.getFrameId(f) for f in self.gait_sequence.feet]

import numpy as np

class TOCABI(Robot):
    def __init__(self, reference_pose="standing"):
        urdf_path = "robots/tocabi_description/urdf/tocabi.urdf"
        srdf_path = "robots/tocabi_description/srdf/tocabi.srdf"

        lock_joints = set(["Waist2_Joint", "Upperbody_Joint", "Neck_Joint", "Head_Joint"])

        super().__init__(urdf_path, srdf_path, reference_pose, lock_joints=lock_joints)
        
        # Foot dimensions for wrench cone
        self.foot_length = 0.3  
        self.foot_width = 0.26   
        
        self.arm_ee_frames = [
            self.model.getFrameId("L_Wrist2_Joint"),
            self.model.getFrameId("R_Wrist2_Joint"),
        ]
        self.nf += 6 * len(self.arm_ee_frames)

        self.hip_frames = [
            self.model.getFrameId("L_HipPitch_Joint"),
            self.model.getFrameId("R_HipPitch_Joint"),
        ]
        
        # State weights
        Q_base_pos_diag = np.concatenate((
            [0] * 2,      # base x/y
            [1000],       # base z
            [10000] * 2,  # base rot x/y
            [0],          # base rot z
        ))
        
        Q_leg_pos_diag         = np.array([1000.0] * 6)
        Q_waist_yaw_pos_diag   = np.array([1000.0])
        Q_arm_reduced_pos_diag = np.array([1000] * 8)

        Q_pos_diag = np.concatenate((Q_base_pos_diag, 
                                     Q_leg_pos_diag, Q_leg_pos_diag,
                                     Q_waist_yaw_pos_diag,
                                     Q_arm_reduced_pos_diag, Q_arm_reduced_pos_diag))  
        
        Q_base_vel_diag = np.concatenate((
            [2000] * 2,      # base lin x/y
            [1000],          # base lin z
            [1000] * 2,      # base ang x/y
            [2000],          # base ang z
        ))
        
        Q_leg_vel_diag   = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        Q_waist_yaw_vel_diag   = np.array([10.0])
        Q_arm_reduced_vel_diag = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])

        Q_vel_diag = np.concatenate((Q_base_vel_diag, 
                                     Q_leg_vel_diag, Q_leg_vel_diag,
                                     Q_waist_yaw_vel_diag,
                                     Q_arm_reduced_vel_diag, Q_arm_reduced_vel_diag))  

        self.Q_diag = np.concatenate((Q_pos_diag, Q_vel_diag))
        self.R_diag = np.concatenate((
            [1e-3] * self.nv,     # accelerations
            [5e-4] * self.nf,         # forces
            [1e-4] * self.nj,         # leg joint torques
        ))

class P73(Robot):
    def __init__(self, reference_pose="standing"):
        urdf_path = "robots/P73_description/urdf/p73.urdf"
        srdf_path = "robots/P73_description/srdf/p73.srdf"

        lock_joints = set(["WaistPitch_Joint", "WaistRoll_Joint", "NeckRoll_Joint", "NeckYaw_Joint", "NeckPitch_Joint"])

        super().__init__(urdf_path, srdf_path, reference_pose, lock_joints=lock_joints)
        
        # Foot dimensions for wrench cone
        self.foot_length = 0.225
        self.foot_width = 0.075
        
        self.hip_frames = [
            self.model.getFrameId("L_HipPitch_Joint"),
            self.model.getFrameId("R_HipPitch_Joint"),
        ]

        self.arm_ee_frames = [
            self.model.getFrameId("L_WristRoll_Joint"),
            self.model.getFrameId("R_WristRoll_Joint"),
        ]
        self.nf += 6 * len(self.arm_ee_frames)
        
        # State weights
        Q_base_pos_diag = np.concatenate((
            [0] * 2,      # base x/y
            [1000],       # base z
            [10000] * 2,  # base rot x/y
            [0],          # base rot z
        ))
        Q_leg_pos_diag   = np.array([1000.0, 1000.0, 1000.0, 100.0, 1000.0, 1000.0])
        Q_waist_yaw_pos_diag   = np.array([1000.0])
        Q_arm_reduced_pos_diag = np.array([1000] * 7)

        Q_pos_diag = np.concatenate((Q_base_pos_diag, 
                                     Q_leg_pos_diag, Q_leg_pos_diag,
                                     Q_waist_yaw_pos_diag,
                                     Q_arm_reduced_pos_diag, Q_arm_reduced_pos_diag))  
        
        Q_base_vel_diag = np.concatenate((
            [2000] * 2,      # base lin x/y
            [1000],          # base lin z
            [1000] * 2,      # base ang x/y
            [2000],          # base ang z
        ))
        
        Q_leg_vel_diag   = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0]) 
        Q_waist_yaw_vel_diag   = np.array([10.0])
        Q_arm_reduced_vel_diag = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])

        Q_vel_diag = np.concatenate((Q_base_vel_diag, 
                                     Q_leg_vel_diag, Q_leg_vel_diag,
                                     Q_waist_yaw_vel_diag,
                                     Q_arm_reduced_vel_diag, Q_arm_reduced_vel_diag))  

        self.Q_diag = np.concatenate((Q_pos_diag, Q_vel_diag))
        self.R_diag = np.concatenate((
            [1e-3] * self.nv,     # accelerations
            [5e-4] * self.nf,         # forces
            [1e-4] * self.nj,         # leg joint torques
        ))
