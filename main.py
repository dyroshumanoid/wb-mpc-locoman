import time
import numpy as np
import pinocchio as pin
import casadi as ca
import matplotlib.pyplot as plt
import meshcat.transformations as tf

from args import *
from utils.robot import *
from utils.visualization import visualize_forces
from optimization import make_ocp, ocp

# Robot params
robot = TOCABI(reference_pose="standing")
# robot = P73(reference_pose="standing")
dynamics ="whole_body_rnea"  # see args.py for options

# Tracking targets
base_vel_des = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])  # linear + angular velocity

# OCP params
nodes = 5      # OCP nodes
tau_nodes = 2   # add torque limits for this many nodes
dt_min = 0.015  # initial time step
dt_max = 0.08   # final time step

# Gait params
gait_type = "walk"              # "walk" or "stand"
gait_period = 1.0               # seconds
swing_height = 0.08             # meters
swing_vel_limits = [0.3, -0.3]  # meters/second

# Solver
solver = "fatrop"  # see args.py for options
warm_start = True
compile_solver = True
load_compiled_solver = None  # None or <filename> in "codegen/lib/"

# MPC
mpc_loops = 200

# Foot parameters for wrench cone
foot_length = robot.foot_length
foot_width = robot.foot_width
mu = 0.9

# Debug
display = True  # plot joint positions, velocities, torques
plot = False  # plot joint positions, velocities, torques

def mpc_loop(ocp):
    solve_times = []
    constr_viol = []

    # Initialize params
    x_init = ocp.x_nom
    t_current = 0
    ocp.update_params(x_init, t_current)

    # Initialize solver
    ocp.init_solver(solver, SOLVER_ARGS[solver])
    if compile_solver:
        ocp.compile_solver()

    if solver == "fatrop" or solver == "ipopt":
        # Get solver function
        if load_compiled_solver:
            solver_function = ca.external("solver_function", "codegen/lib/" + load_compiled_solver)
        else:
            solver_function = ocp.solver_function

        for k in range(mpc_loops):
            # Update params
            t_current = k * dt_min
            
            ocp.update_params(x_init, t_current)
            solver_params = ocp.get_solver_params()

            # Solve
            start_time = time.time()
            solver_out = solver_function(*solver_params)
            if isinstance(solver_out, (list, tuple)):
                sol_x = solver_out[0]
                cv_fun = float(solver_out[1])
            else:
                sol_x = solver_out
                cv_fun = None
            end_time = time.time()
            sol_time = end_time - start_time
            solve_times.append(sol_time)
            print("Solve time (ms): ", sol_time * 1000)
            print("Control frequency (Hz): ", 1.0 / sol_time)

            # Constraint violation
            stacked_params = ca.DM(ocp.opti.value(ocp.opti.p))
            g, lbg, ubg = ocp.g_data(sol_x, stacked_params)
            cv = ocp.constr_viol_norm_inf(g, lbg, ubg)
            constr_viol.append(cv)
            print("CV (inf norm): ", cv)
            if cv_fun is not None:
                print("CV from solver function (inf norm): ", cv_fun)

            # Retract solution and update x_init
            ocp.retract_stacked_sol(sol_x, retract_all=False)
            dx_sol = ocp.DX_prev[1]
            x_init = ocp.dyn.state_integrate()(x_init, dx_sol)

    else:
        for k in range(mpc_loops):
            # Update params
            t_current = k * dt_min
            ocp.update_params(x_init, t_current)

            # Solve
            ocp.solve(retract_all=False)
            solve_times.append(ocp.solve_time)
            constr_viol.append(ocp.constr_viol)

            # Update x_init
            dx_sol = ocp.DX_prev[1]
            x_init = ocp.dyn.state_integrate()(x_init, dx_sol)

    # Compute total horizon time
    T = sum([ocp.opti.value(dt) for dt in ocp.dts])

    print("************** STATS **************")
    print("Avg solve time (ms): ", np.average(solve_times) * 1000)
    print("Std solve time (ms): ", np.std(solve_times) * 1000)
    print("Avg CV (inf norm): ", np.average(constr_viol))
    print("Horizon length (s): ", T)

    return ocp

def main():
    # Initialize robot
    robot.set_gait_sequence(gait_type, gait_period)
    robot_instance = robot.robot
    model = robot.model
    data = robot.data
    q0 = robot.q0
    print("Robot model: ", model)

    pin.computeAllTerms(model, data, q0, np.zeros(model.nv))

    # Setup OCP
    ocp = make_ocp(
        dynamics=dynamics,
        dyn_args=DYN_ARGS[dynamics],
        robot=robot,
        nodes=nodes,
        tau_nodes=tau_nodes,
        warm_start=warm_start,
    )
    ocp.set_time_params(dt_min, dt_max)
    ocp.set_swing_params(swing_height, swing_vel_limits)
    ocp.set_foot_params(foot_length, foot_width, mu)
    ocp.set_tracking_targets(base_vel_des)

    # Run MPC
    ocp = mpc_loop(ocp)

    if display:
        robot_instance.initViewer()
        robot_instance.loadViewerModel("pinocchio")
        
        viewer = robot_instance.viewer
        visual_model = robot_instance.visual_model
        visual_data = robot_instance.visual_data 

        for _ in range(50):
            for (q, forces) in zip(ocp.q_sol, ocp.forces_sol):
                pin.updateGeometryPlacements(model, data, visual_model, visual_data, q)
                
                for geom in visual_model.geometryObjects:
                    visual_name = f"pinocchio/visuals/{geom.name}"
                    geom_id = visual_model.getGeometryId(geom.name)
                    
                    T = visual_data.oMg[geom_id]
                    
                    scale = np.array(geom.meshScale).flatten()
                    Tscale = np.eye(4)
                    Tscale[:3, :3] = np.diag(scale)
                    
                    viewer[visual_name].set_transform(T @ Tscale)

                visualize_forces(viewer, robot, model, data, q, forces)
                time.sleep(dt_min)

    if plot:
        # Plot joint positions, velocities, torques
        if hasattr(ocp, "tau_sol"):
            tau_j_sol = ocp.tau_sol
        else:
            # Compute from RNEA
            tau_j_sol = []
            for k in range(len(ocp.q_sol)):
                q = ocp.q_sol[k].flatten()
                v = ocp.v_sol[k].flatten()
                a = ocp.a_sol[k].flatten()
                forces = ocp.forces_sol[k].flatten()

                tau_rnea = ocp.dyn.rnea_dynamics()(q, v, a, forces)
                tau_rnea = np.array(tau_rnea).flatten()
                tau_j = tau_rnea[6:]
                tau_j_sol.append(tau_j)

        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        labels = ["L_HipRoll", "L_HipPitch", "L_HipYaw", "L_Knee", "L_AnklePitch", "L_AnkleRoll",
                  "R_HipRoll", "R_HipPitch", "R_HipYaw", "R_Knee", "R_AnklePitch", "R_AnkleRoll",
                  ]

        axs[0].set_title("Joint positions (q)")
        for j in range(min(robot.nj, len(labels))):
            # Ignore base (quaternion)
            axs[0].plot([q[7 + j] for q in ocp.q_sol], label=labels[j])
        axs[0].set_xlabel("Time step")
        axs[0].set_ylabel("Position (rad)")

        axs[1].set_title("Joint velocities (v)")
        for j in range(min(robot.nj, len(labels))):
            # Ignore base
            axs[1].plot([v[6 + j] for v in ocp.v_sol], label=labels[j])
        axs[1].set_xlabel("Time step")
        axs[1].set_ylabel("Velocity (rad/s)")

        axs[2].set_title("Joint torques (tau)")
        for j in range(min(robot.nj, len(labels))):
            axs[2].plot([tau[j] for tau in tau_j_sol], label=labels[j])
        axs[2].set_xlabel("Time step")
        axs[2].set_ylabel("Torque (Nm)")

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1, 0.5))

        plt.tight_layout(rect=[0, 0, 0.88, 1])  # adjust for legend
        plt.show()

if __name__ == "__main__":
    main()
