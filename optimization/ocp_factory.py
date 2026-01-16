from .ocp_whole_body_rnea import OCPWholeBodyRNEA

def make_ocp(dynamics, dyn_args, **kwargs):
    ocp_classes = {
        "whole_body_rnea": OCPWholeBodyRNEA,
    }

    if dynamics not in ocp_classes:
        raise ValueError(f"Unknown dynamics type: {dynamics}")

    args = dyn_args.copy()
    args.update(kwargs)

    ocp = ocp_classes[dynamics](**args)
    ocp.setup_problem()
    ocp.set_weights()

    return ocp
