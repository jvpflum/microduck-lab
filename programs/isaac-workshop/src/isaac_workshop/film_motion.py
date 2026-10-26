"""Portable film math and controller boundaries; never generates robot poses.

The runtime schedule, per-actor history and body-matrix conversion are extracted
from the authored six-robot prototype. Synthetic tests do not establish native
execution. Native body_q is a link-origin pose, not a center-of-mass pose.
"""
import math

def require(value, message):
    if not value:
        raise ValueError(message)

def command_at(completed_controls):
    """8 s walk, 4 s stop, 4 s walk, 4 s stop. No yaw gain/navigation."""
    require(type(completed_controls) is int and 0 <= completed_controls <= 1000,
            'Control counter outside film schedule')
    walking = completed_controls < 400 or 600 <= completed_controls < 800
    return (.15 if walking else 0., 0., 0.)

def previous_action(env, action_name):
    """Per-term raw action equals the input after optional wrapper-equivalent clip.

    Using the global manager action here would silently create 84-wide history.
    Pinned JointAction.process_actions copies its own incoming 14-wide slice.
    """
    value = env.action_manager.get_term(action_name).raw_actions
    require(tuple(value.shape) == (1, 14), 'Per-duck action history must be 1x14')
    return value

def body_matrix(q):
 require(len(q)==7 and all(math.isfinite(v)for v in q),'Finite actual body state required')
 x,y,z,w=q[3:];require(abs(x*x+y*y+z*z+w*w-1)<2e-4,'Nonunit native body quaternion')
 return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),q[0]],
         [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),q[1]],
         [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y),q[2]],[0.,0.,0.,1.]]

def stage_world_matrix(body_q, meters_per_unit):
    """Return row-major USD/Gf matrix for metre-local mesh vertices.

    Translation and the three basis vectors must both be rescaled. The caller
    must separately prove link/mesh mapping and undo the current USD parent
    transform. This conversion proves neither provenance nor scene admission.
    """
    require(isinstance(meters_per_unit, (int, float))
            and not isinstance(meters_per_unit, bool)
            and math.isfinite(meters_per_unit) and meters_per_unit > 0,
            'Positive finite stage unit required')
    matrix = body_matrix(body_q)
    scale = 1. / meters_per_unit
    # Exact conversion expression from the native bind_episode loop.
    return [matrix[c][r] * (scale if c < 3 else 1.)
            for r in range(4) for c in range(4)]
