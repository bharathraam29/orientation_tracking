import jax
import jax.numpy as jnp

def qinverse(q):
    """
    JAX implementation of quaternion inv.
    input::
    q --> JAX array of shape (4,)
    returns --> JAX array of shape (4,)
    """
    return jnp.array([q[0], -q[1], -q[2], -q[3]])/ jnp.linalg.norm(q)

def qmult(q1, q2):
    """
    JAX implementation of quaternion multiplications
    q1, q2 --> JAX array of shape (4,)
    returns --> JAX array of shape (4,)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return jnp.array([w, x, y, z])


def qlog(q, epsilon=1e-6):
    """
    JAX implementation of logarithm of quaternion.
    
    q--> JAX array of shape (4,), representing quaternion [q_s, q_x, q_y, q_z].
    
    returns--> JAX array of shape (4,) representing log(q).
    """
    q_s, q_x, q_y, q_z = q  # Extract scalar and vector parts
    q_v = jnp.array([q_x, q_y, q_z])
    
    norm_q = jnp.linalg.norm(q)  # Compute full quaternion norm
    norm_qv = jnp.linalg.norm(q_v)  # Compute norm of the vector part

    # Compute theta = arccos(q_s / norm_q), safely handling edge cases
    theta = jnp.arccos(jnp.clip(q_s / norm_q, -1.0, 1.0)) 
    theta = jnp.arccos(jnp.clip(q_s / norm_q, -1.0, 1.0))

    # Compute log of the vector part safely
    log_v = (theta / norm_qv) * q_v

    # Final quaternion logarithm
    return jnp.concatenate([jnp.array([jnp.log(norm_q)]), log_v])

def qexp(q):
    """
    JAX implementation of exp of quaternion.
    
    q--> JAX array of shape (4,)
    returns--> JAX array of shape (4,) representing log(q).
    """ 
    qs = q[0]  # Scalar part
    qv = q[1:]  # Vector part
    norm_q_v = jnp.linalg.norm(qv)  # ||q_vec||

    exp_qs = jnp.exp(qs)  # e^q0

    return jnp.where(norm_q_v == 0, jnp.array([exp_qs, 0.0, 0.0, 0.0]), exp_qs * jnp.concatenate([jnp.array([jnp.cos(norm_q_v)]), (qv / norm_q_v) * jnp.sin(norm_q_v)]))




def quat_log(q):
    """
    Compute the logarithm of a general (non-unit) quaternion.

    Args:
        q: jnp.array of shape (4,) representing a quaternion (q0, q1, q2, q3)

    Returns:
        jnp.array of shape (4,) representing log(q)
    """
    norm_q = jnp.linalg.norm(q)  # Full quaternion norm
    q=jnp.array(q)
    q0 = q[0]  # Scalar part
    qv = q[1:]  # Vector part

    norm_qv = jnp.linalg.norm(qv)  # Norm of the vector part
    theta = jnp.arccos(jnp.clip(q0 / norm_q, -1.0, 1.0))  # Avoid domain errors

    # Compute log of quaternion
    log_q0 = jnp.log(norm_q)  # Log of norm
    log_v = (theta / (norm_qv + 1e-9)) * qv  # Avoid division by zero

    return jnp.concatenate(([log_q0], log_v))







