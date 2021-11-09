import numpy as np



def euler_solver(func, state, dt=0.01, t_end=1, args=()):
    """
    Solves numerically differential equations.
    :func: (time, state, *args)
    :state: s0
    :dt: timestep
    :t_end: evolve time
    :args: args
    :return: trajectory
    """
    t = 0
    steps = int(t_end / dt)
    state = np.array(state)
    states = [state]
    for _ in range(steps):
        t += dt
        ds = np.array(func(state, t, *args)) * dt
        state = np.add(state, ds)
        states.append(state)
    
    return np.array(states)