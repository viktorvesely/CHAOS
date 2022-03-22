import numpy as np
from enum import IntEnum
from matplotlib import pyplot as plt
from typing import List

# Alias for type hinting
Vector = List[float]

# Define Enum for a state vector
class State(IntEnum):
  position = 0,
  velocity = 1

# Define global constants
rho_0 = 1.2 # kgm^-3
H_n = 10_400 # m
g = 9.81 # ms^-2
mass = 120 # kg

# No parachute
C_1 = 1.2 # 1
A_1 = 1 # m^2

# Parachute
C_2 = 1.75 # 1
A_2 = 16 # m^2

def F_g(m: float) -> Vector:
  return np.array([-m * g])

def rho(h: float) -> float:
  return rho_0 * np.exp(-h / H_n)

def F_d(C: float, h: float, A: float, v: Vector) -> Vector:
  v_magnitude = np.linalg.norm(v)

  if v_magnitude == 0:
    return np.zeros(v.shape)

  v_hat = v / v_magnitude
  
  return (-C * rho(h) * A * v_magnitude ** 2) * v_hat

def forward_euler(state: Vector, params: dict, dt: float) -> Vector:

  F_final = F_g(
      params["m"]
  ) + F_d(
      params["C"],
      state[State.position][0],
      params["A"],
      state[State.velocity]
  )

  dv = F_final / params["m"]
  dx = state[State.velocity]
  
  return [
    state[State.position] + dx * dt,
    state[State.velocity] + dv * dt
  ]

def solve_system(
    t_0: float,
    x_0: float,
    v_0: float
  ) -> tuple:

  # S0
  state = [0, 0]
  state[State.position] = x_0
  state[State.velocity] = v_0

  # Other constants
  params = {
      "m": mass,
      "C": C_1,
      "A": A_1
  }

  # Capture trajectory
  trajectory = [
    state      
  ]

  # Capture time
  t = t_0
  ts = [
    t     
  ]

  landed = False
  while not landed and t < 5 * 60:

    new_state = forward_euler(state, params, dt)
    t += dt
    state = new_state
    landed = new_state[State.position] <= 0 

    trajectory.append(new_state)
    ts.append(t)

  ts = np.array(ts)
  trajectory = np.array(trajectory)

  n_samples = ts.size
  state_size = len(new_state)
  vector_size = len(new_state[0])
  trajectory = np.reshape(trajectory, (state_size, vector_size, n_samples))


  return ts, trajectory, xs

# State 0
t_0 = 0 # s
x_0 = np.array([H_n] # m
v_0 = np.array([0]) # ms^-1
t_deploy_parachute = 2 * 60 # s
dt = 0.1 # s

# Force numpy to raise exceptions
np.seterr(all="raise")
ts, trajectory, xs = solve_system(t_0, x_0, v_0)
print(trajectory.shape)

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Plot height
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Height (m)")
ax[0].ticklabel_format(useOffset=False)
ax[0].plot(ts, xs)

# Plot velocity
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Velocity ($ms^{-1}$)")
ax[1].ticklabel_format(useOffset=False)
ax[1].plot(ts, trajectory[State.position][0])

plt.show(block=True)