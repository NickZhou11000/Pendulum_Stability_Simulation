from math import pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Constants for single inverted pendulum
g = 9.81
L = 0.5
m = 0.1
b = 0.05
dt = 0.02
v_max = 2.0

# PID Controller parameters (manually tuned)
Kp = 50.0
Kd = 10.0

# Initial state: angle in radians from vertical up, angular velocity
theta = np.radians(190)  # Start near upright
dtheta = 0.0
x_cart = 0.0

theta_list = []
cart_list = []
v_list = []
t_list = []


def simulate_step(theta, dtheta, x_cart):
    error = pi - theta
    derr = 0 - dtheta
    v = np.clip(Kp * error + Kd * derr, -v_max, v_max)

    ddtheta = -g / L * np.sin(theta) - b / (m * L**2) * dtheta - v / L * np.cos(theta)
    dtheta += ddtheta * dt
    theta += dtheta * dt
    x_cart += v * dt

    return theta, dtheta, x_cart, v


# Run simulation
n_steps = 500
for i in range(n_steps):
    t_list.append(i * dt)
    theta, dtheta, x_cart, v = simulate_step(theta, dtheta, x_cart)
    theta_list.append(np.degrees(theta) + 180)
    cart_list.append(x_cart)
    v_list.append(v)

# Plotting
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(t_list, theta_list)
plt.axhline(180, color='gray', linestyle='--')
plt.ylabel("Angle (°)")
plt.title("Angle / Position / Speed vs Time")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t_list, cart_list, color="green")
plt.ylabel("Position (m)")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t_list, v_list, color="red")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.grid()

plt.tight_layout()
plt.show()


def animate_single_pendulum():
    fig, ax = plt.subplots()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1, 1.5)
    ax.set_aspect("equal")
    ax.grid()

    (cart_line,) = ax.plot([], [], "s-", lw=2, color="blue")
    (pendulum_line,) = ax.plot([], [], "o-", lw=2, color="red")

    def init():
        cart_line.set_data([], [])
        pendulum_line.set_data([], [])
        return cart_line, pendulum_line

    def update(i):
        x = cart_list[i]
        th = np.radians(theta_list[i] - 180)
        x_p = x + L * np.sin(th)
        y_p = -L * np.cos(th)

        cart_line.set_data([x - 0.2, x + 0.2], [0, 0])
        pendulum_line.set_data([x, x_p], [0, y_p])
        return cart_line, pendulum_line

    ani = animation.FuncAnimation(
        fig, update, frames=len(theta_list), init_func=init, blit=True, interval=20
    )
    plt.show()


animate_single_pendulum()
