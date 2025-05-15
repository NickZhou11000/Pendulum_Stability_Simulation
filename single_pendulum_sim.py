from math import pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class SinglePendulumSimulation:
    def __init__(self):
        # Constants for single inverted pendulum
        self.g = 9.81
        self.L = 0.5
        self.m = 0.1
        self.b = 0.05
        self.dt = 0.02
        self.v_max = 2.0

        # PID Controller parameters
        self.Kp = 50.0
        self.Kd = 10.0

        # Initial state
        self.theta = np.radians(190)  # Start near upright
        self.dtheta = 0.0
        self.x_cart = 0.0

        # Data storage
        self.theta_list = []
        self.theta_list_plot = []
        self.cart_list = []
        self.v_list = []
        self.t_list = []
        
        # Animation properties
        self.ani = None

    def simulate_step(self):
        """Calculate one simulation step"""
        # For the controller, we want error = 0 when pendulum is upright (180 degrees)
        error = pi - self.theta
        derr = 0 - self.dtheta
        v = np.clip(self.Kp * error + self.Kd * derr, -self.v_max, self.v_max)

        ddtheta = -self.g / self.L * np.sin(self.theta) - self.b / (self.m * self.L**2) * self.dtheta - v / self.L * np.cos(self.theta)
        self.dtheta += ddtheta * self.dt
        self.theta += self.dtheta * self.dt
        self.x_cart += v * self.dt

        return self.theta, self.dtheta, self.x_cart, v

    def run_simulation(self, n_steps=500):
        """Run the full simulation for the specified number of steps"""
        # Clear previous data
        self.t_list = []
        self.theta_list = []
        self.theta_list_plot = []
        self.cart_list = []
        self.v_list = []
        
        for i in range(n_steps):
            self.t_list.append(i * self.dt)
            theta, dtheta, x_cart, v = self.simulate_step()
            
            # Store raw angle in radians for animation
            self.theta_list.append(theta)
            # Store angle in degrees for plotting (180° is up, 90° is right)
            self.theta_list_plot.append(np.degrees(theta))
            self.cart_list.append(x_cart)
            self.v_list.append(v)
        
        return self.t_list, self.theta_list_plot, self.cart_list, self.v_list

    def plot_results(self):
        """Plot the simulation results"""
        if not self.t_list:
            print("No simulation data. Run the simulation first.")
            return
            
        plt.figure(figsize=(10, 8))

        plt.subplot(3, 1, 1)
        plt.plot(self.t_list, self.theta_list_plot)
        plt.axhline(180, color='gray', linestyle='--', label="Upright position")
        plt.ylabel("Angle (°)")
        plt.title("Angle / Position / Speed vs Time")
        plt.grid()
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(self.t_list, self.cart_list, color="green")
        plt.ylabel("Position (m)")
        plt.grid()

        plt.subplot(3, 1, 3)
        plt.plot(self.t_list, self.v_list, color="red")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.grid()

        plt.tight_layout()
        plt.show()

    def animate_pendulum(self, repeat=False):
        """Create and display animation of the pendulum
        
        Args:
            repeat (bool): Whether to repeat the animation. Default is False.
        """
        if not self.theta_list:
            print("No simulation data. Run the simulation first.")
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(-3, 3)
        ax.set_ylim(-1, 1.5)
        ax.set_aspect("equal")
        ax.grid()
        
        # Add coordinate system reference
        ax.text(2.7, 0.1, "90°", fontsize=12)
        ax.text(-0.1, 1.3, "180°", fontsize=12)
        ax.text(-2.8, 0.1, "270°", fontsize=12)
        ax.text(-0.1, -0.8, "0°", fontsize=12)
        
        # Create cart as rectangle
        cart_width = 0.4
        cart_height = 0.2
        cart = plt.Rectangle((self.cart_list[0] - cart_width/2, -cart_height/2), 
                            cart_width, cart_height, color='blue')
        ax.add_patch(cart)
        
        # Create pendulum
        pendulum_line, = ax.plot([], [], 'o-', lw=2, color='red', 
                                markerfacecolor='red', markersize=8)
        
        # Add time display
        time_text = ax.text(-2.8, 1.3, '', fontsize=10)
        
        # Add repeat status display
        repeat_text = ax.text(1.5, 1.3, f"Repeat: {'ON' if repeat else 'OFF'}", fontsize=10, 
                            bbox=dict(facecolor='yellow', alpha=0.5))

        def init():
            """Initialize the animation"""
            pendulum_line.set_data([], [])
            time_text.set_text('')
            return pendulum_line, cart, time_text, repeat_text

        def update(i):
            """Update the animation for each frame"""
            x = self.cart_list[i]
            th = self.theta_list[i]
            
            # Fixed coordinate calculation:
            # For 180° (π rad) being up and 90° (π/2 rad) being right
            x_p = x + self.L * np.sin(th)
            y_p = -self.L * np.cos(th)

            # Update cart position
            cart.set_xy((x - cart_width/2, -cart_height/2))
            
            # Update pendulum line
            pendulum_line.set_data([x, x_p], [0, y_p])
            
            # Update time text
            time_text.set_text(f'Time: {i*self.dt:.2f}s, Angle: {np.degrees(th):.1f}°')
            
            return pendulum_line, cart, time_text, repeat_text

        self.ani = animation.FuncAnimation(
            fig, update, frames=len(self.theta_list), init_func=init, 
            blit=True, interval=20, repeat=repeat
        )
        
        plt.title('Single Inverted Pendulum Simulation')
        plt.xlabel('Position (m)')
        plt.ylabel('Height (m)')
        plt.show()
        
        return self.ani
    
    def set_controller_params(self, Kp=None, Kd=None):
        """Update the controller parameters"""
        if Kp is not None:
            self.Kp = Kp
        if Kd is not None:
            self.Kd = Kd
            
    def set_initial_conditions(self, theta_deg=None, dtheta=None, x_cart=None):
        """Set initial conditions for the simulation"""
        if theta_deg is not None:
            self.theta = np.radians(theta_deg)
        if dtheta is not None:
            self.dtheta = dtheta
        if x_cart is not None:
            self.x_cart = x_cart
    
    def reset(self):
        """Reset the simulation to initial values"""
        self.theta = np.radians(190)  # Reset to near upright
        self.dtheta = 0.0
        self.x_cart = 0.0
        
        # Clear stored data
        self.theta_list = []
        self.theta_list_plot = []
        self.cart_list = []
        self.v_list = []
        self.t_list = []


# Example usage
if __name__ == "__main__":
    # Create simulation instance
    sim = SinglePendulumSimulation()
    
    # Run simulation with default parameters
    sim.run_simulation(500)
    
    # Plot results
    sim.plot_results()
    
    # Display animation (default: no repeat)
    sim.animate_pendulum()
    
    # Uncomment to run animation with repeat enabled:
    # sim.animate_pendulum(repeat=True)
    
    # Example: changing parameters and running again
    # sim.reset()
    # sim.set_controller_params(Kp=60.0, Kd=12.0)
    # sim.set_initial_conditions(theta_deg=170)
    # sim.run_simulation(600)
    # sim.plot_results()
    # sim.animate_pendulum(repeat=True)
