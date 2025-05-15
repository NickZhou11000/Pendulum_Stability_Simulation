from math import pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQNModel(nn.Module):
    """Neural Network for Deep Q-learning"""
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        # Larger network with more capacity
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)  # Increased memory size
        self.gamma = 0.99    # Higher discount rate for longer-term rewards
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997  # Slower decay for more exploration
        self.learning_rate = 0.0005  # Lower learning rate for stability
        
        # Check CUDA availability and version
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            print(f"PyTorch is using CUDA version: {cuda_version}")
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, using CPU")
        
        print(f"Using device: {self.device}")
        
        # Create Q networks
        self.model = DQNModel(state_size, action_size).to(self.device)
        self.target_model = DQNModel(state_size, action_size).to(self.device)
        self.update_target_model()
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        """Choose action epsilon-greedily"""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).to(self.device)
        self.model.eval()
        with torch.no_grad():
            action_values = self.model(state_tensor)
        self.model.train()
        
        return torch.argmax(action_values).item()

    def replay(self, batch_size):
        """Train the model using randomly sampled memories"""
        if len(self.memory) < batch_size:
            return
            
        # Sample mini-batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor(np.array([x[0][0] for x in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([x[1] for x in minibatch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([x[2] for x in minibatch])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([x[3][0] for x in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([x[4] for x in minibatch])).unsqueeze(1).to(self.device)
        
        # Compute Q values
        self.model.eval()
        with torch.no_grad():
            # Get max predicted Q values for next states from target model
            next_q_values = self.target_model(next_states)
            next_q_values_max = next_q_values.max(1)[0].unsqueeze(1)
            # Compute target Q values
            target_q_values = rewards + (self.gamma * next_q_values_max * (1 - dones))
        
        # Get current Q values from model for states and actions
        self.model.train()
        current_q_values = self.model(states).gather(1, actions)
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class SinglePendulumSimulationRL:
    def __init__(self):
        # Constants for single inverted pendulum
        self.g = 9.81
        self.L = 0.5
        self.m = 0.1
        self.b = 0.05
        self.dt = 0.02
        self.v_max = 2.0
        
        # Discrete action space (number of possible force values)
        self.n_actions = 11  # -v_max to +v_max in n steps
        
        # State space: [theta, dtheta, x_cart, dx_cart]
        self.state_size = 4
        self.action_size = self.n_actions
        
        # Action space mapping
        self.actions = np.linspace(-self.v_max, self.v_max, self.n_actions)
        
        # RL agent
        self.agent = DQNAgent(self.state_size, self.action_size)
        
        # Initial state
        self.reset()
        
        # Data storage
        self.theta_list = []
        self.theta_list_plot = []
        self.cart_list = []
        self.v_list = []
        self.reward_list = []
        self.t_list = []
        
        # Animation properties
        self.ani = None

    def reset(self):
        """Reset the simulation to initial values"""
        # Random initial angle near upright
        self.theta = np.radians(180 + np.random.uniform(-30, 30))
        self.dtheta = np.random.uniform(-0.5, 0.5)
        self.x_cart = np.random.uniform(-0.5, 0.5)
        self.dx_cart = 0.0
        
        # Clear stored data
        self.theta_list = []
        self.theta_list_plot = []
        self.cart_list = []
        self.v_list = []
        self.reward_list = []
        self.t_list = []
        
        return self._get_state()

    def _get_state(self):
        """Convert the physical state to the RL state vector"""
        # Normalize state values to similar ranges to help the neural network
        theta_norm = ((self.theta % (2*pi)) - pi) / pi  # -1 to 1
        dtheta_norm = np.clip(self.dtheta / 10.0, -1, 1)  # -1 to 1
        x_norm = np.clip(self.x_cart / 3.0, -1, 1)  # -1 to 1
        dx_norm = np.clip(self.dx_cart / 2.0, -1, 1)  # -1 to 1
        
        return np.array([theta_norm, dtheta_norm, x_norm, dx_norm])

    def _calculate_reward(self):
        """Calculate reward based on current state"""
        # Angle from upright position (0 = upright, pi = hanging down)
        angle_diff = abs(((self.theta % (2*pi)) - pi + pi) % (2*pi) - pi)
        
        # More aggressive reward function
        if angle_diff < 0.1:  # Very close to upright
            angle_reward = 10.0
        elif angle_diff < 0.3:  # Somewhat close to upright
            angle_reward = 2.0
        else:
            # Exponential penalty for deviation from upright
            angle_reward = -0.1 * (angle_diff / pi) ** 2
        
        # Stronger penalty for being far from center
        position_penalty = -0.5 * min(1.0, abs(self.x_cart) / 1.5)
        
        # Penalty for high velocity (for smoother control)
        velocity_penalty = -0.1 * min(1.0, abs(self.dtheta) / 5.0)
        
        # Penalty for falling or going out of bounds
        failure = abs(self.x_cart) > 2.5 or angle_diff > 0.8 * pi
        
        if failure:
            return -20.0  # Stronger failure penalty
        else:
            return angle_reward + position_penalty + velocity_penalty
    
    def is_done(self):
        """Check if episode is done"""
        angle_diff = abs(((self.theta % (2*pi)) - pi + pi) % (2*pi) - pi)
        return abs(self.x_cart) > 2.5 or angle_diff > 0.8 * pi or len(self.t_list) >= 500
    
    def simulate_step(self, action_idx):
        """Take one step in the environment"""
        v = self.actions[action_idx]
        
        # Physics simulation
        ddtheta = -self.g / self.L * np.sin(self.theta) - self.b / (self.m * self.L**2) * self.dtheta - v / self.L * np.cos(self.theta)
        self.dtheta += ddtheta * self.dt
        self.theta += self.dtheta * self.dt
        self.dx_cart = v
        self.x_cart += self.dx_cart * self.dt
        
        # Store data
        if len(self.t_list) == 0:
            self.t_list.append(0)
        else:
            self.t_list.append(self.t_list[-1] + self.dt)
        
        self.theta_list.append(self.theta)
        self.theta_list_plot.append(np.degrees(self.theta))
        self.cart_list.append(self.x_cart)
        self.v_list.append(v)
        
        # Calculate reward
        reward = self._calculate_reward()
        self.reward_list.append(reward)
        
        # Check if done
        done = self.is_done()
        
        return self._get_state(), reward, done

    def train(self, episodes=1000, batch_size=64, target_update=5):
        """Train the agent"""
        try:
            # Try to use CUDA if available
            if torch.cuda.is_available():
                print("Training with CUDA acceleration")
            else:
                print("Training on CPU (CUDA not available)")
                
            rewards = []
            best_reward = -float('inf')
            
            for e in range(episodes):
                # Reset environment
                state = self.reset()
                state = np.reshape(state, [1, self.state_size])
                total_reward = 0
                
                # For tracking episode performance
                steps = 0
                max_angle_diff = 0
                
                while True:
                    # Get action
                    action = self.agent.act(state)
                    
                    # Take action
                    next_state, reward, done = self.simulate_step(action)
                    next_state = np.reshape(next_state, [1, self.state_size])
                    
                    # Remember experience
                    self.agent.remember(state, action, reward, next_state, done)
                    
                    # Update state and reward
                    state = next_state
                    total_reward += reward
                    steps += 1
                    
                    # Track maximum angle difference from upright
                    angle_diff = abs(((self.theta % (2*pi)) - pi + pi) % (2*pi) - pi)
                    max_angle_diff = max(max_angle_diff, angle_diff)
                    
                    # Train agent - multiple mini-batch updates per step for better sample efficiency
                    if len(self.agent.memory) > batch_size * 10:
                        for _ in range(4):
                            self.agent.replay(batch_size)
                    
                    if done:
                        break
                
                # Update target network more frequently in early training
                if e < 100 and e % 2 == 0:
                    self.agent.update_target_model()
                elif e % target_update == 0:
                    self.agent.update_target_model()
                
                rewards.append(total_reward)
                
                # Save best model
                if total_reward > best_reward and steps > 200:  # Only save if reasonably stable
                    best_reward = total_reward
                    self.save_model('pendulum_rl_best_model.pt')
                    print(f"New best model with reward: {best_reward:.2f}")
                
                if e % 10 == 0:
                    print(f"Episode: {e}/{episodes}, Steps: {steps}, Reward: {total_reward:.2f}, " +
                          f"Max angle diff: {np.degrees(max_angle_diff):.1f}°, Epsilon: {self.agent.epsilon:.4f}")
            
            return rewards
            
        except RuntimeError as e:
            if "CUDA" in str(e):
                print("\nCUDA Error encountered. This may be due to missing or incompatible CUDA installation.")
                print("Please follow the installation instructions below:\n")
                print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
                print("\nFor your CUDA 12.8, the PyTorch CUDA 12.1 build should be compatible.")
                print("After installation, restart your Python environment.")
                raise Exception("PyTorch CUDA installation required") from e
            else:
                raise e

    def run_trained_agent(self, max_steps=500, initial_angle_deg=170):
        """Run the environment with the trained agent"""
        # Reset environment with specific initial angle for testing
        self.theta = np.radians(initial_angle_deg)
        self.dtheta = 0.0
        self.x_cart = 0.0
        self.dx_cart = 0.0
        
        # Clear stored data
        self.theta_list = []
        self.theta_list_plot = []
        self.cart_list = []
        self.v_list = []
        self.reward_list = []
        self.t_list = []
        
        state = self._get_state()
        state = np.reshape(state, [1, self.state_size])
        
        for _ in range(max_steps):
            # Get action (without exploration)
            action = self.agent.act(state, training=False)
            
            # Take action
            next_state, reward, done = self.simulate_step(action)
            
            # Update state
            state = np.reshape(next_state, [1, self.state_size])
            
            if done:
                break
                
        return self.t_list, self.theta_list_plot, self.cart_list, self.v_list, self.reward_list

    def plot_results(self):
        """Plot the simulation results"""
        if not self.t_list:
            print("No simulation data. Run the simulation first.")
            return
            
        plt.figure(figsize=(12, 10))

        plt.subplot(4, 1, 1)
        plt.plot(self.t_list, self.theta_list_plot)
        plt.axhline(180, color='gray', linestyle='--', label="Upright position")
        plt.ylabel("Angle (°)")
        plt.title("Single Pendulum with Reinforcement Learning")
        plt.grid(True)
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(self.t_list, self.cart_list, color="green")
        plt.ylabel("Position (m)")
        plt.grid(True)

        plt.subplot(4, 1, 3)
        plt.plot(self.t_list, self.v_list, color="red")
        plt.ylabel("Control (m/s)")
        plt.grid(True)
        
        plt.subplot(4, 1, 4)
        plt.plot(self.t_list, self.reward_list, color="purple")
        plt.xlabel("Time (s)")
        plt.ylabel("Reward")
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        
    def plot_training_progress(self, rewards):
        """Plot the training progress"""
        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.title('Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.show()

    def animate_pendulum(self, repeat=False):
        """Create and display animation of the pendulum"""
        if not self.theta_list:
            print("No simulation data. Run the simulation first.")
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(-3, 3)
        ax.set_ylim(-1, 1.5)
        ax.set_aspect("equal")
        ax.grid(True)
        
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
        
        # Add time and reward display
        info_text = ax.text(-2.8, 1.3, '', fontsize=10)
        
        # Add repeat status display
        repeat_text = ax.text(1.5, 1.3, f"Repeat: {'ON' if repeat else 'OFF'}", fontsize=10,
                             bbox=dict(facecolor='yellow', alpha=0.5))

        def init():
            """Initialize the animation"""
            pendulum_line.set_data([], [])
            info_text.set_text('')
            return pendulum_line, cart, info_text, repeat_text

        def update(i):
            """Update the animation for each frame"""
            if i >= len(self.cart_list):
                return pendulum_line, cart, info_text, repeat_text
                
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
            
            # Update info text
            reward = self.reward_list[i] if i < len(self.reward_list) else 0
            info_text.set_text(f'Time: {self.t_list[i]:.2f}s, Angle: {np.degrees(th):.1f}°, Reward: {reward:.2f}')
            
            return pendulum_line, cart, info_text, repeat_text

        self.ani = animation.FuncAnimation(
            fig, update, frames=len(self.theta_list), init_func=init, 
            blit=True, interval=20, repeat=repeat
        )
        
        plt.title('Single Inverted Pendulum with RL Control')
        plt.xlabel('Position (m)')
        plt.ylabel('Height (m)')
        plt.show()
        
        return self.ani

    def save_model(self, filepath):
        """Save the trained model"""
        torch.save(self.agent.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.agent.model.load_state_dict(torch.load(filepath))
        self.agent.update_target_model()
        print(f"Model loaded from {filepath}")
        

# Example usage
if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)  # Set seed for all GPUs
        
    # Print PyTorch and CUDA information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    
    # Create RL simulation
    sim = SinglePendulumSimulationRL()
    
    # Train the model with better parameters (uncomment to train)
    # rewards = sim.train(episodes=1000, batch_size=256, target_update=1)
    # sim.plot_training_progress(rewards)
    # sim.save_model('pendulum_rl_final_model.pt')
    
    # Load a pre-trained model if available
    sim.load_model('pendulum_rl_final_model.pt')
    
    # Run the trained agent
    print("Running simulation with the trained agent...")
    sim.run_trained_agent(initial_angle_deg=170)  # Start with a specific angle
    
    # Plot results
    sim.plot_results()
    
    # Display animation
    sim.animate_pendulum()
