from simulator.controllers.Pid import Pid, PidTunings, PidLimits
from simulator.examples.TemperaturePlant import TemperaturePlant
from simulator.math.QNumber import QNumber
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, TypeVar
import logging

T = TypeVar('T', float, QNumber)

class ScaledPid(Pid):
    def __init__(self, 
                scale_factor: T,
                tunnings: PidTunings[T],
                limits: PidLimits[T],
                auto_mode: bool = True):
        super().__init__(tunnings=tunnings, limits=limits, auto_mode=auto_mode)

        if isinstance(scale_factor, QNumber) and scale_factor.to_float() <= 0:
            raise ValueError("Scale factor must be positive")
        if isinstance(scale_factor, float) and scale_factor <= 0:
            raise ValueError("Scale factor must be positive")

        self.scale_factor = scale_factor
        self.squared_scale_factor = self.scale_factor * self.scale_factor

    def set_point(self, value: float):
        if isinstance(self.scale_factor, float):
            super().set_point(value * self.scale_factor)
        else:
            super().set_point(QNumber(value * self.scale_factor.to_float(), self.scale_factor.fractional_bits))

    def process(self, measured_process_variable: float) -> float:
        if isinstance(self.scale_factor, float):
            return super().process(measured_process_variable * self.scale_factor) / self.squared_scale_factor
        else:
            input = QNumber(measured_process_variable * self.scale_factor.to_float(), self.scale_factor.fractional_bits)
            output = super().process(input).to_float()
            return output / self.squared_scale_factor.to_float()

class PidSimulator:
    def __init__(self, sample_time: float = 0.1, simulation_steps: int = 700):
        self.sample_time = sample_time
        self.simulation_steps = simulation_steps
        self.setpoint_schedule = [
            (0, 20),    # Start at room temperature
            (100, 60),  # Heat to 60°C
            (300, 40),  # Cool to 40°C
            (500, 50)   # Final setpoint 50°C
        ]
        self.configurations = {
            'float': {'type': 'float'},
            'Q15': {'type': 'fixed', 'bits': 15},
            'Q31': {'type': 'fixed', 'bits': 31}
        }
        self.results = {}

    def _create_scaled_pid(self, numeric_type: str, fractional_bits: int = None) -> Pid:
        """Create a PID controller with appropriate numeric type and scaling."""    
        if numeric_type == 'float':
            scale = 0.01
            tunnings = PidTunings(
                kp=1.0 * scale,
                ki=0.1 * scale,
                kd=0.35 * scale
            )
            limits = PidLimits(
                min=-99.0 * scale, 
                max=99.0 * scale
            )
        else:
            scale = QNumber(0.01, fractional_bits)
            tunnings = PidTunings(
                kp=QNumber(1.0 * scale.to_float(), fractional_bits),
                ki=QNumber(0.1 * scale.to_float(), fractional_bits),
                kd=QNumber(0.35 * scale.to_float(), fractional_bits)
            )
            limits = PidLimits(
                min=QNumber(-99.0 * scale.to_float(), fractional_bits),
                max=QNumber(99.0 * scale.to_float(), fractional_bits)
            )
        
        return ScaledPid(scale_factor=scale, tunnings=tunnings, limits=limits)

    def _run_comparison(self):
        """Run the PID simulation comparison."""
        for name, config in self.configurations.items():
            numeric_type = 'float' if config['type'] == 'float' else 'fixed'
            fractional_bits = config.get('bits')
            
            logging.debug('numeric_type: %s, fractional_bits: %s', numeric_type, fractional_bits)

            controller = self._create_scaled_pid(numeric_type, fractional_bits)
            plant = TemperaturePlant(self.sample_time)
            
            time_points = np.arange(self.simulation_steps) * self.sample_time
            setpoints = np.zeros(self.simulation_steps)
            temperatures = np.zeros(self.simulation_steps)
            control_signals = np.zeros(self.simulation_steps)
            
            temperatures[0] = plant.last_output()
            
            current_setpoint = self.setpoint_schedule[0][1]
            current_schedule_idx = 0
            
            for step in range(self.simulation_steps):
                if (current_schedule_idx < len(self.setpoint_schedule) - 1 and 
                    step >= self.setpoint_schedule[current_schedule_idx + 1][0]):
                    current_schedule_idx += 1
                    current_setpoint = self.setpoint_schedule[current_schedule_idx][1]
                
                setpoints[step] = current_setpoint
                
                controller.set_point(current_setpoint)
                control_signal = controller.process(temperatures[step])
                control_signals[step] = control_signal
                
                if step < self.simulation_steps - 1:
                    temperatures[step + 1] = plant.update(control_signal)
            
            self.results[name] = {
                'time': time_points,
                'setpoint': setpoints,
                'temperature': temperatures,
                'control_signal': control_signals
            }

    def plot_results(self):
        """Plot the simulation results."""
        if not self.results:
            raise ValueError("No results to plot. Run the simulation first.")

        # Define colors for each configuration
        colors = ['b', 'r', 'g', 'c', 'm', 'y']  # Add more colors if needed
        config_colors = {name: colors[i % len(colors)] for i, name in enumerate(self.configurations.keys())}

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Use the first configuration's time and setpoint for reference line
        first_config = next(iter(self.configurations.keys()))
        ax1.plot(self.results[first_config]['time'], self.results[first_config]['setpoint'], 
                'k--', label='Setpoint')
        
        # Plot temperature data for each configuration
        for name in self.configurations.keys():
            ax1.plot(self.results[name]['time'], self.results[name]['temperature'], 
                    color=config_colors[name], label=f'Temperature ({name})')
        
        ax1.grid(True)
        ax1.set_ylabel('Temperature (°C)')
        ax1.legend()
        ax1.set_title('Temperature Control Comparison')
        
        # Plot control signals for each configuration
        for name in self.configurations.keys():
            ax2.plot(self.results[name]['time'], self.results[name]['control_signal'], 
                    color=config_colors[name], label=f'Control Signal ({name})')
        
        ax2.grid(True)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Control Signal')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

    def run(self):
        """Run the complete simulation and plot results."""
        logging.basicConfig(level=logging.DEBUG)
        self._run_comparison()
        self.plot_results()

if __name__ == "__main__":
    simulator = PidSimulator()
    simulator.run()
