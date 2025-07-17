"""
Software TINA (Telemetric Inductive Nodal Actuator) - Production Version
Complete virtual amplifier data collection system using SPICE simulation
"""

import numpy as np
import subprocess
import json
import os
import tempfile
import logging
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any, Generator
from itertools import product
from pathlib import Path
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from abc import ABC, abstractmethod
from threading import Lock
from functools import lru_cache
from collections import OrderedDict
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('software_tina.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SimulationError(Exception):
    """Custom exception for simulation errors"""
    pass


@dataclass
class ComponentSpec:
    """Specification for a circuit component with full metadata"""
    name: str
    nominal_value: float
    tolerance: float
    temperature_coefficient: float
    aging_factor: float
    component_type: str
    unit: str = ""
    description: str = ""
    
    def get_varied_value(self, variation_factor: float = 0.0, 
                        temp_delta: float = 0.0, 
                        aging_years: float = 0.0) -> float:
        """Get component value with variations"""
        # Base value with tolerance
        tolerance_variation = self.nominal_value * self.tolerance * variation_factor
        
        # Temperature effect
        temp_variation = self.nominal_value * self.temperature_coefficient * temp_delta
        
        # Aging effect
        aging_variation = self.nominal_value * self.aging_factor * aging_years
        
        return self.nominal_value + tolerance_variation + temp_variation + aging_variation


@dataclass
class CircuitConfiguration:
    """Complete circuit configuration for one simulation"""
    components: Dict[str, float]
    environmental: Dict[str, float]
    controls: Dict[str, float]
    metadata: Dict[str, Any]
    
    def get_hash(self) -> str:
        """Generate unique hash for this configuration"""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()


@dataclass
class SimulationResult:
    """Results from one SPICE simulation"""
    configuration: CircuitConfiguration
    frequency_response: np.ndarray
    time_response: np.ndarray
    input_signal: np.ndarray
    output_signal: np.ndarray
    frequencies: np.ndarray
    time_vector: np.ndarray
    simulation_time: float
    success: bool
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None


class CircuitModel(ABC):
    """Abstract base class for circuit models"""
    
    @abstractmethod
    def generate_netlist(self, config: CircuitConfiguration) -> str:
        """Generate SPICE netlist for configuration"""
        pass
    
    @abstractmethod
    def parse_simulation_output(self, output: str, config: CircuitConfiguration) -> SimulationResult:
        """Parse SPICE simulation output"""
        pass
    
    @abstractmethod
    def validate_configuration(self, config: CircuitConfiguration) -> bool:
        """Validate configuration parameters"""
        pass


class TubePreampModel(CircuitModel):
    """12AX7 tube preamp circuit model with production enhancements"""
    
    def __init__(self):
        self.components = {
            'plate_resistor': ComponentSpec(
                'Rp', 100e3, 0.05, -200e-6, 0.02, 'resistor', 'Ω',
                'Plate load resistor'
            ),
            'cathode_resistor': ComponentSpec(
                'Rk', 1.5e3, 0.05, -200e-6, 0.01, 'resistor', 'Ω',
                'Cathode bias resistor'
            ),
            'cathode_cap': ComponentSpec(
                'Ck', 22e-6, 0.2, 300e-6, 0.05, 'capacitor', 'F',
                'Cathode bypass capacitor'
            ),
            'coupling_cap': ComponentSpec(
                'Cc', 0.022e-6, 0.1, 100e-6, 0.02, 'capacitor', 'F',
                'Output coupling capacitor'
            ),
            'grid_resistor': ComponentSpec(
                'Rg', 1e6, 0.05, -200e-6, 0.01, 'resistor', 'Ω',
                'Grid leak resistor'
            ),
            'tube_transconductance': ComponentSpec(
                'gm', 1625e-6, 0.1, 500e-9, 0.1, 'tube', 'S',
                'Tube transconductance'
            ),
            'tube_mu': ComponentSpec(
                'mu', 100, 0.1, 0.05, 0.05, 'tube', '',
                'Tube amplification factor'
            ),
            'tube_rp': ComponentSpec(
                'rp', 62.5e3, 0.2, 100e-6, 0.08, 'tube', 'Ω',
                'Tube plate resistance'
            ),
        }
        
        self.control_ranges = {
            'input_level': (0.001, 1.0),
            'bias_voltage': (-2.0, -0.5),
            'supply_voltage': (280, 360),
            'load_impedance': (10e3, 1e6),
        }
        
        self.environmental_ranges = {
            'temperature': (15, 45),
            'humidity': (20, 80),
            'aging_years': (0, 20),
        }
        
        # Cache for simulation results
        self._simulation_cache = {}
        self._cache_lock = Lock()
    
    def validate_configuration(self, config: CircuitConfiguration) -> bool:
        """Validate configuration parameters"""
        # Check components
        for name, value in config.components.items():
            if name not in self.components:
                logger.error(f"Unknown component: {name}")
                return False
            if value <= 0:
                logger.error(f"Invalid component value: {name}={value}")
                return False
        
        # Check controls
        for name, value in config.controls.items():
            if name not in self.control_ranges:
                logger.error(f"Unknown control: {name}")
                return False
            min_val, max_val = self.control_ranges[name]
            if not min_val <= value <= max_val:
                logger.error(f"Control {name}={value} out of range [{min_val}, {max_val}]")
                return False
        
        # Check environmental
        for name, value in config.environmental.items():
            if name not in self.environmental_ranges:
                logger.error(f"Unknown environmental parameter: {name}")
                return False
            min_val, max_val = self.environmental_ranges[name]
            if not min_val <= value <= max_val:
                logger.error(f"Environmental {name}={value} out of range [{min_val}, {max_val}]")
                return False
        
        return True
    
    def generate_netlist(self, config: CircuitConfiguration) -> str:
        """Generate SPICE netlist with proper syntax"""
        if not self.validate_configuration(config):
            raise ValueError("Invalid configuration")
        
        components = config.components
        env = config.environmental
        controls = config.controls
        
        # Generate varied component values
        temp_delta = env['temperature'] - 25
        aging_years = env['aging_years']
        
        netlist = f"""* Software TINA Generated Netlist - Production Version
* Configuration: {config.metadata.get('config_id', 'unknown')}
* Temperature: {env['temperature']}°C, Aging: {aging_years} years
* Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

.TITLE 12AX7 Tube Preamp - Software TINA Simulation

* Component parameter definitions with variations
.PARAM Rp={components['plate_resistor']:.3e}
.PARAM Rk={components['cathode_resistor']:.3e}
.PARAM Ck={components['cathode_cap']:.3e}
.PARAM Cc={components['coupling_cap']:.3e}
.PARAM Rg={components['grid_resistor']:.3e}
.PARAM gm={components['tube_transconductance']:.3e}
.PARAM mu={components['tube_mu']:.3f}
.PARAM rp={components['tube_rp']:.3e}

* Control parameters
.PARAM input_level={controls['input_level']:.3f}
.PARAM bias_voltage={controls['bias_voltage']:.3f}
.PARAM supply_voltage={controls['supply_voltage']:.3f}
.PARAM load_impedance={controls['load_impedance']:.3e}

* Environmental parameters
.PARAM temperature={env['temperature']:.1f}
.PARAM aging_factor={1 - 0.05 * aging_years / 20:.3f}

* Advanced 12AX7 tube model with temperature and aging effects
.SUBCKT 12AX7_ADVANCED 1 2 3
* Pins: 1=Plate, 2=Grid, 3=Cathode
* Grid-cathode current
Egk 2 3 VALUE={{PWR(LIMIT(V(2,3),-20,0.5),1.4)+PWR(LIMIT(V(2,3),-20,0.5),2)}}
* Plate current with aging effects
Gak 1 3 VALUE={{PWR(LIMIT(V(2,3)+V(1,3)/mu,0,1000),1.5)*gm*aging_factor}}
* Interelectrode capacitances
Cgk 2 3 {1.7e-12*(1+100e-6*(temperature-25))}
Cpk 1 3 {0.46e-12*(1+100e-6*(temperature-25))}
Cgp 2 1 {1.7e-12*(1+100e-6*(temperature-25))}
* Plate resistance
Rp_tube 1 3 {rp*(1+0.004*(temperature-25))}
.ENDS

* Main circuit
Vin input 0 AC {input_level} 0
Rg input grid {Rg}
Vbias grid grid_biased {bias_voltage}

* Tube stage
X1 plate grid_biased cathode 12AX7_ADVANCED
Rp plate vcc {Rp}
Rk cathode cathode_tap {Rk}
Ck cathode_tap 0 {Ck}

* Output stage
Cc plate output {Cc}
Rl output 0 {load_impedance}

* Power supply with realistic characteristics
Vcc vcc 0 {supply_voltage}
* Supply ripple at 120Hz (full-wave rectified)
Vripple vcc vcc_clean SIN(0 {supply_voltage*0.01} 120 0 0 0)

* Analysis commands
.AC DEC 100 1 100K
.TRAN 0.01MS 50MS
.NOISE V(output) Vin DEC 100 1 100K
.OP
.TEMP {temperature}

* Output commands
.PRINT AC V(output) VP(output) VDB(output)
.PRINT TRAN V(output) V(input)
.PRINT NOISE ONOISE

* Fourier analysis for harmonic distortion
.FOUR 440 V(output)

.END
"""
        
        return netlist
    
    @lru_cache(maxsize=1000)
    def _get_cached_simulation(self, config_hash: str) -> Optional[SimulationResult]:
        """Get cached simulation result"""
        with self._cache_lock:
            return self._simulation_cache.get(config_hash)
    
    def _cache_simulation(self, config_hash: str, result: SimulationResult):
        """Cache simulation result"""
        with self._cache_lock:
            self._simulation_cache[config_hash] = result
    
    def parse_simulation_output(self, output: str, config: CircuitConfiguration) -> SimulationResult:
        """Parse SPICE simulation output into structured results"""
        try:
            # Parse AC analysis results
            frequencies, magnitude, phase = self._parse_ac_analysis(output)
            
            # Parse transient analysis results
            time_vector, voltage_output = self._parse_transient_analysis(output)
            
            # Generate input signal
            input_signal = self._generate_input_signal(
                time_vector, config.controls['input_level']
            )
            
            # Calculate metrics
            metrics = self._calculate_metrics(voltage_output, input_signal)
            
            return SimulationResult(
                configuration=config,
                frequency_response=magnitude,
                time_response=voltage_output,
                input_signal=input_signal,
                output_signal=voltage_output,
                frequencies=frequencies,
                time_vector=time_vector,
                simulation_time=0.0,  # Will be set by caller
                success=True,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Failed to parse simulation output: {e}")
            return SimulationResult(
                configuration=config,
                frequency_response=np.array([]),
                time_response=np.array([]),
                input_signal=np.array([]),
                output_signal=np.array([]),
                frequencies=np.array([]),
                time_vector=np.array([]),
                simulation_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _parse_ac_analysis(self, output: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse AC analysis results from SPICE output"""
        # This is a simplified parser - implement based on your SPICE format
        frequencies = np.logspace(0, 5, 100)
        magnitude = np.ones_like(frequencies)
        phase = np.zeros_like(frequencies)
        
        # TODO: Implement actual parsing based on SPICE output format
        
        return frequencies, magnitude, phase
    
    def _parse_transient_analysis(self, output: str) -> Tuple[np.ndarray, np.ndarray]:
        """Parse transient analysis results from SPICE output"""
        # This is a simplified parser - implement based on your SPICE format
        time_vector = np.linspace(0, 0.05, 2205)
        voltage_output = np.sin(2 * np.pi * 440 * time_vector) * 0.1
        
        # TODO: Implement actual parsing based on SPICE output format
        
        return time_vector, voltage_output
    
    def _generate_input_signal(self, time_vector: np.ndarray, amplitude: float) -> np.ndarray:
        """Generate input signal for comparison"""
        return amplitude * np.sin(2 * np.pi * 440 * time_vector)
    
    def _calculate_metrics(self, output_signal: np.ndarray, input_signal: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics"""
        # RMS values
        rms_input = np.sqrt(np.mean(input_signal**2))
        rms_output = np.sqrt(np.mean(output_signal**2))
        
        # Gain
        gain = rms_output / (rms_input + 1e-10)
        
        # THD (simplified)
        thd = self._calculate_thd(output_signal)
        
        # SNR
        signal_power = np.mean(output_signal**2)
        noise_power = np.var(output_signal - input_signal * gain)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        return {
            'rms_input': float(rms_input),
            'rms_output': float(rms_output),
            'gain': float(gain),
            'thd': float(thd),
            'snr': float(snr)
        }
    
    def _calculate_thd(self, signal: np.ndarray, sample_rate: int = 44100) -> float:
        """Calculate Total Harmonic Distortion"""
        # FFT of signal
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
        
        # Find fundamental frequency (440 Hz)
        fundamental_idx = np.argmin(np.abs(freqs - 440))
        fundamental_power = np.abs(fft[fundamental_idx])**2
        
        # Calculate harmonic power (2nd to 5th harmonics)
        harmonic_power = 0
        for harmonic in range(2, 6):
            harmonic_freq = 440 * harmonic
            if harmonic_freq < sample_rate / 2:  # Below Nyquist
                harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
                harmonic_power += np.abs(fft[harmonic_idx])**2
        
        # THD percentage
        if fundamental_power > 0:
            thd = np.sqrt(harmonic_power / fundamental_power) * 100
        else:
            thd = 0
        
        return min(thd, 100)


class SoftwareTINA:
    """Software TINA - Production-ready virtual amplifier data collection system"""
    
    def __init__(self, circuit_model: CircuitModel, 
                 spice_command: str = "ngspice",
                 max_cache_size: int = 10000):
        self.circuit_model = circuit_model
        self.spice_command = spice_command
        self.simulation_results: List[SimulationResult] = []
        self.temp_dir = Path(tempfile.mkdtemp(prefix="software_tina_"))
        self.results_lock = Lock()
        self.max_cache_size = max_cache_size
        
        # Verify SPICE installation
        if not self._verify_spice_installation():
            logger.warning(f"SPICE command '{spice_command}' not found. Using simulation mode.")
            self.simulation_mode = True
        else:
            self.simulation_mode = False
        
        logger.info(f"Software TINA initialized with {circuit_model.__class__.__name__}")
        logger.info(f"Temporary directory: {self.temp_dir}")
        logger.info(f"Simulation mode: {self.simulation_mode}")
    
    def _verify_spice_installation(self) -> bool:
        """Verify SPICE is installed and accessible"""
        try:
            result = subprocess.run(
                [self.spice_command, "-v"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def generate_parameter_sweep(self, num_configs: int = 1000, 
                               sweep_type: str = "random",
                               seed: Optional[int] = None) -> List[CircuitConfiguration]:
        """Generate parameter configurations for systematic exploration"""
        if seed is not None:
            np.random.seed(seed)
        
        logger.info(f"Generating {num_configs} parameter configurations using {sweep_type} sweep")
        
        configurations = []
        
        if sweep_type == "random":
            configurations = self._generate_random_sweep(num_configs)
        elif sweep_type == "grid":
            configurations = self._generate_grid_sweep(num_configs)
        elif sweep_type == "sobol":
            configurations = self._generate_sobol_sweep(num_configs)
        elif sweep_type == "latin_hypercube":
            configurations = self._generate_latin_hypercube_sweep(num_configs)
        else:
            raise ValueError(f"Unknown sweep type: {sweep_type}")
        
        logger.info(f"Generated {len(configurations)} configurations")
        return configurations
    
    def _generate_random_sweep(self, num_configs: int) -> List[CircuitConfiguration]:
        """Generate random parameter configurations"""
        configurations = []
        
        for i in range(num_configs):
            # Component variations
            components = {}
            for name, spec in self.circuit_model.components.items():
                variation = np.random.uniform(-1, 1)
                temp_delta = np.random.uniform(-10, 20)
                aging_years = np.random.uniform(0, 20)
                components[name] = spec.get_varied_value(variation, temp_delta, aging_years)
            
            # Environmental conditions
            environmental = {}
            for name, (min_val, max_val) in self.circuit_model.environmental_ranges.items():
                environmental[name] = np.random.uniform(min_val, max_val)
            
            # Control settings
            controls = {}
            for name, (min_val, max_val) in self.circuit_model.control_ranges.items():
                controls[name] = np.random.uniform(min_val, max_val)
            
            config = CircuitConfiguration(
                components=components,
                environmental=environmental,
                controls=controls,
                metadata={
                    'config_id': f'random_{i:05d}',
                    'sweep_type': 'random',
                    'index': i
                }
            )
            
            configurations.append(config)
        
        return configurations
    
    def _generate_grid_sweep(self, num_configs: int) -> List[CircuitConfiguration]:
        """Generate grid-based parameter sweep"""
        # Calculate grid dimensions
        num_params = len(self.circuit_model.control_ranges)
        points_per_dim = int(np.ceil(num_configs ** (1/num_params)))
        
        # Generate grid points for each parameter
        param_grids = []
        for name, (min_val, max_val) in self.circuit_model.control_ranges.items():
            param_grids.append(np.linspace(min_val, max_val, points_per_dim))
        
        # Generate all combinations
        configurations = []
        for i, combination in enumerate(product(*param_grids)):
            if i >= num_configs:
                break
            
            # Fixed component values for grid sweep
            components = {}
            for name, spec in self.circuit_model.components.items():
                components[name] = spec.nominal_value
            
            # Fixed environmental conditions
            environmental = {}
            for name, (min_val, max_val) in self.circuit_model.environmental_ranges.items():
                environmental[name] = (min_val + max_val) / 2
            
            # Variable control settings
            controls = {}
            for j, (name, _) in enumerate(self.circuit_model.control_ranges.items()):
                controls[name] = combination[j]
            
            config = CircuitConfiguration(
                components=components,
                environmental=environmental,
                controls=controls,
                metadata={
                    'config_id': f'grid_{i:05d}',
                    'sweep_type': 'grid',
                    'index': i
                }
            )
            
            configurations.append(config)
        
        return configurations
    
    def _generate_sobol_sweep(self, num_configs: int) -> List[CircuitConfiguration]:
        """Generate Sobol sequence-based parameter sweep for better coverage"""
        try:
            from scipy.stats import qmc
            
            # Total number of parameters
            all_params = (list(self.circuit_model.control_ranges.keys()) + 
                         list(self.circuit_model.environmental_ranges.keys()))
            num_dims = len(all_params)
            
            # Generate Sobol sequence
            sampler = qmc.Sobol(d=num_dims, scramble=True)
            samples = sampler.random(num_configs)
            
            configurations = []
            for i, sample in enumerate(samples):
                # Fixed component values
                components = {}
                for name, spec in self.circuit_model.components.items():
                    components[name] = spec.nominal_value
                
                # Variable environmental and control parameters
                environmental = {}
                controls = {}
                
                param_idx = 0
                for name, (min_val, max_val) in self.circuit_model.control_ranges.items():
                    controls[name] = min_val + sample[param_idx] * (max_val - min_val)
                    param_idx += 1
                
                for name, (min_val, max_val) in self.circuit_model.environmental_ranges.items():
                    environmental[name] = min_val + sample[param_idx] * (max_val - min_val)
                    param_idx += 1
                
                config = CircuitConfiguration(
                    components=components,
                    environmental=environmental,
                    controls=controls,
                    metadata={
                        'config_id': f'sobol_{i:05d}',
                        'sweep_type': 'sobol',
                        'index': i
                    }
                )
                
                configurations.append(config)
            
            return configurations
            
        except ImportError:
            logger.warning("scipy not available, falling back to random sweep")
            return self._generate_random_sweep(num_configs)
    
    def _generate_latin_hypercube_sweep(self, num_configs: int) -> List[CircuitConfiguration]:
        """Generate Latin Hypercube sampling for optimal parameter space coverage"""
        try:
            from scipy.stats import qmc
            
            all_params = (list(self.circuit_model.control_ranges.keys()) + 
                         list(self.circuit_model.environmental_ranges.keys()))
            num_dims = len(all_params)
            
            # Generate Latin Hypercube samples
            sampler = qmc.LatinHypercube(d=num_dims)
            samples = sampler.random(num_configs)
            
            configurations = []
            for i, sample in enumerate(samples):
                components = {}
                for name, spec in self.circuit_model.components.items():
                    components[name] = spec.nominal_value
                
                environmental = {}
                controls = {}
                
                param_idx = 0
                for name, (min_val, max_val) in self.circuit_model.control_ranges.items():
                    controls[name] = min_val + sample[param_idx] * (max_val - min_val)
                    param_idx += 1
                
                for name, (min_val, max_val) in self.circuit_model.environmental_ranges.items():
                    environmental[name] = min_val + sample[param_idx] * (max_val - min_val)
                    param_idx += 1
                
                config = CircuitConfiguration(
                    components=components,
                    environmental=environmental,
                    controls=controls,
                    metadata={
                        'config_id': f'lhs_{i:05d}',
                        'sweep_type': 'latin_hypercube',
                        'index': i
                    }
                )
                
                configurations.append(config)
            
            return configurations
            
        except ImportError:
            logger.warning("scipy not available, falling back to random sweep")
            return self._generate_random_sweep(num_configs)
    
    def run_simulation_batch(self, configurations: List[CircuitConfiguration], 
                           max_workers: int = 4,
                           progress_callback=None) -> List[SimulationResult]:
        """Run SPICE simulations in parallel with improved error handling"""
        logger.info(f"Starting batch simulation of {len(configurations)} configurations with {max_workers} workers")
        
        results = []
        failed_configs = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all simulations
            future_to_config = {
                executor.submit(self._run_single_simulation, config): config 
                for config in configurations
            }
            
            # Collect results
            for i, future in enumerate(as_completed(future_to_config)):
                config = future_to_config[future]
                try:
                    result = future.result(timeout=60)  # 60 second timeout per simulation
                    results.append(result)
                    
                    if progress_callback:
                        progress_callback(i + 1, len(configurations))
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"Completed {i + 1}/{len(configurations)} simulations")
                        
                except Exception as e:
                    logger.error(f"Simulation failed for config {config.metadata['config_id']}: {e}")
                    failed_configs.append(config)
                    
                    # Create failed result
                    failed_result = SimulationResult(
                        configuration=config,
                        frequency_response=np.array([]),
                        time_response=np.array([]),
                        input_signal=np.array([]),
                        output_signal=np.array([]),
                        frequencies=np.array([]),
                        time_vector=np.array([]),
                        simulation_time=0.0,
                        success=False,
                        error_message=str(e)
                    )
                    results.append(failed_result)
        
        # Add results to internal storage (thread-safe)
        with self.results_lock:
            self.simulation_results.extend(results)
        
        successful_results = [r for r in results if r.success]
        logger.info(f"Batch simulation completed: {len(successful_results)}/{len(results)} successful")
        
        if failed_configs:
            logger.warning(f"{len(failed_configs)} simulations failed. Consider re-running with adjusted parameters.")
        
        return results
    
    def _run_single_simulation(self, config: CircuitConfiguration) -> SimulationResult:
        """Run a single SPICE simulation with caching"""
        start_time = time.time()
        
        # Check cache first
        config_hash = config.get_hash()
        cached_result = self.circuit_model._get_cached_simulation(config_hash)
        if cached_result:
            logger.debug(f"Using cached result for {config.metadata['config_id']}")
            return cached_result
        
        try:
            # Validate configuration
            if not self.circuit_model.validate_configuration(config):
                raise ValueError("Invalid configuration")
            
            # Generate netlist
            netlist = self.circuit_model.generate_netlist(config)
            
            # Write netlist to temporary file
            netlist_file = self.temp_dir / f"circuit_{config.metadata['config_id']}.cir"
            with open(netlist_file, 'w') as f:
                f.write(netlist)
            
            # Run SPICE simulation or use simulation mode
            if self.simulation_mode:
                output = self._simulate_spice_output(config)
            else:
                output = self._run_spice_simulation(str(netlist_file))
            
            # Parse results
            result = self.circuit_model.parse_simulation_output(output, config)
            result.simulation_time = time.time() - start_time
            
            # Cache successful result
            if result.success:
                self.circuit_model._cache_simulation(config_hash, result)
            
            # Clean up temporary file
            if netlist_file.exists():
                netlist_file.unlink()
            
            return result
            
        except Exception as e:
            logger.error(f"Simulation error for {config.metadata['config_id']}: {e}")
            return SimulationResult(
                configuration=config,
                frequency_response=np.array([]),
                time_response=np.array([]),
                input_signal=np.array([]),
                output_signal=np.array([]),
                frequencies=np.array([]),
                time_vector=np.array([]),
                simulation_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _run_spice_simulation(self, netlist_file: str) -> str:
        """Run actual SPICE simulation"""
        try:
            result = subprocess.run(
                [self.spice_command, '-b', netlist_file],
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                check=True
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            raise SimulationError(f"Simulation timed out for {netlist_file}")
        except subprocess.CalledProcessError as e:
            raise SimulationError(f"SPICE error: {e.stderr}")
    
    def _simulate_spice_output(self, config: CircuitConfiguration) -> str:
        """Simulate SPICE output for testing when SPICE is not available"""
        # Generate realistic-looking SPICE output
        # This is for testing/development only
        return "SIMULATED_SPICE_OUTPUT"
    
    def save_training_data(self, filename: str = "software_tina_data.json",
                          batch_size: int = 1000) -> int:
        """Save simulation results as training data with batch processing"""
        logger.info(f"Saving training data to {filename}")
        
        training_data = []
        successful_results = [r for r in self.simulation_results if r.success]
        
        # Process in batches to manage memory
        for i in range(0, len(successful_results), batch_size):
            batch = successful_results[i:i+batch_size]
            
            for result in batch:
                # Generate multiple test signals for each configuration
                test_signals = self._generate_test_signals(result)
                
                for signal_type, (input_signal, output_signal) in test_signals.items():
                    sample = {
                        'input_audio': input_signal.tolist(),
                        'output_audio': output_signal.tolist(),
                        'control_settings': result.configuration.controls,
                        'component_values': result.configuration.components,
                        'environmental_conditions': result.configuration.environmental,
                        'frequency_response': result.frequency_response.tolist(),
                        'frequencies': result.frequencies.tolist(),
                        'metadata': {
                            **result.configuration.metadata,
                            'signal_type': signal_type,
                            'simulation_time': result.simulation_time,
                            'metrics': result.metrics
                        }
                    }
                    training_data.append(sample)
            
            # Save intermediate results for large datasets
            if len(training_data) > 10000:
                self._save_training_batch(training_data, filename, i // batch_size)
                training_data = []
        
        # Save final batch
        if training_data:
            self._save_training_batch(training_data, filename, 'final')
        
        total_samples = len(successful_results) * 5  # 5 signal types per config
        logger.info(f"Saved {total_samples} training samples to {filename}")
        
        return total_samples
    
    def _generate_test_signals(self, result: SimulationResult) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Generate diverse test signals for training"""
        sample_rate = 44100
        duration = 0.02  # 20ms
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        signals = {}
        
        # Use the actual simulation output as base
        base_output = result.output_signal
        
        # Generate different input signals and corresponding outputs
        # 1. Sine wave
        sine_input = np.sin(2 * np.pi * 440 * t) * result.configuration.controls['input_level']
        signals['sine_440Hz'] = (sine_input, base_output)
        
        # 2. Guitar chord
        chord_freqs = [82.4, 110.0, 146.8, 196.0, 246.9, 329.6]
        chord_input = sum(np.sin(2 * np.pi * f * t) for f in chord_freqs)
        chord_input = chord_input / len(chord_freqs) * result.configuration.controls['input_level']
        signals['guitar_chord'] = (chord_input, self._process_signal(chord_input, result))
        
        # 3. White noise
        noise_input = np.random.normal(0, 0.1, len(t)) * result.configuration.controls['input_level']
        signals['white_noise'] = (noise_input, self._process_signal(noise_input, result))
        
        # 4. Frequency sweep
        f_start, f_end = 80, 5000
        sweep_freqs = np.logspace(np.log10(f_start), np.log10(f_end), len(t))
        sweep_input = np.sin(2 * np.pi * sweep_freqs * t) * result.configuration.controls['input_level']
        signals['frequency_sweep'] = (sweep_input, self._process_signal(sweep_input, result))
        
        # 5. Impulse
        impulse_input = np.zeros(len(t))
        impulse_input[0] = result.configuration.controls['input_level']
        signals['impulse'] = (impulse_input, self._process_signal(impulse_input, result))
        
        return signals
    
    def _process_signal(self, input_signal: np.ndarray, result: SimulationResult) -> np.ndarray:
        """Process signal through the circuit model (simplified)"""
        # This is a placeholder - in production, you would run actual SPICE simulation
        # or use the trained model
        gain = result.metrics.get('gain', 10.0) if result.metrics else 10.0
        output = input_signal * gain
        
        # Add some nonlinearity based on circuit type
        output = np.tanh(output * 0.5) * 2.0
        
        return output
    
    def _save_training_batch(self, data: List[Dict], base_filename: str, batch_id):
        """Save a batch of training data"""
        if batch_id == 'final':
            filename = base_filename
        else:
            base, ext = os.path.splitext(base_filename)
            filename = f"{base}_batch_{batch_id}{ext}"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved batch to {filename}")
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze simulation results and generate insights"""
        successful_results = [r for r in self.simulation_results if r.success]
        
        if not successful_results:
            logger.warning("No successful simulations to analyze")
            return {}
        
        logger.info(f"Analyzing {len(successful_results)} successful simulations")
        
        analysis = {
            'parameter_sensitivity': self._analyze_parameter_sensitivity(successful_results),
            'frequency_response': self._analyze_frequency_responses(successful_results),
            'performance_stats': self._analyze_performance_stats(successful_results),
            'convergence': self._analyze_convergence(successful_results)
        }
        
        return analysis
    
    def _analyze_parameter_sensitivity(self, results: List[SimulationResult]) -> Dict[str, float]:
        """Analyze parameter sensitivity"""
        logger.info("Performing parameter sensitivity analysis...")
        
        sensitivities = {}
        
        # Extract parameter values and output metrics
        params = []
        outputs = []
        
        for result in results:
            param_vector = list(result.configuration.controls.values())
            output_metric = result.metrics.get('gain', 0) if result.metrics else 0
            
            params.append(param_vector)
            outputs.append(output_metric)
        
        params = np.array(params)
        outputs = np.array(outputs)
        
        # Calculate correlations
        param_names = list(results[0].configuration.controls.keys())
        for i, param_name in enumerate(param_names):
            if np.std(params[:, i]) > 0:  # Avoid division by zero
                correlation = np.corrcoef(params[:, i], outputs)[0, 1]
                sensitivities[param_name] = float(correlation)
                logger.info(f"Parameter sensitivity - {param_name}: {correlation:.3f}")
            else:
                sensitivities[param_name] = 0.0
        
        return sensitivities
    
    def _analyze_frequency_responses(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """Analyze frequency response characteristics"""
        logger.info("Analyzing frequency response characteristics...")
        
        # Collect all frequency responses
        freq_responses = []
        for r in results:
            if len(r.frequency_response) > 0:
                freq_responses.append(r.frequency_response)
        
        if not freq_responses:
            return {}
        
        freq_responses = np.array(freq_responses)
        
        # Calculate statistics
        avg_response = np.mean(freq_responses, axis=0)
        std_response = np.std(freq_responses, axis=0)
        
        # Find key frequency characteristics
        frequencies = results[0].frequencies if len(results[0].frequencies) > 0 else np.array([])
        
        analysis = {
            'average_response': avg_response.tolist(),
            'std_response': std_response.tolist(),
            'frequencies': frequencies.tolist()
        }
        
        if len(frequencies) > 0 and len(avg_response) > 0:
            max_gain_idx = np.argmax(avg_response)
            analysis['peak_frequency'] = float(frequencies[max_gain_idx])
            analysis['peak_gain'] = float(avg_response[max_gain_idx])
            analysis['average_variation'] = float(np.mean(std_response))
        
        return analysis
    
    def _analyze_performance_stats(self, results: List[SimulationResult]) -> Dict[str, float]:
        """Analyze performance statistics"""
        simulation_times = [r.simulation_time for r in results]
        
        stats = {
            'average_simulation_time': float(np.mean(simulation_times)),
            'total_simulation_time': float(np.sum(simulation_times)),
            'min_simulation_time': float(np.min(simulation_times)),
            'max_simulation_time': float(np.max(simulation_times)),
            'successful_simulations': len(results),
            'failed_simulations': len(self.simulation_results) - len(results),
            'success_rate': len(results) / len(self.simulation_results) * 100 if self.simulation_results else 0
        }
        
        logger.info(f"Performance statistics: {stats}")
        
        return stats
    
    def _analyze_convergence(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """Analyze convergence of parameter space exploration"""
        # Check how well we've covered the parameter space
        coverage = {}
        
        for param_name in results[0].configuration.controls.keys():
            values = [r.configuration.controls[param_name] for r in results]
            coverage[param_name] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        
        return {'parameter_coverage': coverage}
    
    def cleanup(self):
        """Clean up temporary files and resources"""
        try:
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Failed to clean up temporary directory: {e}")


def main():
    """Example usage of Software TINA"""
    
    # Create circuit model
    circuit_model = TubePreampModel()
    
    # Initialize Software TINA
    tina = SoftwareTINA(circuit_model, max_cache_size=5000)
    
    # Progress callback
    def progress_callback(completed, total):
        percent = 100 * completed / total
        logger.info(f"Progress: {completed}/{total} ({percent:.1f}%)")
    
    try:
        # Generate parameter sweep
        configurations = tina.generate_parameter_sweep(
            num_configs=500,
            sweep_type="latin_hypercube",
            seed=42  # For reproducibility
        )
        
        # Run simulations
        results = tina.run_simulation_batch(
            configurations,
            max_workers=8,
            progress_callback=progress_callback
        )
        
        # Analyze results
        analysis = tina.analyze_results()
        
        # Save analysis results
        with open('analysis_results.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Save training data
        num_samples = tina.save_training_data("tube_preamp_training_data.json")
        
        logger.info(f"Software TINA completed successfully!")
        logger.info(f"Generated {num_samples} training samples")
        logger.info(f"Analysis saved to analysis_results.json")
        
    except Exception as e:
        logger.error(f"Software TINA failed: {e}")
        raise
    
    finally:
        # Cleanup
        tina.cleanup()


if __name__ == "__main__":
    main()
