import React, { useState, useEffect, useCallback } from 'react';
import { Play, Pause, Settings, Database, Cpu, Zap, BarChart3, FileText, Download, Check, AlertCircle } from 'lucide-react';

interface SimulationConfig {
  sweepType: 'random' | 'grid' | 'latin_hypercube' | 'sobol';
  totalSimulations: number;
  parallelWorkers: number;
}

interface PerformanceStats {
  avgSimTime: number;
  successRate: number;
  totalTime: number;
  samplesGenerated: number;
}

interface SimulationResult {
  name: string;
  controls: {
    gain: number;
    bass: number;
    mid: number;
    treble: number;
    master: number;
  };
  rms_output: number;
  thd: number;
  peak_freq: number;
  frequency_response?: number[];
}

interface ComponentSpec {
  name: string;
  value: string;
  tolerance: string;
  tempCoeff: string;
}

const SoftwareTINAInterface: React.FC = () => {
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);
  const [simulationCount, setSimulationCount] = useState<number>(0);
  const [config, setConfig] = useState<SimulationConfig>({
    sweepType: 'latin_hypercube',
    totalSimulations: 1000,
    parallelWorkers: 8
  });
  const [currentPhase, setCurrentPhase] = useState<string>('ready');
  const [results, setResults] = useState<SimulationResult[]>([]);
  const [performanceStats, setPerformanceStats] = useState<PerformanceStats>({
    avgSimTime: 0,
    successRate: 0,
    totalTime: 0,
    samplesGenerated: 0
  });
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);

  // Simulate Software TINA operation
  useEffect(() => {
    if (isRunning) {
      const interval = setInterval(() => {
        setProgress(prev => {
          const increment = Math.random() * 3;
          const newProgress = Math.min(prev + increment, 100);
          const newSimCount = Math.floor(newProgress * config.totalSimulations / 100);
          setSimulationCount(newSimCount);
          
          // Update phase based on progress
          if (newProgress < 20) {
            setCurrentPhase('generating_configs');
          } else if (newProgress < 90) {
            setCurrentPhase('running_simulations');
          } else if (newProgress < 100) {
            setCurrentPhase('analyzing_results');
          } else {
            setCurrentPhase('complete');
            setIsRunning(false);
            
            // Generate mock results
            const mockResults = generateMockResults();
            setResults(mockResults);
            setPerformanceStats({
              avgSimTime: 0.124,
              successRate: 98.7,
              totalTime: (config.totalSimulations * 0.124) / config.parallelWorkers,
              samplesGenerated: Math.floor(config.totalSimulations * 0.987)
            });
          }
          
          return newProgress;
        });
      }, 100);

      return () => clearInterval(interval);
    }
  }, [isRunning, config.totalSimulations, config.parallelWorkers]);

  const generateMockResults = useCallback((): SimulationResult[] => {
    const frequencies = Array.from({length: 100}, (_, i) => Math.pow(10, i * 5 / 99));
    const mockConfigs: SimulationResult[] = [
      {
        name: "Clean Low Gain",
        controls: { gain: 2.1, bass: 5.0, mid: 5.0, treble: 6.2, master: 4.5 },
        rms_output: 0.234,
        thd: 0.8,
        peak_freq: 220,
        frequency_response: frequencies.map(f => 
          20 * Math.log10(Math.max(0.1, 1 + 0.5 * Math.sin(Math.log10(f) * 2) * Math.random()))
        )
      },
      {
        name: "Heavy Distortion",
        controls: { gain: 9.2, bass: 7.5, mid: 6.0, treble: 8.1, master: 7.2 },
        rms_output: 0.867,
        thd: 12.4,
        peak_freq: 1200,
        frequency_response: frequencies.map(f => 
          20 * Math.log10(Math.max(0.1, 1.5 + 0.8 * Math.sin(Math.log10(f) * 3) * Math.random()))
        )
      },
      {
        name: "Mid Crunch",
        controls: { gain: 5.5, bass: 4.2, mid: 7.8, treble: 5.9, master: 6.1 },
        rms_output: 0.445,
        thd: 4.2,
        peak_freq: 800,
        frequency_response: frequencies.map(f => 
          20 * Math.log10(Math.max(0.1, 1.2 + 0.6 * Math.sin(Math.log10(f) * 2.5) * Math.random()))
        )
      }
    ];
    
    return mockConfigs;
  }, []);

  const startSimulation = useCallback(() => {
    setIsRunning(true);
    setProgress(0);
    setSimulationCount(0);
    setCurrentPhase('starting');
    setResults([]);
  }, []);

  const stopSimulation = useCallback(() => {
    setIsRunning(false);
    setCurrentPhase('stopped');
  }, []);

  const updateConfig = useCallback(<K extends keyof SimulationConfig>(
    key: K,
    value: SimulationConfig[K]
  ) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  }, []);

  const exportData = useCallback((format: 'json' | 'hdf5' | 'pytorch' | 'tensorflow') => {
    console.log(`Exporting data in ${format} format...`);
    // In production, this would trigger actual export
    alert(`Data would be exported in ${format.toUpperCase()} format`);
  }, []);

  const phaseMessages: Record<string, string> = {
    ready: "Ready to start Software TINA data collection",
    starting: "Initializing Software TINA...",
    generating_configs: "Generating parameter configurations...",
    running_simulations: "Running SPICE simulations in parallel...",
    analyzing_results: "Analyzing simulation results...",
    complete: "Software TINA data collection complete!",
    stopped: "Simulation stopped by user"
  };

  const advantages = [
    "🚀 100-250x faster than hardware TINA",
    "🔧 No mechanical wear or calibration needed",
    "⚡ No high voltage safety concerns",
    "🎯 Perfect parameter repeatability",
    "📊 Infinite parameter exploration",
    "🌡️ Component aging & temperature modeling",
    "🔄 Parallel processing capability",
    "💾 Zero measurement noise"
  ];

  const componentSpecs: ComponentSpec[] = [
    { name: "Plate Resistor", value: "100kΩ ±5%", tolerance: "±5%", tempCoeff: "-200ppm/°C" },
    { name: "Cathode Cap", value: "22µF ±20%", tolerance: "±20%", tempCoeff: "500ppm/°C" },
    { name: "12AX7 Tube", value: "µ=100 ±10%", tolerance: "±10%", tempCoeff: "Aging: 5%/20yr" },
    { name: "Coupling Cap", value: "0.022µF ±10%", tolerance: "±10%", tempCoeff: "100ppm/°C" },
    { name: "Supply", value: "320V ±5%", tolerance: "±5%", tempCoeff: "Ripple: 1%" }
  ];

  return (
    <div className="max-w-7xl mx-auto p-6 bg-gray-900 text-green-400 font-mono">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2 text-center">
          ⚡ SOFTWARE TINA
        </h1>
        <p className="text-center text-green-300">
          Virtual Telemetric Inductive Nodal Actuator - SPICE-Based Data Collection
        </p>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Control Panel */}
        <div className="bg-gray-800 rounded-lg p-6 border border-green-700">
          <h2 className="text-xl font-bold mb-4 flex items-center">
            <Settings className="mr-2" size={20} />
            Control Panel
          </h2>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Sweep Type</label>
              <select 
                value={config.sweepType} 
                onChange={(e) => updateConfig('sweepType', e.target.value as SimulationConfig['sweepType'])}
                className="w-full bg-gray-700 border border-green-600 rounded px-3 py-2 text-green-400"
                disabled={isRunning}
              >
                <option value="random">Random Sampling</option>
                <option value="grid">Grid Sweep</option>
                <option value="latin_hypercube">Latin Hypercube</option>
                <option value="sobol">Sobol Sequence</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">
                Simulations: {config.totalSimulations.toLocaleString()}
              </label>
              <input
                type="range"
                min="100"
                max="10000"
                value={config.totalSimulations}
                onChange={(e) => updateConfig('totalSimulations', Number(e.target.value))}
                className="w-full"
                disabled={isRunning}
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">
                Parallel Workers: {config.parallelWorkers}
              </label>
              <input
                type="range"
                min="1"
                max="16"
                value={config.parallelWorkers}
                onChange={(e) => updateConfig('parallelWorkers', Number(e.target.value))}
                className="w-full"
                disabled={isRunning}
              />
            </div>
            
            <div className="flex space-x-2">
              <button
                onClick={startSimulation}
                disabled={isRunning}
                className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white py-2 px-4 rounded flex items-center justify-center transition-colors"
              >
                <Play className="mr-2" size={16} />
                Start
              </button>
              
              <button
                onClick={stopSimulation}
                disabled={!isRunning}
                className="flex-1 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 text-white py-2 px-4 rounded flex items-center justify-center transition-colors"
              >
                <Pause className="mr-2" size={16} />
                Stop
              </button>
            </div>

            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="w-full bg-gray-700 hover:bg-gray-600 text-green-400 py-2 px-4 rounded flex items-center justify-center transition-colors"
            >
              {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
            </button>
          </div>
        </div>

        {/* Progress Monitor */}
        <div className="bg-gray-800 rounded-lg p-6 border border-green-700">
          <h2 className="text-xl font-bold mb-4 flex items-center">
            <BarChart3 className="mr-2" size={20} />
            Progress Monitor
          </h2>
          
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Overall Progress</span>
                <span>{progress.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3">
                <div 
                  className="bg-green-500 h-3 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
            
            <div className="text-sm space-y-1">
              <div className="flex justify-between">
                <span>Simulations:</span>
                <span>{simulationCount}/{config.totalSimulations}</span>
              </div>
              <div className="flex justify-between">
                <span>Workers:</span>
                <span>{config.parallelWorkers} parallel</span>
              </div>
              <div className="flex justify-between">
                <span>Est. Time:</span>
                <span>
                  {((config.totalSimulations - simulationCount) * 0.124 / config.parallelWorkers).toFixed(1)}s
                </span>
              </div>
              <div className="flex justify-between">
                <span>Speed vs Hardware:</span>
                <span className="text-yellow-400">
                  {(250 / config.parallelWorkers).toFixed(0)}x faster
                </span>
              </div>
            </div>
            
            <div className="bg-gray-900 p-3 rounded">
              <div className="text-sm text-green-300 flex items-center">
                {currentPhase === 'running_simulations' && (
                  <Cpu className="mr-2 animate-pulse" size={16} />
                )}
                {phaseMessages[currentPhase]}
              </div>
            </div>
          </div>
        </div>

        {/* Advantages */}
        <div className="bg-gray-800 rounded-lg p-6 border border-green-700">
          <h2 className="text-xl font-bold mb-4 flex items-center">
            <Zap className="mr-2" size={20} />
            Software TINA Advantages
          </h2>
          
          <div className="space-y-2">
            {advantages.map((advantage, index) => (
              <div key={index} className="flex items-center text-sm">
                <Check className="mr-2 text-green-500 flex-shrink-0" size={14} />
                <span>{advantage}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Advanced Settings (Collapsible) */}
      {showAdvanced && (
        <div className="mb-8 bg-gray-800 rounded-lg p-6 border border-green-700">
          <h2 className="text-xl font-bold mb-4 flex items-center">
            <Settings className="mr-2" size={20} />
            Advanced Configuration
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">Temperature Range</label>
              <div className="flex space-x-2">
                <input type="number" className="flex-1 bg-gray-700 rounded px-2 py-1" defaultValue="15" />
                <span className="text-green-400">to</span>
                <input type="number" className="flex-1 bg-gray-700 rounded px-2 py-1" defaultValue="45" />
                <span className="text-green-400">°C</span>
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">Aging Range</label>
              <div className="flex space-x-2">
                <input type="number" className="flex-1 bg-gray-700 rounded px-2 py-1" defaultValue="0" />
                <span className="text-green-400">to</span>
                <input type="number" className="flex-1 bg-gray-700 rounded px-2 py-1" defaultValue="20" />
                <span className="text-green-400">years</span>
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">Supply Variation</label>
              <div className="flex space-x-2">
                <input type="number" className="flex-1 bg-gray-700 rounded px-2 py-1" defaultValue="-20" />
                <span className="text-green-400">to</span>
                <input type="number" className="flex-1 bg-gray-700 rounded px-2 py-1" defaultValue="+20" />
                <span className="text-green-400">%</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Results Section */}
      {results.length > 0 && (
        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-4 flex items-center">
            <Database className="mr-2" size={24} />
            Simulation Results
          </h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Performance Statistics */}
            <div className="bg-gray-800 rounded-lg p-6 border border-green-700">
              <h3 className="text-lg font-bold mb-3">Performance Statistics</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>Average Simulation Time:</span>
                  <span className="text-green-300">{performanceStats.avgSimTime.toFixed(3)}s</span>
                </div>
                <div className="flex justify-between">
                  <span>Success Rate:</span>
                  <span className="text-green-300">{performanceStats.successRate.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span>Total Collection Time:</span>
                  <span className="text-green-300">{performanceStats.totalTime.toFixed(1)}s</span>
                </div>
                <div className="flex justify-between">
                  <span>Training Samples Generated:</span>
                  <span className="text-green-300">{performanceStats.samplesGenerated.toLocaleString()}</span>
                </div>
              </div>
              
              <div className="mt-4 p-3 bg-green-900 bg-opacity-20 rounded">
                <div className="text-sm text-green-300">
                  🚀 Hardware TINA would take ~{(performanceStats.totalTime * 250).toFixed(0)} seconds
                  <br />
                  ⚡ Software TINA: {performanceStats.totalTime.toFixed(1)}s (250x faster!)
                </div>
              </div>
            </div>

            {/* Sample Configurations */}
            <div className="bg-gray-800 rounded-lg p-6 border border-green-700">
              <h3 className="text-lg font-bold mb-3">Sample Configurations</h3>
              <div className="space-y-3">
                {results.slice(0, 3).map((result, index) => (
                  <div key={index} className="bg-gray-900 p-3 rounded text-sm">
                    <div className="font-bold text-green-300">{result.name}</div>
                    <div className="mt-1 space-y-1 text-xs">
                      <div>
                        Gain: {result.controls.gain.toFixed(1)}, 
                        Bass: {result.controls.bass.toFixed(1)}, 
                        Mid: {result.controls.mid.toFixed(1)},
                        Treble: {result.controls.treble.toFixed(1)},
                        Master: {result.controls.master.toFixed(1)}
                      </div>
                      <div>
                        RMS: {result.rms_output.toFixed(3)}, 
                        THD: {result.thd.toFixed(1)}%, 
                        Peak: {result.peak_freq}Hz
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Circuit Model Configuration */}
      <div className="mb-8">
        <h2 className="text-2xl font-bold mb-4 flex items-center">
          <Cpu className="mr-2" size={24} />
          Circuit Model Configuration
        </h2>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-gray-800 rounded-lg p-6 border border-green-700">
            <h3 className="text-lg font-bold mb-3">Component Modeling</h3>
            <div className="bg-gray-900 p-4 rounded font-mono text-sm overflow-x-auto">
              <div className="space-y-1">
                {componentSpecs.map((spec, index) => (
                  <div key={index}>
                    {spec.name}: {spec.value} (temp: {spec.tempCoeff})
                  </div>
                ))}
              </div>
            </div>
          </div>
          
          <div className="bg-gray-800 rounded-lg p-6 border border-green-700">
            <h3 className="text-lg font-bold mb-3">Parameter Ranges</h3>
            <div className="bg-gray-900 p-4 rounded font-mono text-sm">
              <div className="space-y-1">
                <div>Input Level: 0.001V - 1.0V</div>
                <div>Bias Voltage: -2.0V - -0.5V</div>
                <div>Supply Voltage: 280V - 360V</div>
                <div>Temperature: 15°C - 45°C</div>
                <div>Aging: 0 - 20 years</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Export Options */}
      <div className="bg-gray-800 rounded-lg p-6 border border-green-700">
        <h2 className="text-xl font-bold mb-4 flex items-center">
          <Download className="mr-2" size={20} />
          Export Training Data
        </h2>
        
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
          <button 
            onClick={() => exportData('json')}
            className="bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded flex items-center justify-center transition-colors"
          >
            <FileText className="mr-2" size={16} />
            JSON Format
          </button>
          
          <button 
            onClick={() => exportData('hdf5')}
            className="bg-purple-600 hover:bg-purple-700 text-white py-2 px-4 rounded flex items-center justify-center transition-colors"
          >
            <Database className="mr-2" size={16} />
            HDF5 Format
          </button>
          
          <button 
            onClick={() => exportData('pytorch')}
            className="bg-orange-600 hover:bg-orange-700 text-white py-2 px-4 rounded flex items-center justify-center transition-colors"
          >
            <Cpu className="mr-2" size={16} />
            PyTorch Dataset
          </button>
          
          <button 
            onClick={() => exportData('tensorflow')}
            className="bg-green-600 hover:bg-green-700 text-white py-2 px-4 rounded flex items-center justify-center transition-colors"
          >
            <Zap className="mr-2" size={16} />
            TensorFlow Data
          </button>
        </div>
        
        <div className="mt-4 p-3 bg-gray-900 rounded">
          <div className="text-sm text-green-300 flex items-start">
            <AlertCircle className="mr-2 flex-shrink-0 mt-0.5" size={16} />
            <span>
              Training data includes: Input/output audio pairs, control parameters, 
              component values, environmental conditions, and frequency responses.
              Each configuration generates multiple test signals for comprehensive training.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SoftwareTINAInterface;