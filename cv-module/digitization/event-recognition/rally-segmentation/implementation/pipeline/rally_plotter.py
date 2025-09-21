import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json


class RallyPlotter:
    """Handles plotting and visualization of rally states and player distances"""
    
    def __init__(self, config):
        self.config = config
        plot_config = config.get("plotting", {})
        
        # Plot settings
        self.save_plots = plot_config.get("save_plots", True)
        self.show_plots = plot_config.get("show_plots", True)
        self.output_dir = plot_config.get("output_directory", "rally_plots")
        self.plot_format = plot_config.get("format", "png")
        self.dpi = plot_config.get("dpi", 300)
        
        # Create output directory
        if self.save_plots:
            Path(self.output_dir).mkdir(exist_ok=True)
        
        # Data storage
        self.frame_numbers = []
        self.distances = []
        self.rally_states = []
        self.intensities = []
        self.state_transitions = []
        
        # Color mapping for rally states
        self.state_colors = {
            "rally_end": "#FF4444",      # Red
            "rally_start": "#FFAA00",    # Orange
            "rally_active": "#44AA44"    # Green
        }
        
        # Plot styling
        plt.style.use('default')
        self.figsize = plot_config.get("figure_size", (12, 8))
    
    def add_data_point(self, frame_count, avg_distance, rally_state, combined_intensity):
        """Add a data point to the plotting data"""
        self.frame_numbers.append(frame_count)
        self.distances.append(avg_distance if avg_distance is not None else np.nan)
        self.rally_states.append(rally_state)
        self.intensities.append(combined_intensity)
    
    def add_state_transition(self, transition_data):
        """Add a state transition marker"""
        self.state_transitions.append(transition_data)
    
    def plot_distance_with_states(self, fps=30.0):
        """Plot average distance over time with rally state coloring"""
        if not self.frame_numbers:
            print("No data available for plotting")
            return None
        
        # Convert frames to time in seconds
        time_seconds = np.array(self.frame_numbers) / fps
        distances = np.array(self.distances)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        
        # Plot 1: Distance with state coloring
        self._plot_distance_colored_by_state(ax1, time_seconds, distances)
        
        # Plot 2: Rally state timeline
        self._plot_rally_state_timeline(ax2, time_seconds)
        
        # Add state transition markers
        self._add_transition_markers(ax1, ax2, fps)
        
        # Formatting
        ax1.set_ylabel('Average Distance (m)', fontsize=12)
        ax1.set_title('Player Distance and Rally States Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Rally State', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if self.save_plots:
            filename = f"rally_distance_analysis.{self.plot_format}"
            filepath = Path(self.output_dir) / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"💾 Plot saved: {filepath}")
        
        # Show plot
        if self.show_plots:
            plt.show()
        
        return fig
    
    def _plot_distance_colored_by_state(self, ax, time_seconds, distances):
        """Plot distance data colored by rally state"""
        # Group consecutive states for efficient plotting
        current_state = None
        start_idx = 0
        
        for i, state in enumerate(self.rally_states):
            if state != current_state:
                # Plot previous segment
                if current_state is not None and i > start_idx:
                    color = self.state_colors.get(current_state, '#666666')
                    ax.plot(time_seconds[start_idx:i], distances[start_idx:i], 
                           color=color, linewidth=2, label=current_state if current_state not in ax.get_legend_handles_labels()[1] else "")
                
                current_state = state
                start_idx = i
        
        # Plot final segment
        if current_state is not None and len(self.rally_states) > start_idx:
            color = self.state_colors.get(current_state, '#666666')
            ax.plot(time_seconds[start_idx:], distances[start_idx:], 
                   color=color, linewidth=2, label=current_state if current_state not in ax.get_legend_handles_labels()[1] else "")
    
    def _plot_rally_state_timeline(self, ax, time_seconds):
        """Plot rally state as a timeline"""
        # Create numerical representation of states
        state_values = {'rally_end': 0, 'rally_start': 1, 'rally_active': 2}
        numeric_states = [state_values.get(state, 0) for state in self.rally_states]
        
        # Create colored segments
        for i in range(len(time_seconds) - 1):
            state = self.rally_states[i]
            color = self.state_colors.get(state, '#666666')
            ax.fill_between([time_seconds[i], time_seconds[i+1]], 
                           [numeric_states[i], numeric_states[i]], 
                           [numeric_states[i] + 0.8, numeric_states[i] + 0.8],
                           color=color, alpha=0.7)
        
        # Customize state timeline
        ax.set_ylim(-0.5, 2.5)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Rally End', 'Rally Start', 'Rally Active'])
    
    def _add_transition_markers(self, ax1, ax2, fps):
        """Add vertical lines at state transitions"""
        for transition in self.state_transitions:
            frame = transition['frame']
            time = frame / fps
            
            # Add vertical line
            ax1.axvline(x=time, color='red', linestyle='--', alpha=0.6, linewidth=1)
            ax2.axvline(x=time, color='red', linestyle='--', alpha=0.6, linewidth=1)
            
            # Add annotation
            from_state = transition['from_state'].replace('rally_', '')
            to_state = transition['to_state'].replace('rally_', '')
            ax1.annotate(f'{from_state}→{to_state}', 
                        xy=(time, max(self.distances) * 0.9), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, rotation=90, alpha=0.7)
    
    def plot_intensity_vs_distance(self, fps=30.0):
        """Create scatter plot of intensity vs distance colored by rally state"""
        if not self.frame_numbers:
            print("No data available for plotting")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter plot for each state
        for state in ['rally_end', 'rally_start', 'rally_active']:
            state_mask = np.array(self.rally_states) == state
            if np.any(state_mask):
                distances_state = np.array(self.distances)[state_mask]
                intensities_state = np.array(self.intensities)[state_mask]
                
                # Remove NaN values
                valid_mask = ~np.isnan(distances_state) & ~np.isnan(intensities_state)
                if np.any(valid_mask):
                    ax.scatter(distances_state[valid_mask], intensities_state[valid_mask],
                             color=self.state_colors[state], label=state, alpha=0.6, s=20)
        
        ax.set_xlabel('Average Distance (m)', fontsize=12)
        ax.set_ylabel('Combined Intensity (m/frame)', fontsize=12)
        ax.set_title('Intensity vs Distance by Rally State', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save plot
        if self.save_plots:
            filename = f"intensity_vs_distance.{self.plot_format}"
            filepath = Path(self.output_dir) / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"💾 Plot saved: {filepath}")
        
        # Show plot
        if self.show_plots:
            plt.show()
        
        return fig
    
    def generate_summary_statistics(self):
        """Generate and print summary statistics"""
        if not self.frame_numbers:
            print("No data available for statistics")
            return None
        
        distances = np.array(self.distances)
        valid_distances = distances[~np.isnan(distances)]
        
        stats = {
            "total_frames": len(self.frame_numbers),
            "valid_distance_measurements": len(valid_distances),
            "avg_distance_overall": np.mean(valid_distances) if len(valid_distances) > 0 else 0,
            "std_distance_overall": np.std(valid_distances) if len(valid_distances) > 0 else 0,
            "min_distance": np.min(valid_distances) if len(valid_distances) > 0 else 0,
            "max_distance": np.max(valid_distances) if len(valid_distances) > 0 else 0,
            "state_statistics": {}
        }
        
        # Calculate statistics per state
        for state in ['rally_end', 'rally_start', 'rally_active']:
            state_mask = np.array(self.rally_states) == state
            state_distances = distances[state_mask]
            valid_state_distances = state_distances[~np.isnan(state_distances)]
            
            if len(valid_state_distances) > 0:
                stats["state_statistics"][state] = {
                    "count": len(valid_state_distances),
                    "avg_distance": np.mean(valid_state_distances),
                    "std_distance": np.std(valid_state_distances),
                    "min_distance": np.min(valid_state_distances),
                    "max_distance": np.max(valid_state_distances)
                }
        
        # Print summary
        print(f"\n{'='*50}")
        print("RALLY DISTANCE ANALYSIS SUMMARY")
        print(f"{'='*50}")
        print(f"Total frames analyzed: {stats['total_frames']}")
        print(f"Valid distance measurements: {stats['valid_distance_measurements']}")
        print(f"Overall average distance: {stats['avg_distance_overall']:.2f} ± {stats['std_distance_overall']:.2f} m")
        print(f"Distance range: {stats['min_distance']:.2f} - {stats['max_distance']:.2f} m")
        
        print(f"\nDistance by Rally State:")
        for state, state_stats in stats["state_statistics"].items():
            print(f"  {state:12}: {state_stats['avg_distance']:.2f} ± {state_stats['std_distance']:.2f} m "
                  f"(n={state_stats['count']}, range: {state_stats['min_distance']:.2f}-{state_stats['max_distance']:.2f})")
        
        # Save statistics to JSON
        if self.save_plots:
            stats_file = Path(self.output_dir) / "rally_statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"📊 Statistics saved: {stats_file}")
        
        return stats
    
    def create_all_plots(self, fps=30.0):
        """Generate all available plots and statistics"""
        print(f"\n📈 Generating rally analysis plots...")
        
        # Main distance plot
        self.plot_distance_with_states(fps)
        
        # Intensity vs distance scatter plot
        self.plot_intensity_vs_distance(fps)
        
        # Generate statistics
        self.generate_summary_statistics()
        
        print(f"✅ Rally analysis complete!")