import csv
import os
from datetime import datetime
import json

class SimpleExperimentTracker:
    """Simple experiment tracker for ML experiments"""
    
    def __init__(self, log_file="experiments_log.csv"):
        self.log_file = log_file
        self.current_experiment = {}
        
        # create the header if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'experiment_name', 'test_accuracy', 'test_precision', 
                    'test_recall', 'test_f1', 'epochs', 'notes'
                ])
    
    def start_experiment(self, name):
        """start a new experiment"""
        self.current_experiment = {
            'name': name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'start_time': datetime.now()
        }
        print(f" Experiment started: {name}")
    
    def log_results(self, accuracy, precision, recall, f1_score, epochs, notes=""):
        """Save the results of the experiment"""
        self.current_experiment.update({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'epochs': epochs,
            'notes': notes
        })
        
        # write to csv
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_experiment['timestamp'],
                self.current_experiment['name'],
                accuracy,
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{f1_score:.4f}",
                epochs,
                notes
            ])
        
        print(f" Results saved to {self.log_file}")
        
        # show the comparison with the previous experiments
        self.show_comparison()
    
    def show_comparison(self):
        """show the comparison with the previous experiments"""
        if not os.path.exists(self.log_file):
            return
            
        experiments = []
        with open(self.log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                experiments.append(row)
        
        if len(experiments) <= 1:
            print("First experiment saved!")
            return
        
        # compare with the previous experiments
        current = experiments[-1]  # last
        previous = experiments[-2]  # second last
        
        print(f"\n === COMPARISON WITH THE PREVIOUS EXPERIMENT ===")
        print(f" Previous experiment: {previous['experiment_name']}")
        print(f" Current experiment:   {current['experiment_name']}")
        print(f"")
        
        # compare
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        
        for metric in metrics:
            if metric in current and metric in previous:
                curr_val = float(current[metric])
                prev_val = float(previous[metric])
                diff = curr_val - prev_val
                
                if diff > 0:
                    emoji = "⬆️"
                    sign = "+"
                elif diff < 0:
                    emoji = "⬇️"  
                    sign = ""
                else:
                    emoji = "➡️ ="
                    sign = ""
                
                print(f"{emoji} {metric.replace('test_', '').upper()}: {curr_val:.4f} ({sign}{diff:+.4f})")
    
    def view_all_experiments(self):
        """show all experiments"""
        if not os.path.exists(self.log_file):
            print(" No experiments saved yet.")
            return
        
        print(f"\n === ALL EXPERIMENTS ===")
        with open(self.log_file, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, 1):
                print(f"{i}. {row['experiment_name'].rsplit('_', 2)[0]} ({row['timestamp']})")
                print(f"   Accuracy: {row['test_accuracy']} | F1: {row['test_f1']}")
                if row['notes']:
                    print(f"   Notă: {row['notes']}")
                print()


def create_tracker(log_file="experiments_log.csv"):
    """create a simple tracker with custom file location"""
    dir_path = os.path.dirname(log_file)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f" Created directory: {dir_path}")
    
    return SimpleExperimentTracker(log_file=log_file) 