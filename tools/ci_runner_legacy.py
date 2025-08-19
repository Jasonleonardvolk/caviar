# CI Runner Wrapper
print("hi")

import subprocess
import sys
import os
import json
from datetime import datetime
import argparse

class CIRunner:
    def __init__(self):
        self.results = {
            'start_time': None,
            'end_time': None,
            'steps': [],
            'overall_status': 'success'
        }
    
    def run_command(self, command, step_name, cwd=None):
        """Run a command and capture output"""
        print(f"\n{'='*60}")
        print(f"Running: {step_name}")
        print(f"Command: {command}")
        print(f"{'='*60}")
        
        step_result = {
            'name': step_name,
            'command': command,
            'start_time': datetime.now().isoformat(),
            'status': 'running'
        }
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=cwd
            )
            
            step_result['stdout'] = result.stdout
            step_result['stderr'] = result.stderr
            step_result['return_code'] = result.returncode
            step_result['status'] = 'success' if result.returncode == 0 else 'failed'
            
            if result.returncode != 0:
                self.results['overall_status'] = 'failed'
                print(f"❌ {step_name} failed with return code: {result.returncode}")
                if result.stderr:
                    print(f"Error output:\n{result.stderr}")
            else:
                print(f"✅ {step_name} completed successfully")
            
            if result.stdout:
                print(f"Output:\n{result.stdout}")
            
        except Exception as e:
            step_result['status'] = 'error'
            step_result['error'] = str(e)
            self.results['overall_status'] = 'failed'
            print(f"❌ Error running {step_name}: {e}")
        
        step_result['end_time'] = datetime.now().isoformat()
        self.results['steps'].append(step_result)
        
        return step_result['status'] == 'success'
    
    def run_pipeline(self, config_file):
        """Run CI pipeline from configuration file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")
            return False
        
        self.results['start_time'] = datetime.now().isoformat()
        
        print(f"\n🚀 Starting CI Pipeline: {config.get('name', 'Unnamed')}")
        print(f"Description: {config.get('description', 'No description')}")
        
        # Run each step in the pipeline
        for step in config.get('steps', []):
            if not self.run_command(
                step['command'],
                step['name'],
                cwd=step.get('cwd')
            ):
                if not step.get('continue_on_error', False):
                    print(f"\n❌ Pipeline failed at step: {step['name']}")
                    break
        
        self.results['end_time'] = datetime.now().isoformat()
        
        # Generate report
        self.generate_report()
        
        return self.results['overall_status'] == 'success'
    
    def generate_report(self):
        """Generate CI run report"""
        report_file = f"ci_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"CI Pipeline Results:")
        print(f"Overall Status: {self.results['overall_status'].upper()}")
        print(f"Report saved to: {report_file}")
        print(f"{'='*60}")

# Example configuration file format:
example_config = {
    "name": "Python Project CI",
    "description": "Run tests and linting for Python project",
    "steps": [
        {
            "name": "Install Dependencies",
            "command": "pip install -r requirements.txt"
        },
        {
            "name": "Run Linter",
            "command": "pylint src/",
            "continue_on_error": True
        },
        {
            "name": "Run Tests",
            "command": "pytest tests/"
        },
        {
            "name": "Generate Coverage Report",
            "command": "coverage run -m pytest && coverage report"
        }
    ]
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CI Runner')
    parser.add_argument('config', help='CI configuration file (JSON)')
    parser.add_argument('--example', action='store_true', help='Generate example config')
    
    args = parser.parse_args()
    
    if args.example:
        with open('ci_config_example.json', 'w') as f:
            json.dump(example_config, f, indent=2)
        print("Example configuration saved to: ci_config_example.json")
    else:
        runner = CIRunner()
        success = runner.run_pipeline(args.config)
        sys.exit(0 if success else 1)
