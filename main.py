import sys
import subprocess
from pathlib import Path

def print_usage():
    print("Asteroids AI Demo Launcher")
    print("Usage: uv run main.py <demo_name>")
    print()
    print("Available demos:")
    print("  human      - Play the game with keyboard controls")
    print("  basic-ai   - Watch a simple mock AI play")
    print("  torch-ai   - Watch PyTorch AI models play")
    print("  test-obs   - Test observation space")
    print()
    print("Examples:")
    print("  uv run main.py human")
    print("  uv run main.py basic-ai")
    print("  uv run main.py torch-ai")

def main():
    if len(sys.argv) < 2:
        print_usage()
        return
    
    demo = sys.argv[1].lower()
    
    # Map demo names to module paths
    demo_map = {
        'human': 'demos.play_human',
        'basic-ai': 'demos.play_basic_ai',
        'torch-ai': 'demos.play_torch_ai',
        'test-obs': 'tests.test_observations',
    }
    
    if demo not in demo_map:
        print(f"Unknown demo: {demo}")
        print()
        print_usage()
        return
    
    module_path = demo_map[demo]
    print(f"Launching {demo} demo...")
    
    # Execute the demo using python -m
    try:
        subprocess.run([sys.executable, '-m', module_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running demo: {e}")
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")

if __name__ == "__main__":
    main()
