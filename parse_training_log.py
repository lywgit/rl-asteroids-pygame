import re
import matplotlib.pyplot as plt
import numpy as np
import argparse

def moving_average(data, window_size):
    """
    Calculate moving average of data with given window size.
    
    Args:
        data (list): Input data
        window_size (int): Size of the moving window
        
    Returns:
        np.array: Moving average values
    """
    if len(data) < window_size:
        return np.array(data)
    
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def print_training_progress(frames, eval_rewards, ma_window=10):
    """
    Print training progress data as a formatted table.
    
    Args:
        frames (list): Frame numbers
        eval_rewards (list): Evaluation rewards
        ma_window (int): Moving average window size
    """
    print(f"\nTraining Progress Data (Moving Average Window: {ma_window})")
    print("=" * 65)
    print(f"{'Frame':<10} {'Eval Reward':<12} {'MA Eval Reward':<15}")
    print("-" * 65)
    
    # Calculate moving average
    if len(eval_rewards) >= ma_window:
        ma_rewards = moving_average(eval_rewards, ma_window)
        
        # Print data row by row
        for i, (frame, eval_reward) in enumerate(zip(frames, eval_rewards)):
            if i < ma_window - 1:
                # Before we have enough data for moving average
                print(f"{frame:<10} {eval_reward:<12.1f} {'N/A':<15}")
            else:
                # With moving average data
                ma_index = i - (ma_window - 1)
                ma_reward = ma_rewards[ma_index]
                print(f"{frame:<10} {eval_reward:<12.1f} {ma_reward:<15.1f}")
    else:
        # Not enough data for moving average
        for frame, eval_reward in zip(frames, eval_rewards):
            print(f"{frame:<10} {eval_reward:<12.1f} {'N/A':<15}")
    
    print("-" * 65)

def parse_training_log(log_file_path):
    """
    Parse training log file to extract evaluation rewards and corresponding frames.
    
    Args:
        log_file_path (str): Path to the log file
        
    Returns:
        tuple: (frames, eval_rewards, best_rewards) lists
    """
    frames = []
    eval_rewards = []
    best_rewards = []
    
    with open(log_file_path, 'r') as file:
        current_frame = None
        
        for line in file:
            line = line.strip()
            
            # Extract frame number from episode lines
            frame_match = re.search(r'frame (\d+),', line)
            if frame_match:
                current_frame = int(frame_match.group(1))
            
            # Extract evaluation results
            eval_match = re.search(r'Evaluation results: ([\d.]+) best: (-?inf|[\d.-]+)', line)
            if eval_match and current_frame is not None:
                eval_reward = float(eval_match.group(1))
                best_reward_str = eval_match.group(2)
                
                # Skip -inf values, only store valid numeric best rewards
                if best_reward_str == '-inf':
                    best_reward = None  # Skip -inf values
                elif best_reward_str == 'inf':
                    best_reward = float('inf')
                else:
                    best_reward = float(best_reward_str)
                
                frames.append(current_frame)
                eval_rewards.append(eval_reward)
                if best_reward is not None:
                    best_rewards.append(best_reward)
                else:
                    # For -inf cases, just use the current eval reward or skip
                    if best_rewards:  # If we have previous best rewards, use the last one
                        best_rewards.append(best_rewards[-1])
                    else:
                        best_rewards.append(eval_reward)  # Use current eval as placeholder
    
    return frames, eval_rewards, best_rewards

def plot_training_progress(frames, eval_rewards, best_rewards, output_file='training_progress.png', ma_window=10):
    """
    Plot training progress showing evaluation rewards over frames.
    
    Args:
        frames (list): Frame numbers
        eval_rewards (list): Evaluation rewards
        best_rewards (list): Best rewards so far
        output_file (str): Output PNG file name
        ma_window (int): Moving average window size for smoothing
    """
    plt.figure(figsize=(12, 8))
    
    # Plot raw evaluation rewards (lighter/more transparent)
    plt.plot(frames, eval_rewards, 'b-', alpha=0.3, linewidth=1, label='Raw Evaluation Reward')
    
    # Calculate and plot moving average
    if len(eval_rewards) >= ma_window:
        ma_rewards = moving_average(eval_rewards, ma_window)
        # Adjust frames for moving average (center the window)
        ma_frames = frames[ma_window-1:]
        plt.plot(ma_frames, ma_rewards, 'b-', alpha=0.8, linewidth=2, 
                label=f'Moving Average (window={ma_window})')
    
    # Plot best rewards (now that we've handled -inf values in parsing)
    if best_rewards:
        plt.plot(frames, best_rewards, 'r-', linewidth=2, label='Best Reward')
    
    # Add scatter points for evaluation rewards (smaller and more transparent)
    plt.scatter(frames, eval_rewards, c='blue', alpha=0.4, s=15, zorder=5)
    
    plt.xlabel('Training Frames', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('Asteroids AI Training Progress - Evaluation Rewards', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add some statistics as text
    if eval_rewards:
        max_eval = max(eval_rewards)
        final_eval = eval_rewards[-1]
        avg_eval = np.mean(eval_rewards)
        
        stats_text = f'Max Eval: {max_eval:.0f}\nFinal Eval: {final_eval:.0f}\nAvg Eval: {avg_eval:.0f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    return plt.gcf()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Parse training log and create evaluation reward plot')
    parser.add_argument('log_file', nargs='?', default='log_20250918_1218.txt',
                        help='Path to the training log file (default: log_20250918_1218.txt)')
    parser.add_argument('--window-size', '-w', type=int, default=10,
                        help='Moving average window size (default: 10)')
    parser.add_argument('--output', '-o', default='training_progress.png',
                        help='Output PNG file name (default: training_progress.png)')
    
    args = parser.parse_args()
    
    try:
        frames, eval_rewards, best_rewards = parse_training_log(args.log_file)
        
        print(f"Parsed {len(frames)} evaluation points from: {args.log_file}")
        print(f"Frame range: {min(frames)} - {max(frames)}")
        print(f"Reward range: {min(eval_rewards):.0f} - {max(eval_rewards):.0f}")
        print(f"Using moving average window size: {args.window_size}")
        
        # Print the training progress data as a table
        print_training_progress(frames, eval_rewards, ma_window=args.window_size)
        
        # Create the plot with moving average smoothing
        plot_training_progress(frames, eval_rewards, best_rewards, 
                             output_file=args.output, ma_window=args.window_size)
        
        # Print some summary statistics
        print(f"\nTraining Statistics:")
        print(f"Total frames trained: {max(frames):,}")
        print(f"Number of evaluations: {len(eval_rewards)}")
        print(f"Best evaluation reward: {max(eval_rewards):.0f}")
        print(f"Final evaluation reward: {eval_rewards[-1]:.0f}")
        print(f"Average evaluation reward: {np.mean(eval_rewards):.1f}")
        
    except FileNotFoundError:
        print(f"Error: Could not find log file '{args.log_file}'")
        print("Make sure the log file path is correct.")
    except Exception as e:
        print(f"Error parsing log file: {e}")

if __name__ == "__main__":
    main()