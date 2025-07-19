import subprocess

# Loop over j from 1 to 4
for j in range(1, 5):
    # Construct the command
    cmd = ['python', 'program1.py', f'i=0', f'j={j}']
    print(f"Running: {' '.join(cmd)}")
    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Print the output
    print(result.stdout)
    # Optional: handle errors
    if result.returncode != 0:
        print(f"Error running command for j={j}")
        print(result.stderr)