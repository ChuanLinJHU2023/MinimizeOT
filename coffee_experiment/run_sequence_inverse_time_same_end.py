import subprocess

# Loop over j from 1 to 4
for i in range(1, 5):
    # Construct the command
    cmd = ['python', 'program.py', f'i={i}', f'j=0']
    print(f"Running: {' '.join(cmd)}")
    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Print the output

    print(result.stdout)
    # Optional: handle errors
    if result.returncode != 0:
        print(f"Error running command for i={i}")
        print(result.stderr)