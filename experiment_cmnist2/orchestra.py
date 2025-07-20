import subprocess
import os

log_files_directory = "output_logs"

# List of program filenames to execute
programs = [
    "program1.py",
    "program2.py",
    "program3.py",
    "program4.py",
    "program5.py",
    "program6.py",
    "program7.py",
    "program8.py",
]

# Ensure the output logs directory exists
os.makedirs(log_files_directory, exist_ok=True)

for prog in programs:
    # Derive log filename
    base_name = os.path.splitext(prog)[0]
    log_filename = os.path.join(log_files_directory, f"log_{base_name}.txt")

    with open(log_filename, 'w') as log_file:
        try:
            # Run the program and capture output
            process = subprocess.Popen(
                ['python', prog],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            # Write output line by line to log file
            for line in process.stdout:
                print(line, end='')  # Optional: print to console
                log_file.write(line)

            process.wait()
        except Exception as e:
            error_msg = f"Error running {prog}: {e}\n"
            print(error_msg)
            log_file.write(error_msg)