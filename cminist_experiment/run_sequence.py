import subprocess

scaling_parameter_c = 128
img_red_zero_path = '../image_digits/red_zero.png'
img_blue_zero_path = '../image_digits/blue_zero.png'
img_red_one_path = '../image_digits/red_one.png'

# First subprocess: red_zero vs blue_zero
result1 = subprocess.run(
    ['python', 'program.py', '--img1', img_red_zero_path, '--img2', img_blue_zero_path, '--scaling_param', str(scaling_parameter_c)],
    capture_output=True,
    text=True
)
dist_zero_to_blue_zero = float(result1.stdout.strip())
print(f"DIST RED ZERO 2 BLUE ZERO: {dist_zero_to_blue_zero}")

# Second subprocess: red_zero vs red_one
result2 = subprocess.run(
    ['python', 'program.py', '--img1', img_red_zero_path, '--img2', img_red_one_path, '--scaling_param', str(scaling_parameter_c)],
    capture_output=True,
    text=True
)
dist_zero_to_red_one = float(result2.stdout.strip())
print(f"DIST RED ZERO 2 RED ONE: {dist_zero_to_red_one}")