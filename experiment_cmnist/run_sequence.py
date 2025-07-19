import subprocess

scaling_parameter_c = 32
img_red_zero_path = '../image_digits/red_zero.png'
img_blue_zero_path = '../image_digits/blue_zero.png'
img_red_one_path = '../image_digits/red_one.png'


# First subprocess: red_zero vs blue_zero
result1 = subprocess.run(
    ['python', 'program1.py', '--img1', img_red_zero_path, '--img2', img_blue_zero_path, '--scaling_param', str(scaling_parameter_c)],
    capture_output=True,
    text=True
)
print('Output of first subprocess (stdout):')
print(result1.stdout)
if result1.stderr:
    print('Errors of first subprocess (stderr):')
    print(result1.stderr)


# Second subprocess: red_zero vs red_one
result2 = subprocess.run(
    ['python', 'program1.py', '--img1', img_red_zero_path, '--img2', img_red_one_path, '--scaling_param', str(scaling_parameter_c)],
    capture_output=True,
    text=True
)
print('Output of second subprocess (stdout):')
print(result2.stdout)
if result2.stderr:
    print('Errors of second subprocess (stderr):')
    print(result2.stderr)
