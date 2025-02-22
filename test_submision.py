import os
import sys
import scipy.io as sio
from pathlib import Path
import subprocess

def combine_input_output(input_dir,output_dir):
    return ' '.join(str(item) for pair in zip(input_dir, output_dir) for item in pair)

def run_project():

    python_file = Path(f"../{eval_dir}/main.py")
    octave_file = Path(f"../{eval_dir}/main.m")

    input_output_dir = combine_input_output(input_dir,output_dir)

    if python_file.is_file():
        print(f'Running with python...\n')
        command = ["python3",f'../{eval_dir}/main.py', ref_dir] + input_output_dir.split()
        result = subprocess.run(command, capture_output=True, text=True)
    elif octave_file.is_file():
        print(f'Running with octave...\n')
        command = ["octave", "--no-gui", "--eval" f"run('../{eval_dir}/main.m')",ref_dir] + input_output_dir.split()
        result = subprocess.run(command, capture_output=True, text=True)
    else:
        result = None

    if result is None:
        print('No main file found or with incorrect name. Please name the main file as main.py or main.m')
        return False
    if not (result.stderr == ""):
        print(f"There are errors in the main file. Please check the code and try again.\n")
        print(f"Errors:\n{result.stderr}\n")
        return False
    
    print(f'Finished running main\n')

    return True

eval_dir = 'part12submission'

if len(sys.argv) < 4:
    print("Usage: python3 test_submission.py ref input1 output1 input2 output2 ... inputN outputN")
    sys.exit(1)

ref_dir = sys.argv[1]
input_dir = sys.argv[2::2]
output_dir = sys.argv[3::2]

if len(input_dir) != len(output_dir):
    print("Error: Number of input directories must match number of output directories")
    sys.exit(1)

for outputDir in output_dir:
    if not os.path.exists(outputDir):
        print(f"Error: directory {outputDir} could not be found. Make sure it exists. If it doesn't you can create it.")
        sys.exit(1)
    for mat_file in Path(outputDir).glob("*.mat"):
        try:
            os.remove(mat_file)
        except Exception as e:
            print(f"Error deleting {mat_file}: {e}")

if run_project():
    for index, output_homo_dir in enumerate(output_dir):
        homo_file = Path(f"{output_homo_dir}/homographies.mat")

        if homo_file.is_file():
            student_data = sio.loadmat(f"{output_homo_dir}/homographies.mat")
        else:
            print(f"No homographies.mat file or with incorrect name in {output_homo_dir}. Please save the homographies in a file named homographies.mat in the corresponding outputK directory.")
            sys.exit(1)
                
        if "H" in student_data:
            student_H = student_data["H"]
        else:
            print(f'No "H" key found in dictionary from {output_homo_dir}/homographies.mat')
            sys.exit(1)
        
        image_files = [f for f in os.listdir(f'{input_dir[index]}') if f.endswith('.jpg')]
        N = len(image_files)
        
        if student_H.shape != (3,3,N):
            print(f'Homographies matrix in {output_homo_dir}/homographies.mat doesn\'t have the correct shape. Has shape {student_H.shape} when it should have (3,3,{N}). Homographies matrix must be (3,3,Nv) where Nv is the number of images of the corresponding input folder.')
            sys.exit(1)

    print('Everything is OK for submission.')
            