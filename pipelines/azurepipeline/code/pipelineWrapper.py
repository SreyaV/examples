import sys
import os
import subprocess
import json 
import azureml.core
from azureml.core import Workspace, Experiment, Run
from azureml.core import ScriptRunConfig


def run_command(program_and_args, # ['python', 'foo.py', '3']
                working_dir=None, # Defaults to current directory
                env=None):

    if working_dir is None:
        working_dir = os.getcwd()

    output = ""
    process = subprocess.Popen(program_and_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=working_dir, shell=False, env=env)
    for line in process.stdout:
        line = line.decode("utf-8").rstrip()
        if line and line.strip():
            # Stream the output
            sys.stdout.write(line)
            sys.stdout.write('\n')
            # Save it for later too
            output += line
            output += '\n'

    process.communicate()
    retcode = process.poll()

    if retcode:
        raise subprocess.CalledProcessError(retcode, process.args, output=output, stderr=process.stderr)

    return retcode, output


if __name__ == "__main__":
    job_info_path = "parent_run.json"
    experiment_name = sys.argv[1]
    run_name = sys.argv[3][:-3] # should be the file name


    if os.path.exists(job_info_path):
        # get parent run id, experiment name from file & workspace obj
        # create child run (id )
        with open(job_info_path, 'r') as f:
            job_info_dict = json.load(f)
        print("dictionary read from file " + job_info_dict+ "\n")
        run_id = job_info_dict["run_id"]
        ws = Workspace.from_config() # TODO set path and auth 
        exp = Experiment(workspace=ws, name=experiment_name)
        run = Run(exp, run_id)
        run.child_run(name=run_name) # TODO: add the step's name 
    else:
        # start run
        ws = Workspace.from_config()
        exp = Experiment(workspace=ws, name=experiment_name) 
        run = exp.start_logging()
        job_info_dict = {"run_id": run._run_id, "experiment_name": exp.name, "experiment_id": exp._id}
        json = json.dumps(job_info_dict)
        f = open(job_info_path,"w")
        f.write(json)
        f.close()
    
    ret, _ = run_command(sys.argv[2:])
    # ret, _ = run_command("python preprocess/data.py")