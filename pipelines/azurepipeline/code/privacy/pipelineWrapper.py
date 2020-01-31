import sys
import os
import subprocess
import json 
import mlflow 
import azureml.core
from azureml.core import Workspace, Experiment, Run
from azureml.core import ScriptRunConfig
import azureml.mlflow
from azureml.mlflow import _setup_remote, _get_mlflow_tracking_uri
from azureml.core.authentication import ServicePrincipalAuthentication

def get_ws():
  auth_args = {
    'tenant_id': os.environ.get('AZ_TENANT_ID'),
    'service_principal_id': os.environ.get('AZ_CLIENT_ID'),
    'service_principal_password': os.environ.get('AZ_CLIENT_SECRET')
  }
 
  ws = Workspace.get(name=os.environ.get('AZ_NAME'), auth=ServicePrincipalAuthentication(**auth_args), subscription_id=os.environ.get('AZ_SUBSCRIPTION_ID'), resource_group=os.environ.get('AZ_RESOURCE_GROUP'))
  return ws

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
    sys.argv.append("1")
    experiment_name = sys.argv[1]
    run_name = sys.argv[3][:-3] # should be the file name

    env_dictionary = {"MLFLOW_EXPERIMENT_NAME": experiment_name} 
    if os.path.exists(job_info_path):
        # get parent run id, experiment name from file & workspace obj
        # create child run (id )
        with open(job_info_path, 'r') as f:
            job_info_dict = json.load(f)
        print("Dictionary read from file " + job_info_dict+ "\n")
        run_id = job_info_dict["run_id"]
        ws = get_ws() # TODO set path and auth 
        exp = Experiment(workspace=ws, name=experiment_name)
        run = Run(exp, run_id)
        run.child_run(name=run_name) # TODO: add the step's name 
        tags = {"mlflow.source.type": "JOB", "mlflow.source.name": "train.py", "mlflow.user": "srevan"}
        run.set_tags(tags)
        # log environment variables
        env_dictionary["MLFLOW_EXPERIMENT_ID"] = exp._id
        env_dictionary["MLFLOW_RUN_ID"] = run_id
        env_dictionary["MLFLOW_TRACKING_URI"] = _get_mlflow_tracking_uri(ws)
        env_dictionary["HOME"] = "~/"
    else:
        # start run
        ws = get_ws()
        exp = Experiment(workspace=ws, name=experiment_name) 
        run = exp.start_logging(snapshot_directory="/scripts") 
        run.child_run(name=run_name) # TODO: add the step's name 
        tags = {"mlflow.source.type": "JOB", "mlflow.source.name": "train.py", "mlflow.user": "srevan"}
        run.set_tags(tags)
        job_info_dict = {"run_id": run._run_id, "experiment_name": exp.name, "experiment_id": exp._id}
        json_dict = json.dumps(job_info_dict)
        with open(job_info_path,"w") as f:
            f.write(json_dict)
            f.close()
        # log environment variables
        env_dictionary["MLFLOW_EXPERIMENT_ID"] = exp._id
        env_dictionary["MLFLOW_RUN_ID"] = run._run_id
        env_dictionary["MLFLOW_TRACKING_URI"] = _get_mlflow_tracking_uri(ws)
        env_dictionary["HOME"] = "~/"
    
    print("Before running train")
    try:
        print("Trying to run train file ")
        ret, _ = run_command([sys.executable] + sys.argv[3:], env=env_dictionary)
    except subprocess.CalledProcessError as e:
        print("Subprocess caused error " + run_name)
        better_e = RuntimeError("{}\n{}".format(e.stderr, e))
        run.fail(error_details=better_e)
        raise better_e
    else:
        run.complete()
        print("Marked as complete")