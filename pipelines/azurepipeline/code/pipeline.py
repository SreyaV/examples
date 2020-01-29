"""Main pipeline file"""
from kubernetes import client as k8s_client
import kfp.dsl as dsl
import kfp.compiler as compiler
from kfp.dsl import _container_op
from kfp.dsl import _resource_op
from kfp.dsl import _ops_group
from kubernetes.client.models import V1EnvVar
from kfp.azure import use_azure_secret


from azureml.core import Workspace
ws=Workspace.from_config()

# transforms a given container op to use the pipelineWrapper
# in each step of the pipeline
def transformer(containerOp):
  containerOp.arguments = ['/scripts/pipelineWrapper.py', 'Privacy', 'python'] + containerOp.arguments
  # shouldn't hard code this experiment name
  
  containerOp.container.set_image_pull_policy("Always")
  containerOp.add_volume(
    k8s_client.V1Volume(
      name='azure',
      persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
        claim_name='azure-managed-disk')
    )
  ).add_volume_mount(k8s_client.V1VolumeMount(mount_path='/mnt/azure', name='azure'))

  containerOp.container.add_env_variable(V1EnvVar(name='AZ_NAME', value=ws.name))\
    .add_env_variable(V1EnvVar(name='AZ_SUBSCRIPTION_ID', value=ws.subscription_id))\
    .add_env_variable(V1EnvVar(name='AZ_RESOURCE_GROUP', value=ws.resource_group))
  containerOp.apply(use_azure_secret('azcreds'))

  return containerOp


@dsl.pipeline(
  name='test',
  description='Privacy Experiment'
)
def test_train(
):
  """Pipeline steps"""

  persistent_volume_path = '/mnt/azure'
  model_name = 'test'
  operations = {}
  image_size = 160
  training_folder = 'train'
  training_dataset = 'train.txt'
  model_folder = 'Privacy'


  # train
  operations['train'] = dsl.ContainerOp(
    name='train',
    image='svangara.azurecr.io/training:3',
    command=['python'],
    arguments=[
      '/scripts/train.py',
      '--outputs', model_folder
    ]
  )

  dsl.get_pipeline_conf().add_op_transformer(transformer)

if __name__ == '__main__':
  compiler.Compiler().compile(test_train, __file__ + '.tar.gz')
