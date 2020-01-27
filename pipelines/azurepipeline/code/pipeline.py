"""Main pipeline file"""
from kubernetes import client as k8s_client
import kfp.dsl as dsl
import kfp.compiler as compiler
from kfp.dsl import _container_op
from kfp.dsl import _resource_op
from kfp.dsl import _ops_group
from kubernetes.client.models import V1EnvVar


from azureml.core import Workspace
ws=Workspace.from_config()

# transforms a given container op to use the pipelineWrapper
# in each step of the pipeline
def transformer(containerOp):
  containerOp.arguments = ['/scripts/pipelineWrapper.py', 'Tacos vs. Burritos', 'python'] + containerOp.arguments
  # shouldn't hard code this experiment name
  
  containerOp.container.set_image_pull_policy("Always")
  containerOp.add_volume(
    k8s_client.V1Volume(
      name='azure',
      persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
        claim_name='azure-managed-disk')
    )
  ).add_volume_mount(k8s_client.V1VolumeMount(
    mount_path='/mnt/azure', name='azure'))
  containerOp.container.add_env_variable(V1EnvVar(name='AZ_NAME', value=ws.name)
    ).add_env_variable(V1EnvVar(name='AZ_SUBSCRIPTION_ID', value=ws.subscription_id)
    ).add_env_variable(V1EnvVar(name='AZ_RESOURCE_GROUP', value=ws.resource_group))

  return containerOp


@dsl.pipeline(
  name='Tacos vs. Burritos',
  description='Simple TF CNN'
)
def tacosandburritos_train(
):
  """Pipeline steps"""

  persistent_volume_path = '/mnt/azure'
  data_download = 'https://aiadvocate.blob.core.windows.net/public/tacodata.zip'
  epochs = 5
  batch = 32
  learning_rate = 0.0001
  model_name = 'tacosandburritos'
  profile_name = 'tacoprofile'
  operations = {}
  image_size = 160
  training_folder = 'train'
  training_dataset = 'train.txt'
  model_folder = 'model'


  # train
  operations['train'] = dsl.ContainerOp(
    name='train',
    image='svangara.azurecr.io/training:1',
    command=['python'],
    arguments=[
      '/scripts/train.py',
      '--outputs', model_folder
    ]
  )

  dsl.get_pipeline_conf().add_op_transformer(transformer)

if __name__ == '__main__':
  compiler.Compiler().compile(tacosandburritos_train, __file__ + '.tar.gz')
