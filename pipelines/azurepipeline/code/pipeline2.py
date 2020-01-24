"""Main pipeline file"""
from kubernetes import client as k8s_client
import kfp.dsl as dsl
import kfp.compiler as compiler
from . import _container_op
from . import _resource_op
from . import _ops_group

# transforms a given container op to use the pipelineWrapper
# in each step of the pipeline
def transformer(containerOp):
  containerOp.arguments = ['pipelineWrapper.py', 'Tacos vs. Burritos', 'python'] + containerOp.arguments
  # shouldn't hard code this experiment name
  return containerOp

@dsl.pipeline(
  name='Tacos vs. Burritos',
  description='Simple TF CNN'
)
def tacosandburritos_train(
    tenant_id,
    service_principal_id,
    service_principal_password,
    subscription_id,
    resource_group,
    workspace
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

  # preprocess data
  operations['preprocess'] = dsl.ContainerOp(
    name='preprocess',
    image='svangara.azurecr.io/preprocess:2',
    command=['python'],
    arguments=[
      '/scripts/data.py',
      '--base_path', persistent_volume_path,
      '--data', training_folder,
      # '--target', training_dataset,
      '--img_size', image_size,
      '--zipfile', data_download
    ]
  )

  # train
  operations['training'] = dsl.ContainerOp(
    name='training',
    image='svangara.azurecr.io/training:2',
    command=['python'],
    arguments=[
      '/scripts/train.py',
      '--base_path', persistent_volume_path,
      # '--data', training_folder,
      '--epochs', epochs,
      '--batch', batch,
      '--image_size', image_size,
      '--lr', learning_rate,
      '--outputs', model_folder,
      '--dataset', training_dataset
    ]
  )
  operations['training'].after(operations['preprocess'])

  for _, op_1 in operations.items():
    op_1.container.set_image_pull_policy("Always")
    op_1.add_volume(
      k8s_client.V1Volume(
        name='azure',
        persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
          claim_name='azure-managed-disk')
      )
    ).add_volume_mount(k8s_client.V1VolumeMount(
      mount_path='/mnt/azure', name='azure'))
  
  dsl.get_pipeline_conf().add_op_transformer(transformer)

if __name__ == '__main__':
  compiler.Compiler().compile(tacosandburritos_train, __file__ + '.tar.gz')
