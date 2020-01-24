"""Main pipeline file"""
from kubernetes import client as k8s_client
import kfp.dsl as dsl
import kfp.compiler as compiler
from kfp.dsl import Pipeline, ContainerOp
from kfp.azure import use_azure_secret
import unittest
import inspect

def transformer(op1):
    op1 = op1.apply(use_azure_secret('azcreds'))
    assert len(op1.env_variables) == 4
    index = 0
    for expected in ['AZ_SUBSCRIPTION_ID', 'AZ_TENANT_ID', 'AZ_CLIENT_ID', 'AZ_CLIENT_SECRET']:
        print(op1.env_variables[index].name)
        print(op1.env_variables[index].value_from.secret_key_ref.name)
        print(op1.env_variables[index].value_from.secret_key_ref.key)
        index += 1
    #containerOp.arguments = 
    return op1

@dsl.pipeline(
  name='Auth Test KF',
  description='Getting secrets from kf deployment'
)
def tester_train():
  """Pipeline steps"""
  operations = {}

  operations['test'] = dsl.ContainerOp(
    name='test',
    image='svangara.azurecr.io/test:1',
    command=['python'],
    arguments=[
      '/scripts/test.py',
    ]
  )

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
  compiler.Compiler().compile(tester_train, __file__ + '.tar.gz')
