from kfp.dsl import Pipeline, ContainerOp
from kfp.azure import use_azure_secret
import unittest
import inspect

class AzExtensionTests(unittest.TestCase):
    def test_default_secret_name(self):
        spec = inspect.getfullargspec(use_azure_secret)
        assert len(spec.defaults) == 1
        assert spec.defaults[0] == 'azcreds'

    def test_use_azure_secret(self):
        with Pipeline('somename') as p:
            op1 = ContainerOp(name='op1', image='image')
            op1 = op1.apply(use_azure_secret('azcreds'))
            assert len(op1.env_variables) == 4

            index = 0
            for expected in ['AZ_SUBSCRIPTION_ID', 'AZ_TENANT_ID', 'AZ_CLIENT_ID', 'AZ_CLIENT_SECRET']:
                print(op1.env_variables[index].name)
                print(op1.env_variables[index].value_from.secret_key_ref.name)
                print(op1.env_variables[index].value_from.secret_key_ref.key)
                index += 1

if __name__ == '__main__':
    unittest.main()