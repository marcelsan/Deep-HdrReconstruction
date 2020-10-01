import importlib
import sys
sys.path.append("..")

from lib.io import print_

def load(network, device):
	module = network[:network.rfind('.')]
	model = network[network.rfind('.')+1:]
	mod = importlib.import_module(module)
	net_func = getattr(mod, model)

	net = net_func().to(device)

	num_params = sum([param.nelement() for param in net.parameters()])
	print_('\tModel {} loaded. Model params = {:2.1f}M\n'.format(network, num_params / 1000000), bold=True)

	return net