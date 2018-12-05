#onv_to_binary.py
import os
import numpy as np
from onv_process import show_onv
data_path = "/home/qyy/workspace/data/onv_9936_thick"
output_path = "/home/qyy/workspace/data/onv_9936_thick_binary"

def convert_to_binary(onv_batch):
	print onv_batch.shape
	#uni,count = np.unique(onv_batch, return_counts=True)
	#print (uni, count)
	#show_onv(onv_batch[0])
	onv_batch[onv_batch >= 127] = 255
	onv_batch[onv_batch < 127] = 0
	uni,count = np.unique(onv_batch, return_counts=True)
	print (uni, count)
	#show_onv(onv_batch[0])
	return onv_batch

if __name__ == "__main__":
	for f in os.listdir(data_path):

		onvs = np.load(os.path.join(data_path, f))
		train_onvs = onvs['train']
		valid_onvs = onvs['valid']
		test_onvs = onvs['test']
		train_onvs = convert_to_binary(train_onvs)
		valid_onvs = convert_to_binary(valid_onvs)
		test_onvs = convert_to_binary(test_onvs)

		outfile = os.path.join(output_path, f)
		if (not os.path.exists(output_path)):
	            os.makedirs(output_path)
		np.savez(outfile, train=train_onvs, valid = valid_onvs, test = test_onvs)

