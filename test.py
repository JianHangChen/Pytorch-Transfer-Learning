from torchvision import models,transforms,datasets
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from torch.autograd import Variable
import torch
import time
import sys
import os



import torchvision
# import torchvision.transforms as transforms

import torch.nn.functional as F
import torch.optim as optim
import cv2





#input:
#python3 test.py --model /dir/to/save/model/

print('please copy input like:')
print('python3 test.py --model /hw6model/')

if(sys.argv[1] == '--model'):
    model_dir = '.'+ sys.argv[2]
    print('model dir is:', model_dir)
else:
	print('please copy input like:')
	print('python3 test.py --model /hw6model/')
	sys.exit()




# API (img2obj)
class img2obj(nn.Module):
	"""docstring for img2obj"""
	def __init__(self):
		super(img2obj, self).__init__()

		# self.btsize = 16

		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225])

		transform = transforms.Compose([
			transforms.Scale(224),
			transforms.ToTensor(),
			normalize,])

		self.classes = ('n01443537', 'n01629819', 'n01641577', 'n01644900', 'n01698640', 'n01742172', 'n01768244',
		'n01770393', 'n01774384', 'n01774750', 'n01784675', 'n01855672', 'n01882714', 'n01910747', 'n01917289',
		'n01944390', 'n01945685', 'n01950731', 'n01983481', 'n01984695', 'n02002724', 'n02056570', 'n02058221',
		'n02074367', 'n02085620', 'n02094433', 'n02099601', 'n02099712', 'n02106662', 'n02113799', 'n02123045',
		'n02123394', 'n02124075', 'n02125311', 'n02129165', 'n02132136', 'n02165456', 'n02190166', 'n02206856',
		'n02226429', 'n02231487', 'n02233338', 'n02236044', 'n02268443', 'n02279972', 'n02281406', 'n02321529',
		'n02364673', 'n02395406', 'n02403003', 'n02410509', 'n02415577', 'n02423022', 'n02437312', 'n02480495',
		'n02481823', 'n02486410', 'n02504458', 'n02509815', 'n02666196', 'n02669723', 'n02699494', 'n02730930',
		'n02769748', 'n02788148', 'n02791270', 'n02793495', 'n02795169', 'n02802426', 'n02808440', 'n02814533',
		'n02814860', 'n02815834', 'n02823428', 'n02837789', 'n02841315', 'n02843684', 'n02883205', 'n02892201',
		'n02906734', 'n02909870', 'n02917067', 'n02927161', 'n02948072', 'n02950826', 'n02963159', 'n02977058',
		'n02988304', 'n02999410', 'n03014705', 'n03026506', 'n03042490', 'n03085013', 'n03089624', 'n03100240',
		'n03126707', 'n03160309', 'n03179701', 'n03201208', 'n03250847', 'n03255030', 'n03355925', 'n03388043',
		'n03393912', 'n03400231', 'n03404251', 'n03424325', 'n03444034', 'n03447447', 'n03544143', 'n03584254',
		'n03599486', 'n03617480', 'n03637318', 'n03649909', 'n03662601', 'n03670208', 'n03706229', 'n03733131',
		'n03763968', 'n03770439', 'n03796401', 'n03804744', 'n03814639', 'n03837869', 'n03838899', 'n03854065',
		'n03891332', 'n03902125', 'n03930313', 'n03937543', 'n03970156', 'n03976657', 'n03977966', 'n03980874',
		'n03983396', 'n03992509', 'n04008634', 'n04023962', 'n04067472', 'n04070727', 'n04074963', 'n04099969',
		'n04118538', 'n04133789', 'n04146614', 'n04149813', 'n04179913', 'n04251144', 'n04254777', 'n04259630',
		'n04265275', 'n04275548', 'n04285008', 'n04311004', 'n04328186', 'n04356056', 'n04366367', 'n04371430',
		'n04376876', 'n04398044', 'n04399382', 'n04417672', 'n04456115', 'n04465501', 'n04486054', 'n04487081',
		'n04501370', 'n04507155', 'n04532106', 'n04532670', 'n04540053', 'n04560804', 'n04562935', 'n04596742', 'n04597913', 'n06596364', 'n07579787', 'n07583066', 'n07614500', 'n07615774', 'n07695742', 'n07711569', 'n07715103', 'n07720875', 'n07734744', 'n07747607', 'n07749582', 'n07753592', 'n07768694', 'n07871810', 'n07873807', 'n07875152', 'n07920052', 'n09193705', 'n09246464', 'n09256479', 'n09332890', 'n09428293', 'n12267677')

		print(len(self.classes))

		#  Creating alex Model

		self.model_vgg  = models.alexnet(pretrained=True)


		for param in self.model_vgg.parameters():
			param.requires_grad = False
		# self.model_vgg.classifier._modules['6'].out_features = 200

		self.model_vgg.classifier._modules['6'] = nn.Linear(4096, 200)
		# for param in self.model_vgg.classifier._modules['6'].parameters():
		# 	param.requires_grad = True

		# print(self.model_vgg)

		self.model_vgg.load_state_dict(torch.load(model_dir+'model_cpu.pth'))
	


		# dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform) for x in ['train', 'val']}

		# dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=128, shuffle=False, num_workers=6) for x in ['train', 'val']}
		# dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
		# dset_classes = dsets['train'].classes


		# trainset = torchvision.datasets.CIFAR100(root = './data',train = True,
		# 	download = True, transform = transform)
		# self.trainloader = torch.utils.data.DataLoader(trainset,batch_size = self.btsize,
		# 	shuffle = True,num_workers = 2)

		# testset = torchvision.datasets.CIFAR100(root='./data',train = False,
		# 	download = True, transform = transform)

		# self.testloader = torch.utils.data.DataLoader(testset, batch_size = self.btsize,
		# 	shuffle = False, num_workers = 2)

	def __forward__(self,x):
			# x = self.pool(F.relu(self.conv1(x)))
			# x = self.pool(F.relu(self.conv2(x)))
			# x = x.view(-1,self.__num_falt_features__(x))
			# x = F.relu(self.fc1(x))
			# x = F.relu(self.fc2(x))
			# x = self.fc3(x)
		# print(self.model_vgg)
		# print(x)
		x = self.model_vgg(x)
		# print(x)
		return x




	# [str] forward([3x224x224 ByteTensor] img)
	def forward(self, img):

		img = img.unsqueeze(0)
		x = img.float()


		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225])

		transform = transforms.Compose([
			# transforms.Scale(224),
			normalize])

		# transform = transforms.Compose([transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
		x = transform(x)

		# print(x)


		outputs = self.__forward__(Variable(x))
		# print(outputs)

		_, predicted = torch.max(outputs.data, 1)

		# print('Predicted: ', ' '.join('%5s' % self.classes[predicted[j]]
		# 	for j in range(4)))

		# print(predicted[0],'okok')
		# print(type(predicted[0]),'okok')

		print(predicted[0])
		return self.classes[predicted[0]]


	# [nil] train()
	def train(self):

		self.__showsample__()

		start = time.time()

		print(self)

		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(self.parameters(), lr = 0.001, momentum = 0.5)

		epochs = 20

		for epoch in range(epochs):  # loop over the dataset
			running_loss = 0.0
			for i, data in enumerate(self.trainloader,0):
				# get the inputs
				inputs, labels = data

				#wrap them in Variables
				inputs, labels = Variable(inputs), Variable(labels)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				outputs = self.__forward__(inputs)
				loss = criterion(outputs,labels)
				loss.backward()
				optimizer.step()

				#print statistics
				running_loss += loss.data[0]

				if i%500 == 499: #print every 2000 mini-batches
					print('[%d, %5d] loss: %.3f' %
						(epoch+1, i+1, running_loss/500))
					running_loss = 0.0

		print('Oh year! Finshed Training')
		end = time.time()
		print('time',end-start)


# test some prediction
		dataiter = iter(self.testloader)
		images, labels = dataiter.next()

		self.__imshow__(torchvision.utils.make_grid(images))
		print('GroundTruth: ', ' '.join('%5s' %
			self.classes[labels[j]] for j in range(4)))

		outputs = self.__forward__(Variable(images))

		_, predicted = torch.max(outputs.data, 1)

		print('Predicted: ', ' '.join('%5s' % self.classes[predicted[j]]
			for j in range(4)))



# accuracy
		correct = 0
		total = 0
		for data in self.testloader:
			images, labels = data
			outputs = self.__forward__(Variable(images))
			__, predicted = torch.max(outputs.data,1)
			total += labels.size(0)
			correct += (predicted == labels).sum()

		print('Accuracy of the network on the 10000 test images:%d %%'
			% (100*correct/total) )


		class_correct = list(0. for i in range(100))
		class_total = list(0. for i in range(100))

		for data in self.testloader:
			images, labels = data
			outputs = self.__forward__(Variable(images))
			_,predicted = torch.max(outputs.data, 1)
			c = (predicted == labels).squeeze()
			for i in range(self.btsize):  # batch size
				label = labels[i]
				class_correct[label] += c[i]
				class_total[label] += 1
		for i in range(100):
			print('Accuracy of %5s : %2d %%' % (
				self.classes[i], 100 * class_correct[i]/class_total[i]))


	# [nil] view([3x32x32 ByteTensor] img) -- view the image and prediction
	# Visualise one object and its caption
	def view(self,img):
		img = img.unsqueeze(0)
		x = img.float()

		transform = transforms.Compose([transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
		x = transform(x)
		images = x

		outputs = self.__forward__(Variable(x))

		_, predicted = torch.max(outputs.data, 1)

		pred_lable = self.classes[predicted[0]]



		self.__imshow__(torchvision.utils.make_grid(images))

		plt.title(pred_lable)
		plt.show()

		return


	# [nil] cam([int] /idx/) -- fetch images from the camera
	def cam(self,idx=0):
		cap = cv2.VideoCapture(idx)


		while(True):
			# Capture frame-by-frame
			ret, frame = cap.read()

			frame = cv2.resize(frame,(224,224))

			# Our operations on the frame come here
			# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			img1 = frame
			img1 = np.swapaxes(img1,0,2) #c w h
			img1 = np.swapaxes(img1,1,2) #c h w

			input_image = torch.from_numpy(img1)
			# input_image = input_image.float()


			npimg = input_image.numpy()
			# draw = plt.imshow(np.transpose(npimg, (1, 2, 0)))
			# plt.show()
			tsimg = torch.from_numpy(npimg)

			pred_label = self.forward(tsimg)
			# pred_label = self.classes[]
			print(pred_label)

			font = cv2.FONT_HERSHEY_SIMPLEX
			bottomLeftCornerOfText = (0,100)
			fontScale = 1
			fontColor = (255,255,255)
			lineType = 2
			cv2.putText(frame,pred_label, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

			# Display the resulting frame
			c2img = cv2.imshow('please see the print result',frame)
			# c2img = cv2.imshow(pred_label,frame)




			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			# time.sleep(5)
			# cv2.destroyWindow(pred_label)

		# When everything done, release the capture
		cap.release()
		cv2.destroyAllWindows()

		return






# img2obj()
TTEST = img2obj()
TTEST.cam()