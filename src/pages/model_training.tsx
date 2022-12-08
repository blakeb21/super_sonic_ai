import type { NextPage } from "next";
import Head from "next/head";
// import { trpc } from "../utils/trpc";
import Link from "next/link";
import Header from "../components/header";
import {Prism as SyntaxHighlighter} from 'react-syntax-highlighter';
import {vscDarkPlus} from 'react-syntax-highlighter/dist/cjs/styles/prism';
import Image from "next/image";

import images from "../../public/images.jpg"


const Home: NextPage = () => {

    const deeplab = `##---------------Source-------------------------##
# Montalvo, J., García-Martín, Á. & Bescós, J. Exploiting semantic segmentation to boost reinforcement learning in video game environments. 
# Multimed Tools Appl (2022). https://doi-org.ezproxy.lib.vt.edu/10.1007/s11042-022-13695-1import 
##---------------Source-------------------------##

import os
import sys
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(script_dir + "/../..")

sys.path.append(os.path.abspath(project_dir + '/source/datasets'))	# add learning directory

from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from torch import optim
from torchvision import  models, transforms
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models

from image_tuple import * 

class DeepLab:
    def __init__(self, weight_file=None):
    
        self.pre_load    = "True" ## Load dataset in memory
        self.pre_trained = "True"
        self.num_classes = 6
        self.ignore_label = 255
        self.lr    = 0.001  # 0.001 if pretrained weights from pytorch. 0.1 if scratch
        self.M = [37,42]         # If training from scratch, reduce learning rate at some point        
        
        self.seed = 42

        ## Create arguments object
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set random seed for reproducibility
        torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
        torch.manual_seed(self.seed)  # CPU seed
        torch.cuda.manual_seed_all(self.seed)  # GPU seed
        random.seed(self.seed)  # python seed for image transformation
        np.random.seed(self.seed)

        self.workers = 0 #Anything over 0 will crash on windows. On linux it should work fine.

        model = models.segmentation.deeplabv3_resnet50(
                weights='DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1',
                progress=True)
        model.classifier = DeepLabHead(2048, 8) # Num Classes
        if weight_file is not None:
            model.load_state_dict(torch.load(weight_file, map_location=torch.device(self.device)))
        model = model.to(self.device)
        self.model=model
        # self.optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        self.optimizer = optim.Adam(model.parameters(),  lr=self.lr)

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.M, gamma=0.1)


    def train_epoch(self, args, train_loader):
        # switch to train mode
        self.model.train()

        train_loss = []
        counter = 1

        criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        for _, (images, mask) in enumerate(train_loader):

            images, mask = images.to(self.device), mask.to(self.device)

            outputs = self.model(images)['out']
    
            #Aggregated per-pixel loss
            loss = criterion(outputs, mask.squeeze(1))
            train_loss.append(loss.item())

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            if counter % 15 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Learning rate: {:.6f}'.format(
                    args.epochs, int(counter * len(images)), len(train_loader.dataset),
                    100. * counter / len(train_loader), loss.item(),
                    self.optimizer.param_groups[0]['lr']))
            counter = counter + 1
        
        return sum(train_loss) / len(train_loss) # per batch averaged loss for the current epoch.

    def _fast_hist(self, label_pred, label_true, num_classes):
        mask = (label_true >= 0) & (label_true < num_classes)
        hist = np.bincount(
            num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
        return hist

    def testing(self, test_loader):

        self.model.eval()

        loss_per_batch = []

        criterion = nn.CrossEntropyLoss(ignore_index=255)

        gts_all, predictions_all = [], []
        with torch.no_grad():
            for _, (images, mask) in enumerate(test_loader):

                images, mask = images.to(self.device), mask.to(self.device)

                outputs = self.model(images)['out']

                loss = criterion(outputs,mask.squeeze(1))
                loss_per_batch.append(loss.item())

                # Adapt output size for histogram calculation.
                preds = outputs.data.max(1)[1].squeeze(1).squeeze(0).cpu().numpy()
                gts_all.append(mask.data.squeeze(0).cpu().numpy())
                predictions_all.append(preds)

        loss_per_epoch = [np.average(loss_per_batch)]

        hist = np.zeros((self.num_classes, self.num_classes))
        for lp, lt in zip(predictions_all, gts_all):
            hist += self._fast_hist(lp.flatten(), lt.flatten(), self.num_classes)

        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))


        mean_iou = np.nanmean(iou)

        print('\nTest set ({:.0f}): Average loss: {:.4f}, mIoU: {:.4f}\n'.format(
            len(test_loader.dataset), loss_per_epoch[-1], mean_iou))

        return (loss_per_epoch, mean_iou)

    def decode_segmap(self, image, nc=8):
        ## Color palette for visualization of the 21 classes
        label_colors = np.array([(0, 0, 0),  # 0=background
                    # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                    (0, 0,255), (127, 127, 0), (0, 255, 0), (255, 0, 0), (255, 255, 0),
                    # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                    (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                    # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                    (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                    # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]
        rgb = np.stack([r, g, b], axis=2)
        return rgb

    def seg_test(self, path, transform=transforms.ToTensor()):
        img = Image.open(path).convert('RGB')
        
        input_image = transform(img).unsqueeze(0).to(self.device)
        self.model.eval()
        timer = time.time()
        out = self.model(input_image)['out'][0]
        print (f'Segmentation Time: {time.time()-timer}')

        segm = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        segm_rgb = self.decode_segmap(segm)
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(img);plt.axis('off')
        fig.add_subplot(1, 2, 2)
        plt.imshow(segm_rgb);plt.axis('off')
        #plt.savefig('1_1.png', format='png',dpi=300,bbox_inches = "tight")
        plt.show()
            
    def segment(self, image, transform=transforms.ToTensor()):
        input_image = transform(image).unsqueeze(0).to(self.device)
        timer = time.time()
        out = self.model(input_image)['out'][0]
        print (f'Segmentation Time: {time.time()-timer}')
        segm = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy() 

        return segm`

    const train = `##---------------Source-------------------------##
# Montalvo, J., García-Martín, Á. & Bescós, J. Exploiting semantic segmentation to boost reinforcement learning in video game environments. 
# Multimed Tools Appl (2022). https://doi-org.ezproxy.lib.vt.edu/10.1007/s11042-022-13695-1import 
##---------------Source-------------------------##
import os
import sys

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(script_dir + "/../..")


sys.path.append(os.path.abspath(project_dir + '/source/datasets'))
sys.path.append(os.path.abspath(project_dir + '/source/vision'))

from deeplab import *
from deeplab_dataset import *

from PIL import Image
import numpy as np
import random
import argparse
import time
from os.path import join
from tqdm import tqdm

import torch


parser = argparse.ArgumentParser()
parser.add_argument("-m","--model",default=None,type=str, help="Name of a partially trained model. Training will continue to optimize these set of weights.")
parser.add_argument("-o","--output_file",default="SegmentationModel",type=str, help="Name of the model. Will be saved on results/deeplab_ckpts")
parser.add_argument("-bs","--batch_size",default=4, choices=range(2,32),type=int, help="Keep it always 2 or more, otherwise it will crash.") ## 
parser.add_argument("-d","--dataset", default='data/segmentation_dataset', help="Path to dataset",type=str)
parser.add_argument("-e","--epochs",default=45,type=int,help="Epochs")
args = parser.parse_args()

def main():

    deep_lab = DeepLab(args.model)
    trainset = SonicDataset(args, 'train')
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=deep_lab.workers, pin_memory=True)

    testset = SonicDataset(args, 'val')
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=deep_lab.workers, pin_memory=True)

    loss_train_epoch = []
    loss_test_epoch = []
    acc_train_per_epoch = []
    acc_test_per_epoch = []
    new_labels = []
    path = os.path.join(project_dir, "results", "deeplab_ckpts")
    if not os.path.isdir(path):
        os.makedirs(path)

    for epoch in tqdm(range(1, args.epochs + 1), desc = f"DeepLabV3_Resnet50 training"):
        st = time.time()
        loss_per_epoch = deep_lab.train_epoch(args, train_loader)

        loss_train_epoch += [loss_per_epoch]

        deep_lab.scheduler.step()

        loss_per_epoch_test, acc_val_per_epoch_i = deep_lab.testing(test_loader)

        loss_test_epoch += loss_per_epoch_test
        acc_test_per_epoch += [acc_val_per_epoch_i]

        if epoch == 1:
            best_acc_val = acc_val_per_epoch_i
            
        else:
            if acc_val_per_epoch_i > best_acc_val:
                best_acc_val = acc_val_per_epoch_i

        
        torch.save(deep_lab.model.state_dict(), os.path.join(path, f'{args.output_file}_{epoch}.pt'))


# Call main
main()`

    const test = `##---------------Source-------------------------##
# Montalvo, J., García-Martín, Á. & Bescós, J. Exploiting semantic segmentation to boost reinforcement learning in video game environments. 
# Multimed Tools Appl (2022). https://doi-org.ezproxy.lib.vt.edu/10.1007/s11042-022-13695-1import 
##---------------Source-------------------------##
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(script_dir + "/../..")


sys.path.append(os.path.abspath(project_dir + '/source/datasets'))
sys.path.append(os.path.abspath(project_dir + '/source/vision'))

from deeplab import *
from deeplab_dataset import *
import argparse




parser = argparse.ArgumentParser()
parser.add_argument("-m","--model",default='results/deeplab_ckpts/SegmentationModel.pt',type=str, help="Name of the model. Will be saved on results/deeplab_ckpts")
parser.add_argument("-i","--image",required=True, help="Path to image",type=str)
args = parser.parse_args()

def main():
    seg = DeepLab(args.model)
    seg.seg_test(os.path.join(project_dir, args.image))
# Call main
main()`

  return (
    <>
      <Head>
        <title>Segmentation Training</title>
        <meta name="description" content="In order to optimize the training efficiency and accuracy of our Deep Q Learning Model, the team preprocessed images to reduce the dimensionality of input images" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="min-h-screen">
        <Header />
        <article>
          <div className="container mx-auto flex flex-col text-center p-6 md:px-8 lg:px-16">
            <h1 className="text-yellow-400 text-2xl m-2">Segmentation Training</h1>
          </div>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <p className="text-center mb-3">The team implemented the DeepLab v3 model from Google&apos;s DeepMind as our semantic segmentation model. This model has achieved great success across domains in numerous publications through its use of atrous convolution and spatial pyramid pooling. DeepMind is available with the TorchVision library and the team made 2 significant design choices to optimize performance. First, we implemented the ResNet50 backbone of the model, which strikes a good balance between model size and model accuracy. Next, we added an 8 node classification head to the model to align with our 8 segmented classes: background1, background 2, stage, Sonic, robots, items, hazards, and mechanicals.</p>
            <p className="text-center">The team elected to instantiate weights from a model pretrained on ImageNet images, hypothesizing that there may be some features and artifacts that carry over from natural images. This model had a mean intersection over union (mIoU) of 0.4385 at the first epoch, validating our hypothesis of starting with a pretrained model. The team then generated a dataset of 40,000 synthetic images containing facets of Sonic environments from all Acts. These images were split into a dataset of 80% training images and 20% validation images. The model trained on this dataset for 45 epochs to reach a final performance of 0.7214 mIoU. A simple test function was created to run “eye tests” of the models performance on unseen images as seen below.</p>
          </div>
        </article>
        <article>
            <div className="container mx-auto flex flex-row p-6 md:px-8 lg:px-16 justify-center">
                <Image alt="Preprocessed sonic image" src={images} width={870} height={920} className="max-h-96 w-auto p-6"></Image>
            </div>
        </article>
        <article className="container text-end">
          <button className="rounded bg-yellow-400 text-black p-2  mb-8 mt-8">
            <Link href={"/final_product"}>DeepQ with Segmentation -&gt;</Link>
          </button>
        </article>
        <article>
          <div className="m-6 md:mx-8 lg:mx-16">
            <h3 className="text-yellow-400 pt-6 text-center">Our Code:</h3>
            <p>source/vision/deeplab.py</p>
            <SyntaxHighlighter 
                showLineNumbers
                style={vscDarkPlus}
                language="python">
                {deeplab} 
            </SyntaxHighlighter>
          </div>
        </article>
        <article>
          <div className="m-6 md:mx-8 lg:mx-16">
            <p>source/drivers/deeplab_train.py</p>
            <SyntaxHighlighter 
                showLineNumbers
                style={vscDarkPlus}
                language="python">
                {train} 
            </SyntaxHighlighter>
          </div>
        </article>
        <article>
          <div className="m-6 md:mx-8 lg:mx-16">
            <p>source/drivers/deeplab_test.py</p>
            <SyntaxHighlighter 
                showLineNumbers
                style={vscDarkPlus}
                language="python">
                {test} 
            </SyntaxHighlighter>
          </div>
        </article>
      </main>
    </>
  );
};

export default Home;
