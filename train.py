import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

import net
from sampler import InfiniteSamplerWrapper

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--model', type=str, choices=['lpadain', 'adain'], default='lpadain',
                        help='Select base model: LPAdaIN or AdaIN') # new! We can change the model!
    parser.add_argument('--content_dir', type=str, required=True,
                        help='Directory path to a batch of content images')
    parser.add_argument('--style_dir', type=str, required=True,
                        help='Directory path to a batch of style images')
    parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

    # training options
    parser.add_argument('--save_dir', default='./experiments',
                        help='Directory to save the model')
    parser.add_argument('--log_dir', default='./logs',
                        help='Directory to save the log')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--max_iter', type=int, default=160000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--style_weight', type=float, default=10.0)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--rec_weight', type=float, default=100.0, help='Reconstruction weight') # We can change reconstruction loss' weight
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--save_model_interval', type=int, default=10000)
    parser.add_argument('--depth', type=int, choices=[1, 2, 3, 4], default=3,
                        help='Depth for training model') # new! We can change the depth!
    parser.add_argument('--cbam', action='store_true', help="Enable CBAM (default: True)")
    parser.add_argument('--no-cbam', dest='cbam', action='store_false', help="Disable CBAM")
    parser.set_defaults(cbam=True) # new! We can on/off cbam!
    parser.add_argument('--mul_cbam', action='store_true', help="Enable multilayer CBAM (default: False)")
    parser.add_argument('--no-mul_cbam', dest='mul_cbam', action='store_false', help="Disable multilayer CBAM")
    parser.set_defaults(mul_cbam=False) # new! We can on/off multilayer cbam! (can't use with 'adain' model, only for 'lpadain')
    parser.add_argument('--rec_cbam', action='store_true', help="Put CBAM Layer in auxiliary branch (default: False)")
    parser.add_argument('--no-rec_cbam', dest='rec_cbam', action='store_false', help="Delete CBAM Layer in auxiliary branch")
    parser.set_defaults(rec_cbam=False) # We can on/off cbam layer in auxiliary branch
    args = parser.parse_args()

    assert(args.model != 'adain' or not args.mul_cbam)
    device = torch.device('cuda')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    decoder = net.decoder
    vgg = net.vgg

    vgg.load_state_dict(torch.load(args.vgg, weights_only=True))

    if args.depth == 1:
        vgg = nn.Sequential(*list(vgg.children())[:4])
        decoder = nn.Sequential(*list(decoder.children())[27:])
    elif args.depth == 2:
        vgg = nn.Sequential(*list(vgg.children())[:11])
        decoder = nn.Sequential(*list(decoder.children())[20:])
    elif args.depth == 3:
        vgg = nn.Sequential(*list(vgg.children())[:18])
        decoder = nn.Sequential(*list(decoder.children())[13:])
    elif args.depth == 4:
        vgg = nn.Sequential(*list(vgg.children())[:31])
    network = net.Net(vgg, decoder, args.depth, args.cbam, args.mul_cbam, args.rec_cbam)
    print(vars(network))
    network.train()
    network.to(device)

    content_tf = train_transform()
    style_tf = train_transform()

    content_dataset = FlatFolderDataset(args.content_dir, content_tf)
    style_dataset = FlatFolderDataset(args.style_dir, style_tf)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads))

    params = list(network.decoder.parameters())
    if args.mul_cbam:
        for i in range(args.depth):
            params = list(getattr(network, 'cbam_{:d}'.format(i + 1)).parameters()) + params
    elif args.cbam:
        params = list(network.cbam.parameters()) + params

    optimizer = torch.optim.Adam(params, lr=args.lr)

    for i in tqdm(range(args.max_iter)):
        adjust_learning_rate(optimizer, iteration_count=i)
        content_images = next(content_iter).to(device)
        style_images = next(style_iter).to(device)
        loss_c, loss_s, loss_r = network(content_images, style_images, args.depth, args.model, args.cbam, args.mul_cbam)
        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        loss_r = args.rec_weight * loss_r
        loss = loss_c + loss_s + loss_r

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('loss_content', loss_c.item(), i + 1)
        writer.add_scalar('loss_style', loss_s.item(), i + 1)
        writer.add_scalar('loss_reconstruction', loss_r.item(), i + 1)

        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            state_dict = {'decoder': network.decoder.state_dict()}
            if args.mul_cbam:
                for j in range(args.depth):
                    state_dict['cbam_{:d}'.format(j + 1)] = getattr(network, 'cbam_{:d}'.format(j + 1)).state_dict()
            elif args.cbam:
                state_dict['cbam'] = network.cbam.state_dict()
            for key in state_dict.keys():
                for param_key, param_value in state_dict[key].items():
                    state_dict[key][param_key] = param_value.to(torch.device('cpu'))
            torch.save(state_dict, save_dir /
                    'layer{:d}_iter_{:d}.pth.tar'.format(args.depth, i + 1))
    writer.close()
