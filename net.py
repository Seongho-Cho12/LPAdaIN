import torch
import torch.nn as nn

from function import adaptive_instance_normalization as adain
from function import calc_mean_std
from function import CBAM

device = torch.device('cuda')

decoder = nn.Sequential(
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 256, (3, 3)),
    # nn.ReLU(),
    # nn.Upsample(scale_factor=2, mode='nearest'),
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(256, 256, (3, 3)),
    # nn.ReLU(),
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(256, 256, (3, 3)),
    # nn.ReLU(),
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(256, 256, (3, 3)),
    # nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        # self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3']: # , 'enc_4' <- erased
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(3): # changed 4 -> 3
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    # def calc_style_loss(self, input, target):
    #     assert (input.size() == target.size())
    #     assert (target.requires_grad is False)
    #     input_mean, input_std = calc_mean_std(input)
    #     target_mean, target_std = calc_mean_std(target)
    #     return self.mse_loss(input_mean, target_mean) + \
    #            self.mse_loss(input_std, target_std)
    
    # new function!
    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        # style_feats = self.encode_with_intermediate(style)
        # content_feat = self.encode(content)
        # t = adain(content_feat, style_feats[-1])
        # t = alpha * t + (1 - alpha) * content_feat
        ### new! ###
        t = content.clone()
        content_feat = content.clone()
        style_feats = self.encode_with_intermediate(style.clone())
        for i in range(3):
            t = getattr(self, 'enc_{:d}'.format(i + 1))(t)
            content_feat = getattr(self, 'enc_{:d}'.format(i + 1))(content_feat)
            t = adain(t, style_feats[i])
            t = alpha * t + (1 - alpha) * content_feat
        cbam = CBAM(t.size(1), r=2).to(device)
        cbam_t = cbam(t).detach()
        ############
        g_t = self.decoder(cbam_t)
        g_t_feats = self.encode_with_intermediate(g_t)

        # loss_c = self.calc_content_loss(g_t_feats[-1], t)
        # loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        # for i in range(1, 4):
        #     loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        ### new! ###
        loss_c = self.calc_content_loss(g_t_feats[-1], cbam_t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 3):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])

        # Auxiliary Branch 1 - Content Reconstruction
        content_rec = self.decoder(content_feat)
        loss_content_rec = self.calc_content_loss(content_rec, content)

        # Auxiliary Branch 2 - Style Reconstruction
        style_rec = self.decoder(style_feats[-1])
        loss_style_rec = self.calc_content_loss(style_rec, style)

        loss_r = loss_content_rec + loss_style_rec
        ############
        return loss_c, loss_s, loss_r