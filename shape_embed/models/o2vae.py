# import torch
from torch import nn
import pythae
from functools import partial


def o2vae():
    from .o2vae.models.decoders.cnn_decoder import CnnDecoder
    from .o2vae.models.encoders_o2.e2scnn import E2SFCNN
    from .o2vae.models.vae import VAE as O2VAE

    # encoder
    q_net = E2SFCNN(
        n_channels=1,
        n_classes=64 * 2,  # bc vae saves mean and stdDev vecors
        # `name`: 'o2_cnn' for o2-invariant encoder. 'cnn_encoder' for standard cnn encoder.
        name="o2_cnn_encoder",
        # `cnn_dims`: must be 6 elements long. Increase numbers for larger model capacity
        cnn_dims=[6, 9, 12, 12, 19, 25],
        # `layer_type`: type of cnn layer (following e2cnn library examples)
        layer_type="inducedgated_norm",  # recommend not changing
        # `N`: Ignored if `name!='o2'`. Negative means the model will be O2-invariant.
        #     Again, see (e2cnn library examples). Recommend not changing.
        N=-3,
    )

    # decoder
    p_net = CnnDecoder(
        zdim=64,
        name="cnn_decoder",  # 'cnn' is the ony option
        # `cnn_dims`: each extra layer doubles the dimension (image width) by a factor of 2.
        #    E.g. if there are 6 elements, image width is 2^6=64
        cnn_dims=[192, 96, 96, 48, 48, 48],
        # cnn_dims=[192, 96, 96, 48, 48, 24, 24, 12, 12],
        out_channels=1,
    )

    # vae
    model = O2VAE(
        q_net=q_net,
        p_net=p_net,
        zdim=64,  # vae bottleneck layer
        do_sigmoid=True,  # whether to make the output be between [0,1]. Usually True.
        loss_kwargs=dict(
            # 'beta' from beta-vae, or the weight on the KL-divergence term https://openreview.net/forum?id=Sy2fzU9gl
            beta=0.01,
            # `recon_loss_type`: "bce" (binary cross entropy) or "mse" (mean square error)
            #    or "ce" (cross-entropy, but warning, not been tested well)
            # recon_loss_type="bce",
            recon_loss_type="mse",
            # for reconstrutcion loss, pixel mask. Must be either `None` or an array with same dimension as the images.
            mask=None,
            align_loss=True,  # whether to align the output image to the input image
            # whether to use efficient Foureier-based loss alignment. (Ignored if align_loss==False)
            align_fourier=True,
            # whether to do align the best rotation AND flip, instead of just rotation. (Ignored if align_loss==False)
            do_flip=True,
            # if doing brute force align loss, this is the rotation discretization. (Ignored if
            #   align_loss==False or if align_fourier==True)
            rot_steps=2,
            # Recommend not changing. The vae prior distribution. Optoins: ("standard","normal","gmm"). See models.vae.VAE for deatils.
            prior_kwargs=dict(
                prior="standard",
            ),
        ),
    )

    # extra attributes
    model.encoder = q_net
    model.decoder = p_net

    return model
