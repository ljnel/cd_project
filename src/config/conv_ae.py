from dataclasses import dataclass


@dataclass
class ConvAEConfig:
    in_len: int
    in_chan: int
    out_chan: int
    latent_dim: int


CAE_CFG = {
    'rec': {
        'inv_pend': ConvAEConfig(in_len=45, in_chan=4, out_chan=20, latent_dim=5),
        'hopper': ConvAEConfig(in_len=70, in_chan=11, out_chan=30, latent_dim=5),
        'half_cheetah': ConvAEConfig(in_len=10, in_chan=17, out_chan=40, latent_dim=5),
        'humanoid': ConvAEConfig(in_len=22, in_chan=348, out_chan=60, latent_dim=5)
    },
    'lat': {
        'inv_pend': ConvAEConfig(in_len=45, in_chan=4, out_chan=20, latent_dim=20),
        'hopper': ConvAEConfig(in_len=70, in_chan=11, out_chan=40, latent_dim=30),
        'half_cheetah': ConvAEConfig(in_len=10, in_chan=17, out_chan=40, latent_dim=30),
        'humanoid': ConvAEConfig(in_len=22, in_chan=348, out_chan=80, latent_dim=30)
    }
}
