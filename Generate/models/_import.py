
GANnet = "cDCGAN" 

if GANnet == "GAN":
    from .GAN import Generator, Discriminator
elif GANnet == "cDCGAN":
    from .cDCGAN import Generator, Discriminator
elif GANnet == "DCGAN":
    from .DCGAN import Generator, Discriminator
    
else:
    None