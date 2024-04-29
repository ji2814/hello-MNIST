
GANnet = "GAN" 

if GANnet == "GAN":
    from .GAN import Generator, Discriminator
elif GANnet == "CGAN":
    from .CGAN import Generator, Discriminator
elif GANnet == "DCGAN":
    from .DCGAN import Generator, Discriminator
    
else:
    None