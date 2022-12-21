# adversarial-U-Net-autoencoder
Damaged image restauration using a GAN model with a U-Net with skip connection autoencoder as a generator
<br>For this project, I created my own database with artificially damaged images and reconstructed them using the DCGAN

<br>
<br>**Generator Description:** Due to the nature of the proposed subject, implementing a classical generator that starts from a noise vector is not possible. An auto-encoder could receive as input the damaged images, transform them into the latent space and be trained to restore a similar image to the ground truth. The concatenated layers have the effect of sending information directly at the decoder network, hence reducing the amount passed through bottleneck layer. The strided convolution used in this study help reduce the number of the parameters, but are more efficient for this case than a classical U-Net with MaxPooling layers. Thus, this type of U-Net will represent the base of the generator
<br>Generator Flowchart:
<br>![generator flowchart](https://user-images.githubusercontent.com/106117736/208969818-9e95bc6c-4b63-4ba1-88c5-9c39bef58929.png)

<br>Discriminator Flowchart:
<br>![dicriminator flowchart](https://user-images.githubusercontent.com/106117736/208969932-582c24c6-10fe-4767-ab02-e64927404c77.png)