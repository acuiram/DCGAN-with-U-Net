# Adversarial-U-Net-Autoencoder
This project aims to restore damaged images using a GAN model with a U-Net with skip connection autoencoder as a generator and was used for my Bachelor's Degree thesis. For this project, I created my own database with artificially damaged images and reconstructed them using the DCGAN
<br>
<br>**Generator Description:** Due to the nature of the proposed subject, implementing a classical generator that starts from a noise vector is not possible. An auto-encoder could receive as input the damaged images, transform them into the latent space and be trained to restore a similar image to the ground truth. The concatenated layers have the effect of sending information directly at the decoder network, hence reducing the amount passed through bottleneck layer. The strided convolution used in this study help reduce the number of the parameters, but are more efficient for this case than a classical U-Net with MaxPooling layers. Thus, this type of U-Net will represent the base of the generator.
<br><br>**Generator Flowchart:**
<br>![generator flowchart](https://user-images.githubusercontent.com/106117736/208969818-9e95bc6c-4b63-4ba1-88c5-9c39bef58929.png)
<br><br>**Discriminator Description:** The discriminator network may be described as a function that translates image data into a probability: it classifies the images as being real(probability of one) or as being fake(probability of 0). The discriminator examines both the real images (training samples) and the generated images.
<br><br>**Discriminator Flowchart:**
<br>![dicriminator flowchart](https://user-images.githubusercontent.com/106117736/208969932-582c24c6-10fe-4767-ab02-e64927404c77.png)
<br><br>Although the database I created was not ideal, here are some of the reconstructed results:
  <img src="result_2 (1)](https://user-images.githubusercontent.com/106117736/208973531-c9acbb97-ad33-4c91-8ac0-347af21d689c.png)" width="20" /> 





