<h2> Heat Map Visualization</h2>
<p>Heatmaps of class activations are very helpful in identifying which parts of an image led the CNN to the final classificaton. It is very important when analyzing misclassified data.</p><br>
<p>We are going to use an implementation of <b>Class Activation Map (CAM)</b></p><br>
<p><b>Here I will be displaying the loaded image and further the output heatmap on the image --></b></p>

![tiger sitting](https://drive.google.com/uc?export=view&id=1ThkmgtJ4YryaIl98vqJnMW-HxKB7gwSG)
<br>
![elephant](https://drive.google.com/uc?export=view&id=16p8Xu2frtbyJVl9WNLgSyqDDki3rxGYR)
<br>
![dog](https://drive.google.com/uc?export=view&id=1jUkhdT2NbvVGc9BmLHbKvs1WeBpdCfp6)

  tiger_output = model.output[:, 292]
    last_conv_layer = model.get_layer('block5_conv3')
    
    # Gradients of the Tiger class wrt to the block5_conv3 filer
    grads = K.gradients(tiger_output, last_conv_layer.output)[0]
    
    # Each entry is the mean intensity of the gradient over a specific feature-map channel 
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    
    # Accesses the values we just defined given our sample image
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    
    # Values of pooled_grads_value, conv_layer_output_value given our input image
    pooled_grads_value, conv_layer_output_value = iterate([x])
    
    # We multiply each channel in the feature-map array by the 'importance' 
    # of this channel regarding the input image 
    for i in range(512):
        #channel-wise mean of the resulting feature map is the Heatmap of the CAM
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    
    heatmap = np.mean(conv_layer_output_value, axis=-1)
