## Explaination of the code:
```
# Function to return the model
def mobilenet_v2():
    # Load the pretrained model from torch hub
    model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
    
    # Make the layers as non trainable, since we want only the last layer to be trained
    for param in model.parameters():
        param.requires_grad = False
        
    # Get the number of features which in the last layer
    num_filters = model.classifier[1].in_features

    # Modified the last layer, as make the out features as 4 since we have only 4 classes.
    model.classifier[1] = torch.nn.Linear(num_filters, 4)
    
    # make this layer trainable
    model.classifier[1].requires_grad = True
    
    # return the model
    return model 
```

## Resizing Strategy:
The image has been resized to 224 * 224. All the corrupt images has been removed.

## Model:
The model is Mobilenet v2 for training.

## accuracy vs epochs graphs for train and test curves

## 10 misclassified images for each of the classes as an Image Gallery
<img src='https://github.com/futartup/eva-session-2/blob/master/mobilenet-v2-session-2/misclassified.jpg' width="2000" height="1000">
