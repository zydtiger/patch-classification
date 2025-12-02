Things to note:
model param size

- ResNet-50: 24692297 trainable parameters, further trained on top of pytorch hub SOTA resnet50 pretrained weights
  - Early stopped after 12 epochs
  - Final validation acc: 99.64%
    - Batch size: 32
    - Total time: 61min
  - VGG-16: 121683785 trainable params, further trained on SOTA VGG16 pretrained
    - Early stopping after 15 epochs
    - Final validation acc: 98.87%
    - Batch size: 32
    - Total time: 79min
  - Embedding:
    - Virchow2 params (pretrained, not fine-tuned, directly used to generate embeddings): 632M
    - Total time: 9.5min
  - PatchMIL: ??? params, trained from scratch
    - ???
    - Total time: 8.3min
  - Multiclass MLP: ??? params, trained from scratch
    - ???
    - Total time: 16.7min

model training epochs, and total training time
model eval accuracy
model ROC multiclass
model confusion matrix
