Subject: A technical architecture diagram of VGG16 neural network for histopathology image classification, specifically showing the sequential convolution blocks and dense classifier layers characteristic of VGG architecture.

Style: Academic textbook illustration, clean vector style, schematic, white background.

Composition (Left to Right):

1. Input: A batch of 3x224x224 RGB images on the far left containing histopathology tissue patches (showing cellular structure with purple/pink staining typical of H&E slides). Label: "B×3×224×224".
2. Initial Convolution Block: Two convolutional layers stacked vertically as thin blocks, with the width and height of each block corresponding to the spatial dimensions (224×224) and thin layering showing the 64 channels.
   - Color: Light slate blue with thin black outlines
   - Label: "2× 3×3 Conv, 64 filters"
   - Output dimension: "B×64×224×224"
3. Max Pooling Layer 1: A pooling layer represented as a smaller block.
   - Color: Light gray
   - Label: "2×2 Max Pool, stride=2"
   - Output dimension: "B×64×112×112"
4. Convolution Block 2: Two convolutional layers stacked vertically as thin blocks, with width and height corresponding to spatial dimensions (112×112) and thin layering showing the 128 channels.
   - Color: Progressive shades of medium blue
   - Label: "2× 3×3 Conv, 128 filters"
   - Output dimension: "B×128×112×112"
5. Max Pooling Layer 2: Another pooling layer.
   - Color: Light gray
   - Label: "2×2 Max Pool, stride=2"
   - Output dimension: "B×128×56×56"
6. Convolution Block 3: Three convolutional layers stacked vertically as thin blocks, with width and height corresponding to spatial dimensions (56×56) and thin layering showing the 256 channels.
   - Color: Progressive shades of teal
   - Label: "3× 3×3 Conv, 256 filters"
   - Output dimension: "B×256×56×56"
7. Max Pooling Layer 3: Another pooling layer.
   - Color: Light gray
   - Label: "2×2 Max Pool, stride=2"
   - Output dimension: "B×256×28×28"
8. Convolution Block 4: Three convolutional layers stacked vertically as thin blocks, with width and height corresponding to spatial dimensions (28×28) and thin layering showing the 512 channels.
   - Color: Progressive shades of green
   - Label: "3× 3×3 Conv, 512 filters"
   - Output dimension: "B×512×28×28"
9. Max Pooling Layer 4: Another pooling layer.
   - Color: Light gray
   - Label: "2×2 Max Pool, stride=2"
   - Output dimension: "B×512×14×14"
10. Convolution Block 5: Three convolutional layers stacked vertically as thin blocks, with width and height corresponding to spatial dimensions (14×14) and thin layering showing the 512 channels.
    - Color: Progressive shades of orange
    - Label: "3× 3×3 Conv, 512 filters"
    - Output dimension: "B×512×14×14"
11. Max Pooling Layer 5: Final pooling layer.
    - Color: Light gray
    - Label: "2×2 Max Pool, stride=2"
    - Output dimension: "B×512×7×7"
12. Global Average Pooling: A single small block representing adaptive average pooling.
    - Color: Peach/orange
    - Label: "Global Avg Pool"
    - Output dimension: "B×512×7×7"
13. Custom Classifier Head: Three fully connected layers shown as columns of circles.
    - First Layer: Large column of green circles (4096 neurons)
      - Label: "FC1: 512×7×7 → 4096"
      - Add dropout indicator: small "Dropout(0.5)" text
    - Second Layer: Medium column of teal circles (1024 neurons)
      - Label: "FC2: 4096 → 1024"
      - Add dropout indicator: small "Dropout(0.5)" text
    - Output Layer: Small column of red circles (9 neurons)
      - Label: "Output: 1024 → 9"
      - Add small text labels: "ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM"

Connections:
- Use dotted gray lines projecting from small square patches to visualize convolution kernels
- Show direct connections between consecutive convolutional layers (no skip connections like ResNet)
- Dense mesh of thin blue lines connecting the fully connected layers
- Include max pooling reduction arrows showing spatial dimension reduction

Annotations:
- Include curly brackets above groups of layers labeled with operation names
- Add dimension labels below each major block
- Highlight the uniform 3×3 convolution pattern throughout all blocks
- Add text labels for key architectural features: "Sequential Design", "Uniform Convolutions", "Progressive Depth", "Dense Classifier", "Feature Extractor"
- Include small icons showing the histopathology classes at the output
- Add labels showing the progressive reduction: "×2" at each pooling layer

Technical Details to Emphasize:
- The characteristic VGG16 sequential pattern (no residual connections)
- The uniform 3×3 kernel size throughout all convolutional layers
- Progressive reduction in spatial dimensions (224→112→56→28→14→7)
- Progressive increase in channel depth (64→128→256→512→512)
- Large fully connected classifier layers with dropout
- Total parameter count: 121.7M
- Input size annotation: "B×3×224×224"

Vibe: Educational, precise, high-contrast, informative, professional medical AI architecture diagram suitable for academic papers or technical documentation.