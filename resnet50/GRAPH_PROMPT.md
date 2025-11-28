Subject: A technical architecture diagram of ResNet50 neural network for histopathology image classification, specifically showing the residual connections and bottleneck blocks characteristic of ResNet architecture.

Style: Academic textbook illustration, clean vector style, schematic, white background.

Composition (Left to Right):

1. Input: A batch of 3x224x224 RGB images on the far left containing histopathology tissue patches (showing cellular structure with purple/pink staining typical of H&E slides). Label: "B×3×224×224".
2. Initial Convolution: A single large convolutional block rendered as an isometric stack of 2D planes showing 64 channels.
   - Color: Light slate blue with thin black outlines
   - Label: "7×7 Conv, stride=2"
   - Output dimension: "B×64×112×112"
3. Max Pooling: A pooling layer represented as a smaller block.
   - Color: Light gray
   - Label: "3×3 Max Pool, stride=2"
   - Output dimension: "B×64×56×56"
4. Residual Stage 1: Three bottleneck blocks arranged in a stack, each showing the 1×1, 3×3, 1×1 convolution pattern.
   - Color: Progressive shades of blue getting darker
   - Show residual connections as curved arrows skipping over blocks
   - Label: "Stage 1: 3× Bottleneck"
   - Output dimension: "B×256×56×56"
5. Residual Stage 2: Four bottleneck blocks with increased channel depth.
   - Color: Progressive shades of teal
   - Show residual connections as curved arrows
   - Label: "Stage 2: 4× Bottleneck"
   - Output dimension: "B×512×28×28"
6. Residual Stage 3: Six bottleneck blocks, the deepest stage.
   - Color: Progressive shades of green
   - Show residual connections as curved arrows
   - Label: "Stage 3: 6× Bottleneck"
   - Output dimension: "B×1024×14×14"
7. Residual Stage 4: Three bottleneck blocks, final stage.
   - Color: Progressive shades of orange
   - Show residual connections as curved arrows
   - Label: "Stage 4: 3× Bottleneck"
   - Output dimension: "B×2048×7×7"
8. Global Average Pooling: A single small block representing adaptive average pooling.
   - Color: Peach/orange
   - Label: "Global Avg Pool"
   - Output dimension: "B×2048"
9. Custom Classifier Head: Three fully connected layers shown as columns of circles.
   - First Layer: Column of green circles (512 neurons)
     - Label: "FC1: 2048→512"
   - Second Layer: Column of teal circles (256 neurons)
     - Label: "FC2: 512→256"
   - Output Layer: Column of red circles (9 neurons)
     - Label: "Output: 256→9"
     - Add small text labels: "ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM"

Connections:
- Use dotted gray lines projecting from small square patches to visualize convolution kernels
- Show residual connections as curved gray arrows that skip over bottleneck blocks
- Dense mesh of thin blue lines connecting the fully connected layers
- Include skip connection addition operation symbol (+) where residual paths merge

Annotations:
- Include curly brackets above groups of layers labeled with operation names
- Add dimension labels below each major block
- Highlight the bottleneck structure within each residual block (1×1→3×3→1×1)
- Add text labels for key architectural features: "Residual Connections", "Bottleneck Design", "Feature Extractor", "Classifier Head"
- Include small icons showing the histopathology classes at the output

Technical Details to Emphasize:
- The characteristic ResNet residual loop pattern
- The bottleneck architecture (1×1, 3×3, 1×1)
- Progressive reduction in spatial dimensions and increase in channel depth
- Custom classifier head with batch normalization and dropout
- Total parameter count: 24.7M
- Input size annotation: "B×3×224×224"

Vibe: Educational, precise, high-contrast, informative, professional medical AI architecture diagram suitable for academic papers or technical documentation.