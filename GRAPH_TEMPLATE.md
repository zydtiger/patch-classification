Subject: A technical architecture diagram of a neural network, specifically [Insert Network Type, e.g., a Convolutional Neural Network for image classification].

Style: Academic textbook illustration, clean vector style, schematic, white background.

Composition (Left to Right):

1.  Input: A [size, e.g., 28x28] square image on the far left containing [subject, e.g., a handwritten digit].
2.  Feature Extraction Phase: Display a series of [Number] blocks representing convolutional layers. Render these as **isometric stacks of 2D planes** to show channel depth.
    - Color: Light slate blue with thin black outlines.
    - Connection: Use **dotted gray lines** projecting from a small square patch on the previous layer to a single point on the next layer to visualize the kernel window.
3.  Transition: A simplified "Flattening" layer represented as a long, diagonal rectangular bar (color: peach/orange).
4.  Classification Phase: A Fully Connected network section.
    - Hidden Layers: Columns of green circles.
    - Output Layer: A column of red circles.
    - Connection: A dense mesh of thin blue lines connecting the circles.

Annotations:

- Include **curly brackets** above groups of layers labeled with operation names (e.g., "Convolution", "Max-Pooling") and kernel sizes.
- Include dimension labels below each block (e.g., "$W \times H \times C$").

Vibe: Educational, precise, high-contrast, informative.
