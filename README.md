# IO-ML-CL
Institute of Optics' module on continuous learning for Machine Learning algorithms.


# Class Incremental Learning with Rehearsal and Knowledge Distillation

## Project Overview

This project implements class-incremental learning strategies for image classification using the GTSRB (German Traffic Sign Recognition Benchmark) dataset. The core challenge addressed is "catastrophic forgetting" - the tendency of neural networks to forget previously learned classes when training on new classes. 

The project compares three different learning strategies:
1. **Fine-tuning** - The simplest approach that leads to catastrophic forgetting
2. **Rehearsal + Knowledge Distillation (KD)** - A combined approach to retain knowledge of previous classes
3. **Upper bound** - Non-incremental learning using all data at once (optimal target performance)

## The Catastrophic Forgetting Problem

When neural networks are trained sequentially on different tasks or classes, they tend to overwrite previously learned knowledge. This project demonstrates this problem using the GTSRB dataset, where traffic sign classes are introduced 8 at a time across 5 tasks.

## Solution Strategies

### Fine-tuning
The simplest approach that just applies gradient descent on new data. This approach serves as a baseline and clearly shows the catastrophic forgetting phenomenon.

### Rehearsal + Knowledge Distillation
This combined approach uses two techniques to mitigate forgetting:

1. **Memory Rehearsal**: Maintains a buffer of examples from previous classes to periodically revisit
   - Buffer size: 200 examples
   - Balanced class representation
   - Examples from previous tasks are mixed with current task examples during training

2. **Knowledge Distillation**: Transfers knowledge from the previous version of the model
   - Uses KL divergence loss between softened outputs of current and previous model
   - Temperature scaling is applied to soften distributions
   - Balances classification loss and distillation loss

## Implementation Details

### Dataset
- GTSRB (German Traffic Sign Recognition Benchmark)
- 43 different traffic sign classes
- 32x32 pixel images
- Split into 5 tasks with 8-9 classes per task

### Model Architecture
- Simple CNN architecture
- Convolutional layers with leaky ReLU activations
- Expandable output layer that grows with each new task

### Key Features
- Memory buffer implementation with balanced class storage
- Knowledge distillation with temperature scaling
- Incremental expansion of the classifier layer
- Performance tracking with Weights & Biases integration

## Results

The project demonstrates how the combined rehearsal and knowledge distillation approach significantly outperforms simple fine-tuning, maintaining much higher accuracy across all tasks. The results show:

1. **Upper bound**: Maintains consistently high accuracy (~85%)
2. **Rehearsal + KD**: Shows gradual decline but maintains reasonable performance (~55-70%)
3. **Fine-tuning**: Shows catastrophic forgetting, with accuracy dropping to near zero

![image](https://github.com/user-attachments/assets/60590462-22b6-41e0-a6b7-4a9343abcf10)

## How to Run

1. Install dependencies:
```bash
pip install torch torchvision tqdm wandb torchinfo matplotlib pandas
```

2. Run the main script:
```bash
python main.py
```

3. To visualize results, check the Weights & Biases dashboard.

## Key Parameters
- `buffer_size`: Size of memory buffer (default: 200)
- `alpha`: Weight for balancing classification and distillation loss (default: 0.5)
- `T`: Temperature for knowledge distillation (default: 2.0)
- `classes_per_task`: Number of classes per incremental task (default: 8)
- `num_tasks`: Total number of sequential tasks (default: 5)

## Code Structure
- Memory Buffer class for storing and sampling examples from previous tasks
- Knowledge Distillation loss implementation
- Training loops with rehearsal and distillation
- Performance evaluation functions
- Weights & Biases integration for experiment tracking

## Useful links
[Colab](https://colab.research.google.com/github/stepherbin/teaching/blob/master/IOGS/projet/Code_snipets_for_class_incremental_GTSRB.ipynb)

[Courses and slides](https://sites.google.com/view/neurips2022-llm-tutorial)

[Survey](https://github.com/zhoudw-zdw/CIL_Survey/)

[GMVandeVen](http://github.com/GMvandeVen/continual-learning)

[Gdumb](https://github.com/drimpossible/GDumb)

[Paper list](https://github.com/ContinualAI/continual-learning-papers)

