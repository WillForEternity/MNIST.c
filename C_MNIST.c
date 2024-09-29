// C_MNIST using CSV formatted dataset

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

// Constants for network architecture and training
#define MAX_LAYERS 10                // Maximum number of layers in the neural network
#define MAX_NEURONS_PER_LAYER 512    // Maximum neurons in any single layer
#define INPUT_SIZE 784               // Size of input layer (28x28 pixels)
#define OUTPUT_SIZE 10               // Number of output classes (digits 0-9)


#define NUM_TRAINING_SAMPLES 60000   // Number of training samples in MNIST CSV
#define NUM_TEST_SAMPLES 10000       // Number of test samples in MNIST CSV
#define EPOCHS 10                    // Number of training epochs
#define BATCH_SIZE 1500              // Size of each mini-batch, adjusted to reduce batches per epoch
#define BATCH_PRINT_INTERVAL 10      // Interval at which batch results are printed
#define PRINT_IMAGE_SAMPLES 5        // Number of images to print per batch

// Learning rate parameters
#define INITIAL_LEARNING_RATE 0.01   // Starting learning rate
#define DECAY_RATE 0.01              // Rate at which learning rate decays
#define MIN_LEARNING_RATE 0.0001     // Minimum learning rate

// Neural Network structure definition
typedef struct {
    int num_layers;                          // Total number of layers in the network
    int layer_sizes[MAX_LAYERS];             // Array to store the size of each layer
    double **weights;                        // 2D array for weights between layers
    double **biases;                         // 2D array for biases of each neuron
    double **activations;                    // 2D array for neuron activations
    double **z_values;                       // 2D array for weighted sums before activation
} NeuralNetwork;

// Function declarations
double adjust_learning_rate(double current_rate, int epoch);
void init_network(NeuralNetwork* nn, int num_layers, int* layer_sizes);
void free_network(NeuralNetwork* nn);
void forward_propagation(NeuralNetwork* nn, double* input);
void backward_propagation(NeuralNetwork* nn, double* target, double learning_rate);
void* safe_malloc(size_t size);
double relu(double x);
double relu_derivative(double x);
void softmax(double* input, int length);
void load_csv_data(const char* filename, double*** images, int** labels, int num_samples, int image_size);
void shuffle_data(double** images, int* labels, int n);
void print_image(double* image, int width, int height);
void countdown(int seconds);

int main() {
    // Seed the random number generator
    srand(time(NULL));

    // Load MNIST training data from CSV files
    double** train_images;
    int* train_labels;
    load_csv_data("mnist_train.csv", &train_images, &train_labels, NUM_TRAINING_SAMPLES, INPUT_SIZE);

    // Load MNIST test data from CSV files
    double** test_images;
    int* test_labels;
    load_csv_data("mnist_test.csv", &test_images, &test_labels, NUM_TEST_SAMPLES, INPUT_SIZE);

    // Define the network architecture
    int num_layers = 4;
    int layer_sizes[] = {INPUT_SIZE, 128, 64, OUTPUT_SIZE};

    // Declare and initialize the neural network
    NeuralNetwork nn;
    init_network(&nn, num_layers, layer_sizes);

    // Set the initial learning rate
    double learning_rate = INITIAL_LEARNING_RATE;

    // Training parameters
    int epochs = EPOCHS;
    int batch_size = BATCH_SIZE;
    int num_batches = NUM_TRAINING_SAMPLES / batch_size;

    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        int correct = 0;

        // Shuffle the training data
        shuffle_data(train_images, train_labels, NUM_TRAINING_SAMPLES);

        for (int batch = 0; batch < num_batches; batch++) {
            // Initialize batch loss and correct count
            double batch_total_loss = 0.0;
            int batch_correct = 0;

            // Process each mini-batch
            for (int i = 0; i < batch_size; i++) {
                int index = batch * batch_size + i;
                double* input = train_images[index];
                int label = train_labels[index];

                // Perform forward propagation
                forward_propagation(&nn, input);

                // Create the target output vector (one-hot encoding)
                double target[OUTPUT_SIZE] = {0.0};
                target[label] = 1.0;

                // Calculate the loss for this example (cross-entropy loss)
                double loss = 0.0;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    if (target[j] == 1) {
                        loss -= log(nn.activations[num_layers - 1][j] + 1e-15);  // Prevent log(0)
                    }
                }
                total_loss += loss;
                batch_total_loss += loss;

                // Perform backward propagation to update weights and biases
                backward_propagation(&nn, target, learning_rate);

                // Determine the network's prediction
                int predicted_label = 0;
                double max_activation = nn.activations[num_layers - 1][0];
                for (int j = 1; j < OUTPUT_SIZE; j++) {
                    if (nn.activations[num_layers - 1][j] > max_activation) {
                        max_activation = nn.activations[num_layers - 1][j];
                        predicted_label = j;
                    }
                }

                // Increment correct count if prediction matches label
                if (predicted_label == label) {
                    correct++;
                    batch_correct++;
                }

                // Print the image and labels for the first few samples in each batch
                if (i < PRINT_IMAGE_SAMPLES) {
                    printf("Epoch %d, Batch %d, Sample %d:\n", epoch + 1, batch + 1, i + 1);
                    print_image(input, 28, 28);
                    printf("True Label: %d, Predicted Label: %d\n\n", label, predicted_label);
                }
            }

            // After processing the batch, check if we need to print batch results
            if (batch < 10 || (batch + 1) % BATCH_PRINT_INTERVAL == 0) {
                double batch_accuracy = (double)batch_correct / batch_size * 100;
                double batch_avg_loss = batch_total_loss / batch_size;

                printf("    - Batch %d, Accuracy: %.2f%%, Avg Loss: %.6f\n",
                       batch + 1, batch_accuracy, batch_avg_loss);
            }
        }

        // Calculate average loss for this epoch
        double avg_loss = total_loss / NUM_TRAINING_SAMPLES;

        // Adjust the learning rate based on the epoch
        learning_rate = adjust_learning_rate(learning_rate, epoch);

        // Print the formatted results for this epoch
        printf("\n\n========================================\n");
        printf("           Epoch %d Summary\n", epoch + 1);
        printf("========================================\n");
        printf("Accuracy       : %.2f%%\n", (double)correct / NUM_TRAINING_SAMPLES * 100);
        printf("Average Loss   : %.6f\n", avg_loss);
        printf("Learning Rate  : %.6f\n", learning_rate);
        printf("========================================\n");

        if (epoch < epochs - 1) {
        printf("Next epoch starting in: ");
        countdown(10);
        printf("\n\n");
    }
    }

    // Testing phase
    printf("\nTesting the neural network:\n\n");
    int correct = 0;

    for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
        double* input = test_images[i];
        int label = test_labels[i];

        // Perform forward propagation to get the network's prediction
        forward_propagation(&nn, input);

        // Determine the prediction
        int predicted_label = 0;
        double max_activation = nn.activations[num_layers - 1][0];
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (nn.activations[num_layers - 1][j] > max_activation) {
                max_activation = nn.activations[num_layers - 1][j];
                predicted_label = j;
            }
        }

        // Increment correct count if prediction matches label
        if (predicted_label == label) correct++;
    }

    // Print the overall test accuracy
    printf("Test Accuracy: %.2f%%\n", (double)correct / NUM_TEST_SAMPLES * 100);

    // Free the memory allocated for the neural network and data
    free_network(&nn);

    for (int i = 0; i < NUM_TRAINING_SAMPLES; i++) {
        free(train_images[i]);
    }
    free(train_images);
    free(train_labels);

    for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
        free(test_images[i]);
    }
    free(test_images);
    free(test_labels);

    return 0;
}

void countdown(int seconds) {
    for (int i = seconds; i > 0; i--) {
        printf("%d...", i);
        fflush(stdout);
        sleep(1);
    }
    printf("\n");
}

// Function to adjust the learning rate based on the current epoch
double adjust_learning_rate(double current_rate, int epoch) {
    // Calculate the new learning rate using the decay formula
    double new_rate = INITIAL_LEARNING_RATE / (1 + DECAY_RATE * epoch);

    // Ensure the new learning rate does not fall below the minimum learning rate
    if (new_rate < MIN_LEARNING_RATE) {
        new_rate = MIN_LEARNING_RATE;
    }

    // Return the adjusted learning rate
    return new_rate;
}

// Function to initialize the neural network
void init_network(NeuralNetwork* nn, int num_layers, int* layer_sizes) {
    // Check if the number of layers is valid
    if (num_layers < 2 || num_layers > MAX_LAYERS) {
        fprintf(stderr, "Invalid number of layers\n");
        exit(EXIT_FAILURE);
    }

    // Set the number of layers in the network structure
    nn->num_layers = num_layers;
    // Copy the layer sizes array into the network structure
    memcpy(nn->layer_sizes, layer_sizes, num_layers * sizeof(int));

    // Allocate memory for the weights, biases, activations, and z_values arrays
    nn->weights = (double**)safe_malloc((num_layers - 1) * sizeof(double*));
    nn->biases = (double**)safe_malloc((num_layers - 1) * sizeof(double*));
    nn->activations = (double**)safe_malloc(num_layers * sizeof(double*));
    nn->z_values = (double**)safe_malloc(num_layers * sizeof(double*));

    // Initialize weights and biases
    for (int i = 0; i < num_layers - 1; i++) {
        int current_layer_size = layer_sizes[i];
        int next_layer_size = layer_sizes[i + 1];

        // Allocate memory for weights and biases between current layer and next layer
        nn->weights[i] = (double*)safe_malloc(current_layer_size * next_layer_size * sizeof(double));
        nn->biases[i] = (double*)safe_malloc(next_layer_size * sizeof(double));

        // He initialization for weights
        double std_dev = sqrt(2.0 / current_layer_size);
        for (int j = 0; j < current_layer_size * next_layer_size; j++) {
            // Initialize weights with normal distribution (approximate using Box-Muller transform)
            double u1 = (double)rand() / RAND_MAX;
            double u2 = (double)rand() / RAND_MAX;
            double z = sqrt(-2.0 * log(u1 + 1e-9)) * cos(2.0 * M_PI * u2);
            nn->weights[i][j] = z * std_dev;
        }

        // Initialize biases to small positive values
        for (int j = 0; j < next_layer_size; j++) {
            nn->biases[i][j] = 0.01;
        }
    }

    // Allocate memory for activations and z_values
    for (int i = 0; i < num_layers; i++) {
        int layer_size = layer_sizes[i];
        nn->activations[i] = (double*)safe_malloc(layer_size * sizeof(double));
        nn->z_values[i] = (double*)safe_malloc(layer_size * sizeof(double));
    }
}

// Function to free all dynamically allocated memory for the neural network
void free_network(NeuralNetwork* nn) {
    // Free weights and biases
    for (int i = 0; i < nn->num_layers - 1; i++) {
        free(nn->weights[i]);
        free(nn->biases[i]);
    }

    // Free activations and z_values
    for (int i = 0; i < nn->num_layers; i++) {
        free(nn->activations[i]);
        free(nn->z_values[i]);
    }

    // Free arrays of pointers
    free(nn->weights);
    free(nn->biases);
    free(nn->activations);
    free(nn->z_values);
}

// Function to perform forward propagation through the neural network
void forward_propagation(NeuralNetwork* nn, double* input) {
    // Copy the input data to the activations of the input layer
    memcpy(nn->activations[0], input, nn->layer_sizes[0] * sizeof(double));

    // Loop through each layer starting from the first hidden layer
    for (int l = 1; l < nn->num_layers; l++) {
        int current_layer_size = nn->layer_sizes[l];
        int previous_layer_size = nn->layer_sizes[l - 1];

        // Compute activations for the current layer
        for (int j = 0; j < current_layer_size; j++) {
            double z = nn->biases[l - 1][j];  // Start with the bias

            // Sum the weighted activations from the previous layer
            for (int k = 0; k < previous_layer_size; k++) {
                double weight = nn->weights[l - 1][k * current_layer_size + j];
                double activation = nn->activations[l - 1][k];
                z += weight * activation;
            }

            nn->z_values[l][j] = z;  // Store the weighted sum before activation

            // Apply activation function
            if (l == nn->num_layers - 1) {
                // Output layer - will apply softmax after computing all z-values
                // Temporarily store z; softmax will be applied later
            } else {
                // Hidden layers - ReLU activation
                nn->activations[l][j] = relu(z);
            }
        }

        // If this is the output layer, apply softmax to the entire layer
        if (l == nn->num_layers - 1) {
            // Copy z_values to activations to apply softmax
            memcpy(nn->activations[l], nn->z_values[l], current_layer_size * sizeof(double));
            softmax(nn->activations[l], current_layer_size);
        }
    }
}

// Function to perform backward propagation through the neural network
void backward_propagation(NeuralNetwork* nn, double* target, double learning_rate) {
    int num_layers = nn->num_layers;
    int* layer_sizes = nn->layer_sizes;
    double** activations = nn->activations;
    double** weights = nn->weights;
    double** biases = nn->biases;
    double** z_values = nn->z_values;

    // Allocate memory for delta values for each layer
    double** deltas = (double**)safe_malloc(num_layers * sizeof(double*));
    for (int l = 0; l < num_layers; l++) {
        deltas[l] = (double*)safe_malloc(layer_sizes[l] * sizeof(double));
    }

    // Step 1: Compute delta for the output layer
    int L = num_layers - 1;  // Index of the output layer
    for (int i = 0; i < layer_sizes[L]; i++) {
        double a = activations[L][i];  // Activation of the output neuron
        double y = target[i];          // Target value (one-hot encoded)
        // For softmax with cross-entropy loss, delta = a - y
        deltas[L][i] = a - y;
    }

    // Step 2: Backpropagate the error to hidden layers
    for (int l = L - 1; l > 0; l--) {
        int current_layer_size = layer_sizes[l];
        int next_layer_size = layer_sizes[l + 1];

        for (int i = 0; i < current_layer_size; i++) {
            double sum = 0.0;
            // Compute the weighted sum of deltas from the next layer
            for (int j = 0; j < next_layer_size; j++) {
                double weight = weights[l][i * next_layer_size + j];
                sum += weight * deltas[l + 1][j];
            }
            double z = z_values[l][i];  // Pre-activation value

            // Compute the delta for current layer neurons
            deltas[l][i] = sum * relu_derivative(z);
        }
    }

    // Step 3: Update weights and biases
    for (int l = 0; l < num_layers - 1; l++) {
        int current_layer_size = layer_sizes[l];
        int next_layer_size = layer_sizes[l + 1];

        // Update weights between layer l and l+1
        for (int i = 0; i < current_layer_size; i++) {
            for (int j = 0; j < next_layer_size; j++) {
                double delta = deltas[l + 1][j];
                double a = activations[l][i];
                // Gradient descent weight update
                weights[l][i * next_layer_size + j] -= learning_rate * a * delta;
            }
        }
        // Update biases for layer l+1
        for (int j = 0; j < next_layer_size; j++) {
            biases[l][j] -= learning_rate * deltas[l + 1][j];
        }
    }

    // Step 4: Free allocated memory for deltas
    for (int l = 0; l < num_layers; l++) {
        free(deltas[l]);
    }
    free(deltas);
}

// Safe memory allocation function
void* safe_malloc(size_t size) {
    // Allocate memory of the specified size
    void* ptr = malloc(size);

    // Check if memory allocation was successful
    if (ptr == NULL) {
        // If allocation failed, print an error message and exit the program
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Return the pointer to the allocated memory
    return ptr;
}

// ReLU activation function
double relu(double x) {
    return x > 0 ? x : 0;
}

// Derivative of the ReLU function
double relu_derivative(double x) {
    // x is the pre-activation value z
    return x > 0 ? 1.0 : 0.0;
}

// Softmax activation function
void softmax(double* input, int length) {
    double max = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max) {
            max = input[i];
        }
    }
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        input[i] = exp(input[i] - max);  // Subtract max for numerical stability
        sum += input[i];
    }
    for (int i = 0; i < length; i++) {
        input[i] /= sum;
    }
}

// Function to load MNIST data from a CSV file
void load_csv_data(const char* filename, double*** images, int** labels, int num_samples, int image_size) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Cannot open file: %s\n", filename);
        exit(1);
    }

    // Allocate memory for images and labels
    *images = (double**)malloc(num_samples * sizeof(double*));
    *labels = (int*)malloc(num_samples * sizeof(int));

    char line[16384];  // Adjust size according to the expected line length
    int sample = 0;
    while (fgets(line, sizeof(line), fp) && sample < num_samples) {
        char* token = strtok(line, ",");
        (*labels)[sample] = atoi(token);  // First value is the label

        (*images)[sample] = (double*)malloc(image_size * sizeof(double));
        int pixel = 0;
        while ((token = strtok(NULL, ",")) && pixel < image_size) {
            (*images)[sample][pixel] = atoi(token) / 255.0;  // Normalize pixel value
            pixel++;
        }
        sample++;
    }

    fclose(fp);
}

// Function to shuffle the training data
void shuffle_data(double** images, int* labels, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        // Swap images
        double* temp_image = images[i];
        images[i] = images[j];
        images[j] = temp_image;

        // Swap labels
        int temp_label = labels[i];
        labels[i] = labels[j];
        labels[j] = temp_label;
    }
}

// Function to print the image in ASCII format
void print_image(double* image, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            double pixel_value = image[i * width + j];
            if (pixel_value > 0.75) {
                printf("@");
            } else if (pixel_value > 0.5) {
                printf("#");
            } else if (pixel_value > 0.25) {
                printf("+");
            } else {
                printf(" ");
            }
        }
        printf("\n");
    }
}
